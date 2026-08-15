"""SQLite persistence for jobs and results.

A connection is opened per operation rather than shared. SQLite connections are
not safely shareable across threads, and the worker pool plus the API event loop
means several threads touch this store. WAL mode keeps concurrent readers from
blocking the writer, which is all the concurrency an MVP needs.

The full result lives in one `result_json` blob, but the handful of fields the
library UI needs to draw a card — title, duration, dimensions, counts — are
also written to their own columns when a result is saved. That is deliberate
denormalisation: rendering a page of the library would otherwise mean parsing
and validating fifty result documents, and searching it would mean `LIKE` over
JSON. Both are cheap to keep correct because `save_result` is the only writer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from video_understanding.core.errors import JobNotFoundError
from video_understanding.core.models import (
    JobStatus,
    ProcessingStage,
    VideoJob,
    VideoSummary,
    VideoUnderstanding,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    video_id      TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    status        TEXT NOT NULL,
    stage         TEXT NOT NULL,
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT,
    source_path   TEXT,
    result_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""

# Columns added after the first release. Applied with ALTER TABLE on open, which
# is how SQLite does migrations when there is no migration framework: additive,
# idempotent, and safe to run against a database written by an older build.
MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("title", "TEXT"),
    ("summary", "TEXT"),
    ("duration", "REAL"),
    ("width", "INTEGER"),
    ("height", "INTEGER"),
    ("size_bytes", "INTEGER"),
    ("scene_count", "INTEGER"),
    ("highlight_count", "INTEGER"),
    ("transcript_count", "INTEGER"),
    ("has_audio", "INTEGER"),
    ("topics_json", "TEXT"),
)

# `q=` searches these. Filename is included because it is what the user
# recognises before the analysis has given the video a title.
SEARCH_COLUMNS = ("filename", "title", "summary", "topics_json")

SORTS: dict[str, str] = {
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
    # `duration IS NULL` first keeps unanalysed rows at the bottom without
    # relying on NULLS LAST, which older SQLite builds do not parse.
    "longest": "duration IS NULL, duration DESC, created_at DESC",
    "shortest": "duration IS NULL, duration ASC, created_at DESC",
    "name": "filename COLLATE NOCASE ASC",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class Database:
    """Job store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self.init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate(connection)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Add any columns this build expects but the file does not have."""
        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        added = [name for name, _ in MIGRATIONS if name not in existing]
        for name, column_type in MIGRATIONS:
            if name in existing:
                continue
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {column_type}")  # noqa: S608

        if added:
            logger.info("migrated jobs table: added %s", ", ".join(added))
            self._backfill_summaries(connection)

    def _backfill_summaries(self, connection: sqlite3.Connection) -> None:
        """Populate the new card columns for videos analysed by an older build.

        Runs once, immediately after the columns appear. Parsing every stored
        result is affordable exactly because it never happens again.
        """
        rows = connection.execute(
            "SELECT video_id, result_json FROM jobs WHERE result_json IS NOT NULL"
        ).fetchall()

        for row in rows:
            try:
                result = VideoUnderstanding.model_validate(json.loads(row["result_json"]))
            except (json.JSONDecodeError, ValueError):
                continue
            fields = _summary_fields(result)
            connection.execute(
                f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE video_id = ?",  # noqa: S608
                (*fields.values(), row["video_id"]),
            )

        if rows:
            logger.info("backfilled library fields for %d existing video(s)", len(rows))

    # --- Writes -------------------------------------------------------------

    def create_job(self, job: VideoJob) -> VideoJob:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    video_id, filename, status, stage, progress, error,
                    created_at, updated_at, started_at, completed_at,
                    source_path, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.video_id,
                    job.filename,
                    job.status.value,
                    job.stage.value,
                    job.progress,
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.source_path,
                    None,
                ),
            )
        logger.info("created job %s (%s)", job.video_id, job.filename)
        return job

    def update_status(
        self,
        video_id: str,
        *,
        status: JobStatus | None = None,
        stage: ProcessingStage | None = None,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        """Patch the mutable fields of a job. Absent arguments are left alone."""
        assignments: list[str] = ["updated_at = ?"]
        values: list[object] = [_now()]

        if status is not None:
            assignments.append("status = ?")
            values.append(status.value)
            if status is JobStatus.PROCESSING:
                assignments.append("started_at = COALESCE(started_at, ?)")
                values.append(_now())
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                assignments.append("completed_at = ?")
                values.append(_now())
        if stage is not None:
            assignments.append("stage = ?")
            values.append(stage.value)
            if progress is None:
                progress = stage.progress
        if progress is not None:
            assignments.append("progress = ?")
            values.append(max(0.0, min(1.0, progress)))
        if error is not None:
            assignments.append("error = ?")
            values.append(error)

        values.append(video_id)
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE video_id = ?",  # noqa: S608
                values,
            )
            if cursor.rowcount == 0:
                raise JobNotFoundError(f"no job with id '{video_id}'")

    def save_result(self, video_id: str, result: VideoUnderstanding) -> None:
        fields = _summary_fields(result)
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE jobs
                   SET result_json = ?, status = ?, stage = ?, progress = 1.0,
                       updated_at = ?, completed_at = ?, error = NULL,
                       {', '.join(f'{k} = ?' for k in fields)}
                 WHERE video_id = ?
                """,  # noqa: S608 - column names come from a module constant
                (
                    result.model_dump_json(),
                    JobStatus.COMPLETED.value,
                    ProcessingStage.DONE.value,
                    _now(),
                    _now(),
                    *fields.values(),
                    video_id,
                ),
            )
            if cursor.rowcount == 0:
                raise JobNotFoundError(f"no job with id '{video_id}'")
        logger.info("saved result for job %s", video_id)

    def mark_failed(self, video_id: str, error: str) -> None:
        self.update_status(video_id, status=JobStatus.FAILED, error=error)

    def delete_job(self, video_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE video_id = ?", (video_id,))
            return cursor.rowcount > 0

    # --- Reads --------------------------------------------------------------

    def get_job(self, video_id: str) -> VideoJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE video_id = ?", (video_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def require_job(self, video_id: str) -> VideoJob:
        job = self.get_job(video_id)
        if job is None:
            raise JobNotFoundError(f"no job with id '{video_id}'")
        return job

    def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        query: str | None = None,
        status: JobStatus | None = None,
        sort: str = "newest",
    ) -> list[VideoJob]:
        where, values = _filter_clause(query, status)
        order = SORTS.get(sort, SORTS["newest"])
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs {where} ORDER BY {order} LIMIT ? OFFSET ?",  # noqa: S608
                (*values, limit, offset),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def list_summaries(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        query: str | None = None,
        status: JobStatus | None = None,
        sort: str = "newest",
    ) -> list[VideoSummary]:
        """List videos for the library, without parsing any stored result."""
        where, values = _filter_clause(query, status)
        order = SORTS.get(sort, SORTS["newest"])
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs {where} ORDER BY {order} LIMIT ? OFFSET ?",  # noqa: S608
                (*values, limit, offset),
            ).fetchall()
        return [_row_to_summary(row) for row in rows]

    def count_jobs(self, *, query: str | None = None, status: JobStatus | None = None) -> int:
        where, values = _filter_clause(query, status)
        with self._connect() as connection:
            return connection.execute(
                f"SELECT COUNT(*) FROM jobs {where}", values  # noqa: S608
            ).fetchone()[0]

    def status_counts(self) -> dict[str, int]:
        """How many videos sit in each status, for the library's filter chips."""
        counts = {status.value: 0 for status in JobStatus}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[row["status"]] = row["n"]
        counts["all"] = sum(counts.values())
        return counts

    def reset_stale_jobs(self) -> int:
        """Fail jobs left mid-flight by a previous process.

        Work happens in-process, so anything still marked `processing` at
        startup died with the last run and will never make progress.
        """
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                   SET status = ?, error = ?, updated_at = ?, completed_at = ?
                 WHERE status IN (?, ?)
                """,
                (
                    JobStatus.FAILED.value,
                    "Processing was interrupted by a server restart. Please re-upload.",
                    _now(),
                    _now(),
                    JobStatus.PROCESSING.value,
                    JobStatus.QUEUED.value,
                ),
            )
            count = cursor.rowcount
        if count:
            logger.warning("failed %d stale job(s) left over from a previous run", count)
        return count

    def healthy(self) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False


def _filter_clause(query: str | None, status: JobStatus | None) -> tuple[str, list[object]]:
    """Build the shared WHERE clause for listing, counting and searching."""
    clauses: list[str] = []
    values: list[object] = []

    if status is not None:
        clauses.append("status = ?")
        values.append(status.value)

    term = (query or "").strip()
    if term:
        # Escape the LIKE wildcards so a search for "100%" means what it says.
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        conditions = " OR ".join(
            f"{column} LIKE ? ESCAPE '\\'" for column in SEARCH_COLUMNS
        )
        clauses.append(f"({conditions})")
        values.extend([pattern] * len(SEARCH_COLUMNS))

    return (f"WHERE {' AND '.join(clauses)}" if clauses else "", values)


def _summary_fields(result: VideoUnderstanding) -> dict[str, object]:
    """The denormalised card columns derived from a completed analysis."""
    metadata = result.metadata
    return {
        "title": result.title,
        "summary": result.summary,
        "duration": result.duration,
        "width": metadata.width if metadata else None,
        "height": metadata.height if metadata else None,
        "size_bytes": metadata.size_bytes if metadata else None,
        "scene_count": len(result.scenes),
        "highlight_count": len(result.highlights),
        "transcript_count": len(result.transcript),
        "has_audio": int(metadata.has_audio) if metadata else None,
        "topics_json": json.dumps(result.topics) if result.topics else None,
    }


def _row_value(row: sqlite3.Row, key: str, default: object = None) -> object:
    """Read a column that may predate this build's schema."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _row_to_summary(row: sqlite3.Row) -> VideoSummary:
    topics_json = _row_value(row, "topics_json")
    topics: list[str] = []
    if isinstance(topics_json, str):
        try:
            parsed = json.loads(topics_json)
            topics = [str(t) for t in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            topics = []

    return VideoSummary(
        video_id=row["video_id"],
        filename=row["filename"],
        status=JobStatus(row["status"]),
        stage=ProcessingStage(row["stage"]),
        progress=row["progress"],
        error=row["error"],
        created_at=_parse_time(row["created_at"]) or datetime.now(UTC),
        updated_at=_parse_time(row["updated_at"]) or datetime.now(UTC),
        title=_row_value(row, "title") or None,
        summary=_row_value(row, "summary") or "",
        duration=_row_value(row, "duration"),
        width=_row_value(row, "width"),
        height=_row_value(row, "height"),
        size_bytes=_row_value(row, "size_bytes"),
        scene_count=_row_value(row, "scene_count", 0),
        highlight_count=_row_value(row, "highlight_count", 0),
        transcript_count=_row_value(row, "transcript_count", 0),
        has_audio=bool(_row_value(row, "has_audio", 0)),
        topics=topics,
    )


def _row_to_job(row: sqlite3.Row) -> VideoJob:
    result = None
    if row["result_json"]:
        try:
            result = VideoUnderstanding.model_validate(json.loads(row["result_json"]))
        except (json.JSONDecodeError, ValueError) as exc:
            # A stored result that no longer validates should not make the job
            # unreadable; surface the job without its result.
            logger.error("stored result for %s is unreadable: %s", row["video_id"], exc)

    return VideoJob(
        video_id=row["video_id"],
        filename=row["filename"],
        status=JobStatus(row["status"]),
        stage=ProcessingStage(row["stage"]),
        progress=row["progress"],
        error=row["error"],
        created_at=_parse_time(row["created_at"]) or datetime.now(UTC),
        updated_at=_parse_time(row["updated_at"]) or datetime.now(UTC),
        started_at=_parse_time(row["started_at"]),
        completed_at=_parse_time(row["completed_at"]),
        source_path=row["source_path"],
        result=result,
    )
