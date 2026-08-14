"""SQLite persistence for jobs and results.

A connection is opened per operation rather than shared. SQLite connections are
not safely shareable across threads, and the worker pool plus the API event loop
means several threads touch this store. WAL mode keeps concurrent readers from
blocking the writer, which is all the concurrency an MVP needs.
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
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                   SET result_json = ?, status = ?, stage = ?, progress = 1.0,
                       updated_at = ?, completed_at = ?, error = NULL
                 WHERE video_id = ?
                """,
                (
                    result.model_dump_json(),
                    JobStatus.COMPLETED.value,
                    ProcessingStage.DONE.value,
                    _now(),
                    _now(),
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

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[VideoJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def count_jobs(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

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
