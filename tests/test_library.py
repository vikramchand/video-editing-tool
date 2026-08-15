"""The library: denormalised summaries, search, filtering and thumbnails.

These cover the read model the web UI browses, which is deliberately separate
from the per-job record: it must survive an old database file, a video whose
file has moved, and a search term containing SQL wildcards.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_ffmpeg
from video_understanding.api.app import create_app
from video_understanding.core.models import JobStatus, VideoJob, VideoUnderstanding
from video_understanding.storage.database import SCHEMA, Database
from video_understanding.storage.files import resolve_source
from video_understanding.video import thumbnails


@pytest.fixture
def database(tmp_path) -> Database:
    return Database(tmp_path / "library.db")


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _job(video_id: str, filename: str, status: JobStatus = JobStatus.QUEUED) -> VideoJob:
    return VideoJob(video_id=video_id, filename=filename, status=status)


class TestSummaries:
    def test_summary_columns_are_written_on_save(self, database, understanding):
        database.create_job(_job("aaa", "kitchen.mp4"))
        database.save_result("aaa", understanding)

        summary = database.list_summaries()[0]
        assert summary.title == "Kitchen Trip"
        assert summary.duration == pytest.approx(30.0)
        assert summary.width == 1080
        assert summary.scene_count == 3
        assert summary.highlight_count == 1
        assert summary.transcript_count == 3
        assert summary.has_audio is True

    def test_summary_is_readable_before_any_result_exists(self, database):
        database.create_job(_job("bbb", "pending.mp4"))
        summary = database.list_summaries()[0]

        assert summary.status is JobStatus.QUEUED
        assert summary.duration is None
        assert summary.scene_count == 0

    def test_listing_does_not_parse_the_result_blob(self, database, understanding, monkeypatch):
        """The point of the extra columns: a page of the library is columns only."""
        database.create_job(_job("ccc", "clip.mp4"))
        database.save_result("ccc", understanding)

        def explode(*args, **kwargs):
            raise AssertionError("list_summaries must not validate stored results")

        monkeypatch.setattr(VideoUnderstanding, "model_validate", explode)
        assert database.list_summaries()[0].title == "Kitchen Trip"

    def test_topics_round_trip(self, database, metadata):
        result = VideoUnderstanding(duration=5.0, metadata=metadata, topics=["cooking", "travel"])
        database.create_job(_job("ddd", "topics.mp4"))
        database.save_result("ddd", result)
        assert database.list_summaries()[0].topics == ["cooking", "travel"]


class TestSearchAndFilter:
    @pytest.fixture
    def populated(self, database, metadata) -> Database:
        for video_id, filename, title, topics in [
            ("v1", "beach_day.mp4", "Sunset at the Beach", ["travel"]),
            ("v2", "kitchen.mov", "Making Pasta", ["cooking"]),
            ("v3", "100%_battery.mp4", None, []),
        ]:
            database.create_job(_job(video_id, filename))
            database.save_result(
                video_id,
                VideoUnderstanding(
                    duration=10.0, metadata=metadata, title=title,
                    summary=f"A video called {filename}", topics=topics,
                ),
            )
        database.create_job(_job("v4", "queued.mp4"))
        return database

    def test_search_matches_the_filename(self, populated):
        assert [s.video_id for s in populated.list_summaries(query="kitchen")] == ["v2"]

    def test_search_matches_the_title(self, populated):
        assert [s.video_id for s in populated.list_summaries(query="pasta")] == ["v2"]

    def test_search_matches_a_topic(self, populated):
        assert [s.video_id for s in populated.list_summaries(query="travel")] == ["v1"]

    def test_search_is_case_insensitive(self, populated):
        assert [s.video_id for s in populated.list_summaries(query="BEACH")] == ["v1"]

    def test_wildcards_in_the_query_are_literal(self, populated):
        # Unescaped, `%` would match every row rather than the one file named
        # after it.
        assert [s.video_id for s in populated.list_summaries(query="100%")] == ["v3"]
        assert populated.count_jobs(query="%") == 1

    def test_status_filter(self, populated):
        assert [s.video_id for s in populated.list_summaries(status=JobStatus.QUEUED)] == ["v4"]

    def test_count_matches_the_same_filters(self, populated):
        assert populated.count_jobs(query="pasta") == 1
        assert populated.count_jobs(status=JobStatus.COMPLETED) == 3
        assert populated.count_jobs() == 4

    def test_status_counts(self, populated):
        counts = populated.status_counts()
        assert counts["completed"] == 3
        assert counts["queued"] == 1
        assert counts["failed"] == 0
        assert counts["all"] == 4

    def test_sorting(self, database, metadata):
        for video_id, filename, duration in [("s", "z.mp4", 5.0), ("l", "a.mp4", 50.0)]:
            database.create_job(_job(video_id, filename))
            database.save_result(
                video_id, VideoUnderstanding(duration=duration, metadata=metadata)
            )

        assert [s.video_id for s in database.list_summaries(sort="longest")] == ["l", "s"]
        assert [s.video_id for s in database.list_summaries(sort="shortest")] == ["s", "l"]
        assert [s.video_id for s in database.list_summaries(sort="name")] == ["l", "s"]

    def test_unknown_sort_falls_back_to_newest(self, populated):
        assert [s.video_id for s in populated.list_summaries(sort="bogus")] == \
            [s.video_id for s in populated.list_summaries(sort="newest")]


class TestMigration:
    def test_opening_a_pre_migration_database_adds_the_columns(self, tmp_path, understanding):
        path = tmp_path / "old.db"
        connection = sqlite3.connect(path)
        connection.executescript(SCHEMA)
        connection.execute(
            """
            INSERT INTO jobs (video_id, filename, status, stage, progress,
                              created_at, updated_at, result_json)
            VALUES ('old1', 'legacy.mp4', 'completed', 'done', 1.0,
                    '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00', ?)
            """,
            (understanding.model_dump_json(),),
        )
        connection.commit()
        connection.close()

        summary = Database(path).list_summaries()[0]

        # Backfilled from the blob that was already there, not left empty.
        assert summary.title == "Kitchen Trip"
        assert summary.scene_count == 3
        assert summary.duration == pytest.approx(30.0)

    def test_migration_is_idempotent(self, tmp_path, understanding):
        path = tmp_path / "twice.db"
        first = Database(path)
        first.create_job(_job("x", "x.mp4"))
        first.save_result("x", understanding)

        second = Database(path)  # re-opening must not re-run or fail
        assert second.list_summaries()[0].title == "Kitchen Trip"


class TestResolveSource:
    def test_prefers_the_recorded_path(self, tmp_path):
        recorded = tmp_path / "elsewhere" / "abc.mp4"
        recorded.parent.mkdir()
        recorded.write_bytes(b"x")
        assert resolve_source(str(recorded), "abc", tmp_path / "uploads") == recorded

    def test_falls_back_to_the_uploads_directory(self, tmp_path):
        """A path recorded inside a container has to resolve on the host."""
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        relocated = uploads / "abc.mp4"
        relocated.write_bytes(b"x")

        assert resolve_source("/app/data/uploads/abc.mp4", "abc", uploads) == relocated

    def test_falls_back_to_the_video_id_with_any_extension(self, tmp_path):
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        (uploads / "abc.mov").write_bytes(b"x")

        assert resolve_source("/gone/abc.mp4", "abc", uploads).name == "abc.mov"

    def test_returns_none_when_the_file_is_really_gone(self, tmp_path):
        assert resolve_source("/gone/abc.mp4", "abc", tmp_path) is None

    def test_handles_a_job_with_no_recorded_path(self, tmp_path):
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        (uploads / "abc.mp4").write_bytes(b"x")
        assert resolve_source(None, "abc", uploads).name == "abc.mp4"


class TestThumbnailTimestamp:
    def test_prefers_the_best_highlight(self, understanding):
        job = VideoJob(video_id="a", filename="a.mp4", result=understanding)
        assert thumbnails.choose_timestamp(job) == pytest.approx(15.0)

    def test_falls_back_to_the_most_important_scene(self, understanding):
        understanding.highlights = []
        job = VideoJob(video_id="a", filename="a.mp4", result=understanding)
        # Scene 2 (10-20) carries the highest importance.
        assert thumbnails.choose_timestamp(job) == pytest.approx(15.0)

    def test_falls_back_to_a_point_inside_a_short_video(self, metadata):
        result = VideoUnderstanding(duration=8.0, metadata=metadata)
        job = VideoJob(video_id="a", filename="a.mp4", result=result)
        timestamp = thumbnails.choose_timestamp(job)
        assert 0 < timestamp < 8.0

    def test_a_job_without_a_result_still_gets_a_timestamp(self):
        assert thumbnails.choose_timestamp(VideoJob(video_id="a", filename="a.mp4")) > 0

    def test_frame_paths_are_distinct_per_time_and_width(self, settings):
        first = thumbnails.frame_path("a", 1.5, 160, settings)
        assert first != thumbnails.frame_path("a", 1.6, 160, settings)
        assert first != thumbnails.frame_path("a", 1.5, 320, settings)
        assert first == thumbnails.frame_path("a", 1.5, 160, settings)

    def test_purge_removes_every_cached_still(self, settings):
        settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        for name in ["a.jpg", "a_t000001000_w160.jpg", "b.jpg"]:
            (settings.thumbnails_dir / name).write_bytes(b"x")

        assert thumbnails.purge("a", settings) == 2
        assert [p.name for p in settings.thumbnails_dir.iterdir()] == ["b.jpg"]

    def test_thumbnail_is_none_when_the_source_is_missing(self, settings):
        job = VideoJob(video_id="ghost", filename="g.mp4", source_path="/gone/g.mp4")
        assert thumbnails.ensure_thumbnail(job, settings) is None


class TestLibraryEndpoint:
    def test_empty_library(self, client):
        payload = client.get("/v1/library").json()
        assert payload == {
            "items": [], "total": 0, "limit": 24, "offset": 0,
            "counts": {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "all": 0},
        }

    def test_rejects_an_unknown_sort(self, client):
        assert client.get("/v1/library?sort=drop-table").status_code == 422

    def test_rejects_an_unknown_status(self, client):
        assert client.get("/v1/library?status=elsewhere").status_code == 422

    @requires_ffmpeg
    def test_uploaded_video_appears_with_its_card_fields(self, client, sample_video):
        with open(sample_video, "rb") as handle:
            video_id = client.post(
                "/v1/videos", files={"file": ("holiday.mp4", handle, "video/mp4")}
            ).json()["video_id"]

        payload = client.get("/v1/library").json()
        assert payload["total"] == 1
        assert payload["counts"]["all"] == 1

        item = payload["items"][0]
        assert item["video_id"] == video_id
        assert item["filename"] == "holiday.mp4"
        assert item["has_render"] is False

    @requires_ffmpeg
    def test_search_reaches_the_endpoint(self, client, sample_video):
        with open(sample_video, "rb") as handle:
            client.post("/v1/videos", files={"file": ("holiday.mp4", handle, "video/mp4")})

        assert client.get("/v1/library?q=holiday").json()["total"] == 1
        assert client.get("/v1/library?q=nothing-like-this").json()["total"] == 0

    @requires_ffmpeg
    def test_pagination_reports_the_unpaged_total(self, client, sample_video):
        for index in range(3):
            with open(sample_video, "rb") as handle:
                client.post(
                    "/v1/videos", files={"file": (f"clip{index}.mp4", handle, "video/mp4")}
                )

        payload = client.get("/v1/library?limit=2").json()
        assert len(payload["items"]) == 2
        assert payload["total"] == 3


class TestThumbnailEndpoints:
    @requires_ffmpeg
    def test_thumbnail_is_generated_on_demand_and_cached(self, client, settings, sample_video):
        with open(sample_video, "rb") as handle:
            video_id = client.post(
                "/v1/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
            ).json()["video_id"]

        response = client.get(f"/v1/videos/{video_id}/thumbnail")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

        cached = thumbnails.thumbnail_path(video_id, settings)
        assert cached.is_file() and cached.stat().st_size > 0

    @requires_ffmpeg
    def test_frame_endpoint_snaps_the_width(self, client, settings, sample_video):
        with open(sample_video, "rb") as handle:
            video_id = client.post(
                "/v1/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
            ).json()["video_id"]

        assert client.get(f"/v1/videos/{video_id}/frame?t=1.0&w=300").status_code == 200
        # 300 is not an offered size; it must land on the nearest one that is.
        assert thumbnails.frame_path(video_id, 1.0, 320, settings).is_file()

    def test_frame_rejects_an_absurd_width(self, client):
        assert client.get("/v1/videos/nope/frame?t=1&w=99999").status_code == 422

    def test_thumbnail_for_an_unknown_video_is_404(self, client):
        assert client.get("/v1/videos/does-not-exist/thumbnail").status_code == 404

    @requires_ffmpeg
    def test_delete_removes_cached_stills(self, client, settings, sample_video):
        with open(sample_video, "rb") as handle:
            video_id = client.post(
                "/v1/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
            ).json()["video_id"]
        client.get(f"/v1/videos/{video_id}/thumbnail")
        client.get(f"/v1/videos/{video_id}/frame?t=1.0&w=160")

        client.delete(f"/v1/videos/{video_id}")
        assert not list(settings.thumbnails_dir.glob(f"{video_id}*"))


class TestSourceResolutionThroughTheAPI:
    @requires_ffmpeg
    def test_source_is_served_after_the_recorded_path_goes_stale(
        self, client, settings, sample_video
    ):
        """The exact shape of a database written inside Docker, read on the host."""
        with open(sample_video, "rb") as handle:
            video_id = client.post(
                "/v1/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
            ).json()["video_id"]

        database = Database(settings.db_path)
        with database._connect() as connection:  # noqa: SLF001 - simulating another host
            connection.execute(
                "UPDATE jobs SET source_path = ? WHERE video_id = ?",
                (f"/app/data/uploads/{video_id}.mp4", video_id),
            )

        assert Path(settings.uploads_dir / f"{video_id}.mp4").is_file()
        assert client.get(f"/v1/videos/{video_id}/source").status_code == 200


class TestVideoResponseAdditions:
    @requires_ffmpeg
    def test_has_render_reflects_the_file_on_disk(self, client, settings, sample_video):
        with open(sample_video, "rb") as handle:
            video_id = client.post(
                "/v1/videos", files={"file": ("clip.mp4", handle, "video/mp4")}
            ).json()["video_id"]

        assert client.get(f"/v1/videos/{video_id}").json()["has_render"] is False

        (settings.renders_dir / f"{video_id}_edit.mp4").write_bytes(b"not really a video")
        assert client.get(f"/v1/videos/{video_id}").json()["has_render"] is True
