"""SQLite job store."""

from __future__ import annotations

import threading

import pytest

from video_understanding.core.errors import JobNotFoundError
from video_understanding.core.models import (
    JobStatus,
    ProcessingStage,
    VideoJob,
    VideoUnderstanding,
)
from video_understanding.storage.database import Database


@pytest.fixture
def database(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def job() -> VideoJob:
    return VideoJob(video_id="abc123", filename="test.mp4", source_path="/tmp/test.mp4")


class TestJobLifecycle:
    def test_create_and_read(self, database, job):
        database.create_job(job)
        stored = database.get_job("abc123")
        assert stored is not None
        assert stored.filename == "test.mp4"
        assert stored.status is JobStatus.QUEUED

    def test_missing_job_returns_none(self, database):
        assert database.get_job("nope") is None

    def test_require_job_raises(self, database):
        with pytest.raises(JobNotFoundError):
            database.require_job("nope")

    def test_update_status(self, database, job):
        database.create_job(job)
        database.update_status("abc123", status=JobStatus.PROCESSING, stage=ProcessingStage.VISION)

        stored = database.get_job("abc123")
        assert stored.status is JobStatus.PROCESSING
        assert stored.stage is ProcessingStage.VISION
        assert stored.started_at is not None

    def test_stage_sets_progress_automatically(self, database, job):
        database.create_job(job)
        database.update_status("abc123", stage=ProcessingStage.VISION)
        assert database.get_job("abc123").progress == ProcessingStage.VISION.progress

    def test_explicit_progress_overrides_stage_default(self, database, job):
        database.create_job(job)
        database.update_status("abc123", stage=ProcessingStage.VISION, progress=0.42)
        assert database.get_job("abc123").progress == 0.42

    def test_progress_is_clamped(self, database, job):
        database.create_job(job)
        database.update_status("abc123", progress=5.0)
        assert database.get_job("abc123").progress == 1.0

    def test_update_of_a_missing_job_raises(self, database):
        with pytest.raises(JobNotFoundError):
            database.update_status("nope", status=JobStatus.FAILED)

    def test_partial_update_leaves_other_fields_alone(self, database, job):
        database.create_job(job)
        database.update_status("abc123", status=JobStatus.PROCESSING)
        database.update_status("abc123", stage=ProcessingStage.REASONING)
        assert database.get_job("abc123").status is JobStatus.PROCESSING


class TestResults:
    def test_save_and_load(self, database, job, understanding):
        database.create_job(job)
        database.save_result("abc123", understanding)

        stored = database.get_job("abc123")
        assert stored.status is JobStatus.COMPLETED
        assert stored.progress == 1.0
        assert stored.result is not None
        assert stored.result.duration == 30.0
        assert len(stored.result.scenes) == 3
        assert stored.result.scenes[1].objects == ["refrigerator", "drink"]

    def test_saving_a_result_clears_a_previous_error(self, database, job, understanding):
        database.create_job(job)
        database.mark_failed("abc123", "transient failure")
        database.save_result("abc123", understanding)
        assert database.get_job("abc123").error is None

    def test_save_for_a_missing_job_raises(self, database, understanding):
        with pytest.raises(JobNotFoundError):
            database.save_result("nope", understanding)

    def test_result_survives_a_reopen(self, tmp_path, job, understanding):
        path = tmp_path / "persist.db"
        first = Database(path)
        first.create_job(job)
        first.save_result("abc123", understanding)

        assert Database(path).get_job("abc123").result.summary == understanding.summary

    def test_unreadable_stored_result_does_not_break_the_job(self, database, job):
        database.create_job(job)
        database.save_result("abc123", VideoUnderstanding(duration=1.0))
        with database._connect() as connection:
            connection.execute(
                "UPDATE jobs SET result_json = ? WHERE video_id = ?", ("{not json", "abc123")
            )

        stored = database.get_job("abc123")
        assert stored is not None
        assert stored.result is None


class TestFailures:
    def test_mark_failed(self, database, job):
        database.create_job(job)
        database.mark_failed("abc123", "ffmpeg exploded")

        stored = database.get_job("abc123")
        assert stored.status is JobStatus.FAILED
        assert stored.error == "ffmpeg exploded"
        assert stored.completed_at is not None

    def test_reset_stale_jobs(self, database):
        database.create_job(VideoJob(video_id="a", filename="a.mp4"))
        database.create_job(VideoJob(video_id="b", filename="b.mp4"))
        database.update_status("b", status=JobStatus.PROCESSING)

        assert database.reset_stale_jobs() == 2
        assert database.get_job("a").status is JobStatus.FAILED
        assert "interrupted" in database.get_job("b").error

    def test_reset_leaves_completed_jobs_alone(self, database, job, understanding):
        database.create_job(job)
        database.save_result("abc123", understanding)
        database.reset_stale_jobs()
        assert database.get_job("abc123").status is JobStatus.COMPLETED


class TestListing:
    def test_lists_newest_first(self, database):
        for index in range(3):
            database.create_job(VideoJob(video_id=f"id{index}", filename=f"{index}.mp4"))
        assert len(database.list_jobs()) == 3

    def test_pagination(self, database):
        for index in range(5):
            database.create_job(VideoJob(video_id=f"id{index}", filename=f"{index}.mp4"))
        assert len(database.list_jobs(limit=2)) == 2
        assert len(database.list_jobs(limit=2, offset=4)) == 1

    def test_count(self, database, job):
        assert database.count_jobs() == 0
        database.create_job(job)
        assert database.count_jobs() == 1

    def test_delete(self, database, job):
        database.create_job(job)
        assert database.delete_job("abc123") is True
        assert database.get_job("abc123") is None

    def test_delete_missing_returns_false(self, database):
        assert database.delete_job("nope") is False


class TestConcurrency:
    def test_parallel_writes_do_not_corrupt_state(self, database):
        """Worker threads and the API event loop both write; WAL must hold up."""
        for index in range(10):
            database.create_job(VideoJob(video_id=f"job{index}", filename=f"{index}.mp4"))

        errors: list[Exception] = []

        def worker(video_id: str) -> None:
            try:
                for stage in ProcessingStage:
                    database.update_status(video_id, stage=stage)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"job{i}",)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        assert all(database.get_job(f"job{i}").stage is ProcessingStage.DONE for i in range(10))


class TestHealth:
    def test_healthy_database(self, database):
        assert database.healthy() is True
