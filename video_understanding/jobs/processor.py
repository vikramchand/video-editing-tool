"""Background job queue.

Deliberately the simplest thing that works: a bounded `ThreadPoolExecutor`
plus SQLite for state. No Redis, no Celery, no broker. Video work is dominated
by subprocess (FFmpeg) and HTTP (Ollama) waits, both of which release the GIL,
so threads are a genuine fit rather than a compromise.

`WORKER_COUNT` defaults to 1 because local model inference is memory-bound: two
concurrent jobs on a 16 GB Mac will swap rather than go faster.

The V4 roadmap swaps this class for a real queue. Nothing outside it knows how
jobs are scheduled, so that swap stays local.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from video_understanding.core.config import Settings
from video_understanding.core.errors import VideoUnderstandingError
from video_understanding.core.models import JobStatus, ProcessingStage
from video_understanding.jobs.pipeline import VideoPipeline
from video_understanding.storage.database import Database

logger = logging.getLogger(__name__)


class JobProcessor:
    """Accepts jobs, runs them on a worker pool, records progress in SQLite."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        *,
        pipeline_factory=None,
    ) -> None:
        self.database = database
        self.settings = settings
        # Injectable so tests can substitute a pipeline without touching threads.
        self._pipeline_factory = pipeline_factory or (lambda: VideoPipeline(settings))
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_count, thread_name_prefix="vu-worker"
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pipeline: VideoPipeline | None = None
        self._pipeline_lock = threading.Lock()
        self._shutdown = False

    def _get_pipeline(self) -> VideoPipeline:
        """Build the pipeline once and reuse it.

        This matters: it keeps the Whisper model loaded across jobs instead of
        paying the load cost on every upload.
        """
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    self._pipeline = self._pipeline_factory()
        return self._pipeline

    def submit(self, video_id: str, video_path: str | Path) -> None:
        """Queue a video for processing."""
        if self._shutdown:
            raise RuntimeError("job processor is shutting down")

        with self._lock:
            if video_id in self._futures and not self._futures[video_id].done():
                logger.warning("job %s is already running; ignoring duplicate submit", video_id)
                return
            future = self._executor.submit(self._run, video_id, Path(video_path))
            self._futures[video_id] = future

        future.add_done_callback(lambda _: self._forget(video_id))
        logger.info("queued job %s", video_id)

    def _forget(self, video_id: str) -> None:
        with self._lock:
            self._futures.pop(video_id, None)

    def _run(self, video_id: str, video_path: Path) -> None:
        """Worker body. Never raises: every outcome is recorded in the database."""
        logger.info("starting job %s", video_id)
        try:
            self.database.update_status(
                video_id, status=JobStatus.PROCESSING, stage=ProcessingStage.VALIDATING
            )
        except Exception:  # noqa: BLE001 - the job row may have been deleted
            logger.exception("could not mark job %s as processing", video_id)
            return

        def on_stage(stage: ProcessingStage, detail: str) -> None:
            try:
                self.database.update_status(video_id, stage=stage, progress=stage.progress)
            except Exception:  # noqa: BLE001 - progress reporting must never fail a job
                logger.debug("progress update failed for %s", video_id, exc_info=True)

        try:
            result = self._get_pipeline().process(
                video_path,
                work_dir=self.settings.work_dir / video_id,
                on_stage=on_stage,
            )
            self.database.save_result(video_id, result)
            logger.info("job %s completed", video_id)

        except VideoUnderstandingError as exc:
            # Expected failure: the message is safe and useful to show the user.
            logger.warning("job %s failed: %s", video_id, exc)
            self._fail(video_id, str(exc))

        except Exception as exc:  # noqa: BLE001 - a worker must never die silently
            logger.error("job %s crashed:\n%s", video_id, traceback.format_exc())
            self._fail(video_id, f"Unexpected error: {type(exc).__name__}: {exc}")

    def _fail(self, video_id: str, message: str) -> None:
        try:
            self.database.mark_failed(video_id, message)
        except Exception:  # noqa: BLE001
            logger.exception("could not record failure for job %s", video_id)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for future in self._futures.values() if not future.done())

    def wait_for(self, video_id: str, timeout: float | None = None) -> bool:
        """Block until a job finishes. Used by tests and the synchronous CLI."""
        with self._lock:
            future = self._futures.get(video_id)
        if future is None:
            return True
        try:
            future.result(timeout=timeout)
        except Exception:  # noqa: BLE001 - failures are already recorded
            return True
        return True

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        logger.info("job processor shut down")
