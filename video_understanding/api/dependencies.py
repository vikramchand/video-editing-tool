"""Shared application state and FastAPI dependencies."""

from __future__ import annotations

import logging

from fastapi import Request

from video_understanding.core.config import Settings, get_settings
from video_understanding.jobs.processor import JobProcessor
from video_understanding.storage.database import Database

logger = logging.getLogger(__name__)


class AppState:
    """Owns the database and worker pool for the process lifetime."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.database = Database(self.settings.db_path)
        self.processor = JobProcessor(self.database, self.settings)

    def startup(self) -> None:
        stale = self.database.reset_stale_jobs()
        logger.info(
            "ready: data_dir=%s, workers=%d, providers=%s%s",
            self.settings.data_dir,
            self.settings.worker_count,
            self.settings.provider_summary(),
            f", failed {stale} stale job(s)" if stale else "",
        )

    def shutdown(self) -> None:
        self.processor.shutdown(wait=False)


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_database(request: Request) -> Database:
    return request.app.state.app_state.database


def get_processor(request: Request) -> JobProcessor:
    return request.app.state.app_state.processor


def get_config(request: Request) -> Settings:
    return request.app.state.app_state.settings
