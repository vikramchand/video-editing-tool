"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from video_understanding import __version__
from video_understanding.api.dependencies import AppState
from video_understanding.api.routes import health, videos
from video_understanding.core.config import Settings, get_settings
from video_understanding.core.errors import (
    JobNotFoundError,
    ProviderError,
    ValidationError,
    VideoUnderstandingError,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

DESCRIPTION = """
Local-first video understanding. Upload a video, get back a structured
representation of what happens in it: transcript, scenes, highlights, and
editing suggestions.

Everything runs on your machine. No cloud APIs, no keys, no data leaving the
host.
"""


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # uvicorn's access log is noisy next to pipeline progress.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(settings)
        app.state.app_state = state
        state.startup()
        try:
            yield
        finally:
            state.shutdown()

    app = FastAPI(
        title="Video Understanding API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(videos.router)

    _register_exception_handlers(app)
    _mount_web_ui(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Map domain errors onto HTTP responses in one place."""

    @app.exception_handler(JobNotFoundError)
    async def _not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def _invalid(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider(request: Request, exc: ProviderError) -> JSONResponse:
        # 503: the request was fine, a local model backend was not.
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "provider": exc.provider, "model": exc.model},
        )

    @app.exception_handler(VideoUnderstandingError)
    async def _domain(request: Request, exc: VideoUnderstandingError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})


def _mount_web_ui(app: FastAPI) -> None:
    """Serve the bundled single-page UI at `/`, if it is present."""
    if not WEB_DIR.is_dir():
        logger.warning("web UI directory not found at %s", WEB_DIR)
        return

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")


app = create_app()
