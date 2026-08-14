"""Health and introspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from video_understanding import __version__
from video_understanding.ai.providers import registry
from video_understanding.api.dependencies import get_config, get_database
from video_understanding.core.config import Settings, get_hardware
from video_understanding.core.models import HealthResponse
from video_understanding.storage.database import Database
from video_understanding.video import ffmpeg_utils

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_config),
    database: Database = Depends(get_database),
) -> HealthResponse:
    """Report whether the API and everything it depends on are usable.

    Returns 200 either way: `status` distinguishes fully-working from degraded,
    which is more useful for diagnosis than a failed request. FFmpeg and the
    database are hard requirements; a missing model only degrades.
    """
    ffmpeg_ok = ffmpeg_utils.available()
    database_ok = database.healthy()
    providers = registry.health_report(settings)

    checks = {
        "ffmpeg": ffmpeg_utils.version() or "not found on PATH",
        "database": str(settings.db_path) if database_ok else "unreachable",
        "hardware": get_hardware().describe(),
        "in_container": str(get_hardware().in_container).lower(),
        "ollama_host": settings.ollama_host,
        "workers": str(settings.worker_count),
    }

    degraded = (
        not ffmpeg_ok
        or not database_ok
        or any(not value.startswith("ok") for value in providers.values())
    )

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        ffmpeg=ffmpeg_ok,
        database=database_ok,
        providers=providers,
        checks=checks,
    )


@router.get("/v1/config", tags=["system"])
async def get_configuration(settings: Settings = Depends(get_config)) -> dict:
    """Show the active configuration, so behaviour is explainable at runtime."""
    return {
        "providers": settings.provider_summary(),
        "video": {
            "max_duration_seconds": settings.video_max_duration_seconds,
            "max_upload_mb": settings.video_max_upload_mb,
            "frames_per_scene": settings.video_frames_per_scene,
            "max_frames": settings.video_max_frames,
            "frame_width": settings.video_frame_width,
            "allowed_extensions": sorted(settings.allowed_extensions),
        },
        "scenes": {
            "detector": settings.scene_detector,
            "threshold": settings.scene_threshold,
            "ffmpeg_threshold": settings.scene_ffmpeg_threshold,
            "min_duration": settings.scene_min_duration,
        },
        "runtime": {
            "worker_count": settings.worker_count,
            "fail_on_provider_error": settings.fail_on_provider_error,
            "hardware": get_hardware().describe(),
        },
    }
