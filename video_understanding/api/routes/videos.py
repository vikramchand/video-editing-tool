"""Video upload, status, result, and rendering endpoints."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse

from video_understanding.ai.transcription import to_srt
from video_understanding.api.dependencies import get_config, get_database, get_processor
from video_understanding.core.config import Settings
from video_understanding.core.errors import RenderError, ValidationError
from video_understanding.core.models import (
    JobStatus,
    RenderRequest,
    RenderResponse,
    StatusResponse,
    UploadResponse,
    VideoJob,
    VideoResponse,
)
from video_understanding.jobs.processor import JobProcessor
from video_understanding.storage.database import Database
from video_understanding.video import rendering
from video_understanding.video.metadata import validate_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/videos", tags=["videos"])

CHUNK_SIZE = 1024 * 1024


def _require_job(database: Database, video_id: str) -> VideoJob:
    job = database.get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"video '{video_id}' not found")
    return job


def _media_type_for(path: Path) -> str:
    """Browser-friendly MIME type for a stored upload."""
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }.get(path.suffix.lower(), "application/octet-stream")


def _safe_suffix(filename: str, settings: Settings) -> str:
    """Extension check before anything touches the disk."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported file type '{suffix or filename}'. "
                f"Allowed: {', '.join(sorted(settings.allowed_extensions))}"
            ),
        )
    return suffix


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def upload_video(
    file: UploadFile = File(..., description="The video file to analyse"),
    database: Database = Depends(get_database),
    processor: JobProcessor = Depends(get_processor),
    settings: Settings = Depends(get_config),
) -> UploadResponse:
    """Upload a video and queue it for processing.

    Returns immediately with a `video_id`; poll `/v1/videos/{id}/status` for
    progress and `/v1/videos/{id}` for the result.
    """
    original_name = file.filename or "upload.mp4"
    suffix = _safe_suffix(original_name, settings)

    video_id = uuid.uuid4().hex[:12]
    destination = settings.uploads_dir / f"{video_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Stream to disk in chunks so a large upload never lands in memory, and so
    # the size limit is enforced while writing rather than after.
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"file exceeds the {settings.video_max_upload_mb:.0f} MB limit"
                        ),
                    )
                handle.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        logger.exception("failed to store upload")
        raise HTTPException(status_code=500, detail=f"could not store upload: {exc}") from exc
    finally:
        await file.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    # Validate before queueing so the caller gets a real error instead of a
    # job that fails a second later.
    try:
        validate_upload(destination, settings)
    except ValidationError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = VideoJob(
        video_id=video_id,
        filename=original_name,
        status=JobStatus.QUEUED,
        source_path=str(destination),
    )
    database.create_job(job)
    processor.submit(video_id, destination)

    return UploadResponse(video_id=video_id, status=JobStatus.QUEUED, filename=original_name)


@router.get("", response_model=list[VideoResponse])
async def list_videos(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_results: bool = Query(default=False, description="Include full results in the list"),
    database: Database = Depends(get_database),
) -> list[VideoResponse]:
    """List uploaded videos, newest first."""
    return [
        _to_response(job, include_result=include_results)
        for job in database.list_jobs(limit=limit, offset=offset)
    ]


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    database: Database = Depends(get_database),
) -> VideoResponse:
    """Get a video's status and, once complete, its structured understanding."""
    return _to_response(_require_job(database, video_id), include_result=True)


@router.get("/{video_id}/status", response_model=StatusResponse)
async def get_status(
    video_id: str,
    database: Database = Depends(get_database),
) -> StatusResponse:
    """Lightweight progress poll."""
    job = _require_job(database, video_id)
    return StatusResponse(
        video_id=job.video_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/{video_id}/source")
async def get_source(
    video_id: str,
    database: Database = Depends(get_database),
) -> FileResponse:
    """Serve the original upload, so the web UI can preview it."""
    job = _require_job(database, video_id)
    source = Path(job.source_path) if job.source_path else None
    if source is None or not source.is_file():
        raise HTTPException(status_code=404, detail="source video is no longer on disk")

    # `inline` (rather than FileResponse's default `attachment`) so the web UI's
    # <video> element plays it instead of downloading it. FileResponse already
    # honours Range requests, which is what makes seeking work.
    return FileResponse(
        source,
        media_type=_media_type_for(source),
        headers={"Content-Disposition": f'inline; filename="{Path(job.filename).name}"'},
    )


@router.get("/{video_id}/transcript.srt", response_class=PlainTextResponse)
async def get_transcript_srt(
    video_id: str,
    database: Database = Depends(get_database),
) -> PlainTextResponse:
    """Download the transcript as an SRT subtitle file."""
    job = _require_job(database, video_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail=f"video is '{job.status.value}', not completed")
    if not job.result.transcript:
        raise HTTPException(status_code=404, detail="this video has no transcript")

    return PlainTextResponse(
        to_srt(job.result.transcript),
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{video_id}.srt"'},
    )


@router.post("/{video_id}/render", response_model=RenderResponse)
async def render_video(
    video_id: str,
    request: RenderRequest | None = None,
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_config),
) -> RenderResponse:
    """Render an edited video from the edit decisions.

    Supports the MVP editing set: removing segments, concatenating the
    remainder, and burning captions.
    """
    options = request or RenderRequest()
    job = _require_job(database, video_id)

    if job.status is not JobStatus.COMPLETED or job.result is None:
        raise HTTPException(
            status_code=409,
            detail=f"video is '{job.status.value}'; rendering needs a completed analysis",
        )
    if not job.source_path or not Path(job.source_path).is_file():
        raise HTTPException(status_code=404, detail="source video is no longer on disk")

    result = job.result
    suggestions = (
        options.edit_suggestions
        if options.edit_suggestions is not None
        else result.edit_suggestions
    )

    try:
        keep = rendering.compute_keep_segments(
            result.duration,
            suggestions,
            apply_removals=options.apply_removals,
            highlights_only=options.highlights_only,
            highlights=result.highlights,
        )
        captions = (
            rendering.remap_transcript(result.transcript, keep) if options.burn_captions else None
        )
        if options.burn_captions and not captions:
            raise HTTPException(
                status_code=400,
                detail="captions were requested but this video has no transcript",
            )

        output_path = settings.renders_dir / f"{video_id}_edit.mp4"
        rendering.render_edit(
            job.source_path,
            output_path,
            keep,
            captions=captions,
            source_duration=result.duration,
        )
    except RenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as a 500 with context
        logger.exception("render failed for %s", video_id)
        raise HTTPException(status_code=500, detail=f"rendering failed: {exc}") from exc

    removed = sum(1 for s in suggestions if s.action == "remove") if options.apply_removals else 0
    return RenderResponse(
        video_id=video_id,
        output_path=str(output_path),
        download_url=f"/v1/videos/{video_id}/render/download",
        source_duration=result.duration,
        output_duration=rendering.total_duration(keep),
        segments_kept=len(keep),
        segments_removed=removed,
        captions_burned=bool(captions),
    )


@router.get("/{video_id}/render/download")
async def download_render(
    video_id: str,
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_config),
) -> FileResponse:
    """Download a previously rendered edit."""
    _require_job(database, video_id)
    output_path = settings.renders_dir / f"{video_id}_edit.mp4"
    if not output_path.is_file():
        raise HTTPException(
            status_code=404, detail="no render exists yet; POST to /render first"
        )
    return FileResponse(
        output_path, media_type="video/mp4", filename=f"{video_id}_edit.mp4"
    )


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: str,
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_config),
) -> None:
    """Delete a video, its result, and its files."""
    job = _require_job(database, video_id)

    if job.source_path:
        Path(job.source_path).unlink(missing_ok=True)
    (settings.renders_dir / f"{video_id}_edit.mp4").unlink(missing_ok=True)
    database.delete_job(video_id)


def _to_response(job: VideoJob, *, include_result: bool) -> VideoResponse:
    return VideoResponse(
        video_id=job.video_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error=job.error,
        filename=job.filename,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result if include_result else None,
    )
