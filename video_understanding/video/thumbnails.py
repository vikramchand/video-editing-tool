"""Poster frames for the library.

The library needs a still per video. One is worth a filename and a status chip
put together, and it costs a single ffmpeg seek — so thumbnails are generated
once and cached on disk, under a name derived from the video id. That makes the
cache self-describing: no column to keep in sync, and deleting the directory
only costs the next request a re-seek.

The frame is chosen from the analysis when there is one: the top highlight
beats the most important scene, which beats an arbitrary point near the start.
A video that failed before producing a result still gets a thumbnail, because
a failed row in the library should still look like the video it came from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from video_understanding.core.config import Settings
from video_understanding.core.errors import FFmpegError
from video_understanding.core.models import VideoJob
from video_understanding.storage.files import resolve_source
from video_understanding.video.frames import extract_still

logger = logging.getLogger(__name__)

THUMBNAIL_WIDTH = 480


def choose_timestamp(job: VideoJob) -> float:
    """Pick the most representative moment of a video.

    Falls back progressively so this works for a job that has no result yet.
    """
    result = job.result
    if result is None:
        return 1.0

    if result.highlights:
        best = max(result.highlights, key=lambda h: h.score)
        return max(0.0, (best.start + best.end) / 2)

    if result.scenes:
        best_scene = max(result.scenes, key=lambda s: s.importance)
        return max(0.0, (best_scene.start + best_scene.end) / 2)

    # A tenth of the way in skips the black frame or lens ramp that phone
    # cameras often start with, without landing past the end of a short clip.
    return round(max(0.5, result.duration * 0.1), 2)


def thumbnail_path(video_id: str, settings: Settings) -> Path:
    return settings.thumbnails_dir / f"{video_id}.jpg"


def frame_path(video_id: str, timestamp: float, width: int, settings: Settings) -> Path:
    """Cache location for an arbitrary still, keyed by time and width.

    Millisecond precision in the name is what makes the cache a cache: the UI
    asks for the same scene starts on every visit, so the second visit costs a
    file read rather than an ffmpeg seek.
    """
    return settings.thumbnails_dir / f"{video_id}_t{int(round(timestamp * 1000)):09d}_w{width}.jpg"


def ensure_frame(job: VideoJob, timestamp: float, width: int, settings: Settings) -> Path | None:
    """Return a still from an arbitrary point in the video, generating it once.

    Used for the scene strip: a list of scene descriptions is far easier to
    scan when each one carries the frame it is describing.
    """
    destination = frame_path(job.video_id, timestamp, width, settings)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    source = resolve_source(job.source_path, job.video_id, settings.uploads_dir)
    if source is None:
        return None

    try:
        created = extract_still(source, max(0.0, timestamp), destination, width)
    except FFmpegError as exc:
        logger.warning("frame extraction failed for %s at %.2fs: %s", job.video_id, timestamp, exc)
        return None

    return destination if created else None


def purge(video_id: str, settings: Settings) -> int:
    """Delete every cached still for a video. Returns how many were removed."""
    removed = 0
    for path in settings.thumbnails_dir.glob(f"{video_id}*.jpg"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def ensure_thumbnail(job: VideoJob, settings: Settings) -> Path | None:
    """Return the video's poster frame, generating it if it is missing.

    Returns None when no frame can be produced — no source file on disk, or an
    ffmpeg that cannot decode this container. Callers treat that as "show the
    placeholder", never as an error.
    """
    destination = thumbnail_path(job.video_id, settings)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    source = resolve_source(job.source_path, job.video_id, settings.uploads_dir)
    if source is None:
        return None

    timestamp = choose_timestamp(job)
    try:
        created = extract_still(source, timestamp, destination, THUMBNAIL_WIDTH)
        # Seeking can land past the end of a clip shorter than the analysis
        # claimed; the first frame always exists.
        if not created and timestamp > 0:
            created = extract_still(source, 0.0, destination, THUMBNAIL_WIDTH)
    except FFmpegError as exc:
        logger.warning("thumbnail generation failed for %s: %s", job.video_id, exc)
        return None

    return destination if created else None
