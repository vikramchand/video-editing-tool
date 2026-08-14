"""Video metadata extraction and upload validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from video_understanding.core.config import Settings
from video_understanding.core.errors import FFmpegError, ValidationError
from video_understanding.core.models import VideoMetadata
from video_understanding.video import ffmpeg_utils

logger = logging.getLogger(__name__)


def _parse_fraction(value: str | None) -> float:
    """ffprobe reports frame rates as strings like '30000/1001'."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, _, den = value.partition("/")
            denominator = float(den)
            return float(num) / denominator if denominator else 0.0
        return float(value)
    except ValueError:
        return 0.0


def _rotation_of(stream: dict) -> int:
    """Phone videos carry rotation in a side-data matrix or a tag."""
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(float(side["rotation"])) % 360
            except (TypeError, ValueError):
                continue
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return int(float(tag)) % 360
        except (TypeError, ValueError):
            pass
    return 0


def probe(path: str | Path) -> dict:
    """Return the raw ffprobe JSON for a media file."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"file not found: {path}")

    raw = ffmpeg_utils.run_ffprobe(
        [
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {path.name}") from exc


def extract_metadata(path: str | Path) -> VideoMetadata:
    """Extract structured metadata from a video file."""
    path = Path(path)
    data = probe(path)

    streams = data.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise ValidationError(
            f"{path.name} contains no video stream (found: "
            f"{', '.join(sorted({s.get('codec_type', '?') for s in streams})) or 'nothing'})"
        )

    video = video_streams[0]
    fmt = data.get("format") or {}

    # Prefer the container duration; fall back to the stream's own.
    duration = 0.0
    for candidate in (fmt.get("duration"), video.get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(
        video.get("r_frame_rate")
    )
    rotation = _rotation_of(video)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)

    # A 90/270 degree rotation means the displayed frame is transposed.
    if rotation in (90, 270):
        width, height = height, width

    try:
        size_bytes = int(fmt.get("size") or path.stat().st_size)
    except (TypeError, ValueError):
        size_bytes = path.stat().st_size

    return VideoMetadata(
        duration=max(0.0, duration),
        width=width,
        height=height,
        fps=round(fps, 3),
        video_codec=video.get("codec_name"),
        audio_codec=audio_streams[0].get("codec_name") if audio_streams else None,
        has_audio=bool(audio_streams),
        size_bytes=size_bytes,
        container=(fmt.get("format_name") or "").split(",")[0] or None,
        rotation=rotation,
        created_at=(fmt.get("tags") or {}).get("creation_time"),
    )


def validate_upload(path: str | Path, settings: Settings) -> VideoMetadata:
    """Validate an uploaded file and return its metadata.

    Raises `ValidationError` with a message safe to show the caller.
    """
    path = Path(path)

    if not path.is_file():
        raise ValidationError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise ValidationError(f"unsupported file type '{suffix or path.name}'. Allowed: {allowed}")

    size = path.stat().st_size
    if size == 0:
        raise ValidationError("uploaded file is empty")
    if size > settings.max_upload_bytes:
        raise ValidationError(
            f"file is {size / 1024 / 1024:.1f} MB, which exceeds the "
            f"{settings.video_max_upload_mb:.0f} MB limit"
        )

    try:
        metadata = extract_metadata(path)
    except FFmpegError as exc:
        raise ValidationError(f"could not read video: {exc}") from exc

    if metadata.duration <= 0:
        raise ValidationError("video has zero duration or an unreadable timeline")
    if metadata.duration > settings.video_max_duration_seconds:
        raise ValidationError(
            f"video is {metadata.duration:.1f}s, which exceeds the "
            f"{settings.video_max_duration_seconds:.0f}s limit"
        )

    return metadata
