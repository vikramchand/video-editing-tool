"""Audio extraction for speech-to-text.

Whisper-family models expect 16 kHz mono PCM, so we normalise to exactly that
once here rather than letting each transcription provider resample.
"""

from __future__ import annotations

import logging
from pathlib import Path

from video_understanding.core.errors import FFmpegError
from video_understanding.video import ffmpeg_utils

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> Path | None:
    """Extract mono PCM audio from `video_path`.

    Returns the written path, or None when the video has no audio track — a
    silent video is a perfectly valid input, it just skips transcription.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not has_audio_stream(video_path):
        logger.info("%s has no audio stream; skipping extraction", video_path.name)
        return None

    ffmpeg_utils.run(
        [
            "-y",
            "-i", str(video_path),
            "-vn",                      # drop video
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",                 # mono
            "-loglevel", "error",
            str(output_path),
        ]
    )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise FFmpegError(f"audio extraction produced an empty file for {video_path.name}")

    logger.info(
        "extracted %.1f KB of audio to %s", output_path.stat().st_size / 1024, output_path.name
    )
    return output_path


def has_audio_stream(video_path: str | Path) -> bool:
    """True when the file carries at least one audio stream."""
    try:
        out = ffmpeg_utils.run_ffprobe(
            [
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path),
            ]
        )
    except FFmpegError:
        return False
    return bool(out.strip())


def audio_duration(audio_path: str | Path) -> float:
    """Duration of an audio file in seconds (0.0 if unreadable)."""
    try:
        out = ffmpeg_utils.run_ffprobe(
            [
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(audio_path),
            ]
        )
        return float(out.strip())
    except (FFmpegError, ValueError):
        return 0.0
