"""Transcription stage."""

from __future__ import annotations

import logging
from pathlib import Path

from video_understanding.core.models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Wraps a `TranscriptionProvider` and normalises its output."""

    def __init__(self, provider) -> None:
        self.provider = provider

    def transcribe(self, audio_path: str | Path | None, duration: float = 0.0) -> Transcript:
        """Transcribe extracted audio.

        A video with no audio track yields an empty transcript rather than an
        error; silent video is a normal input.
        """
        if audio_path is None:
            logger.info("no audio track; returning empty transcript")
            return Transcript(provider=getattr(self.provider, "name", "none"), model="none")

        transcript = self.provider.transcribe(str(audio_path))
        return normalize_transcript(transcript, duration)


def normalize_transcript(transcript: Transcript, duration: float = 0.0) -> Transcript:
    """Sort, clamp and de-overlap transcript segments.

    Whisper occasionally emits segments that run past the end of the audio or
    that overlap by a few milliseconds. Downstream code slices the transcript by
    scene window, so monotonic, in-range timestamps matter.
    """
    segments = sorted(transcript.segments, key=lambda s: (s.start, s.end))
    cleaned: list[TranscriptSegment] = []
    previous_end = 0.0

    for segment in segments:
        start = max(0.0, segment.start)
        end = segment.end
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        # Nudge overlaps forward rather than dropping the segment's text.
        if start < previous_end:
            start = previous_end
        if end < start:
            end = start
        if not segment.text.strip():
            continue

        cleaned.append(
            segment.model_copy(update={"start": round(start, 3), "end": round(end, 3)})
        )
        previous_end = end

    return Transcript(
        language=transcript.language,
        text=" ".join(s.text for s in cleaned).strip(),
        segments=cleaned,
        provider=transcript.provider,
        model=transcript.model,
    )


def to_srt(segments: list[TranscriptSegment]) -> str:
    """Render transcript segments as an SRT subtitle file."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_srt_time(segment.start)} --> {_srt_time(segment.end)}\n{segment.text}\n"
        )
    return "\n".join(blocks)


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding can carry
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
