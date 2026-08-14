"""Shared fixtures.

The guiding rule: **no test requires a model**. Anything touching AI goes
through the mock providers. Tests that genuinely need FFmpeg are marked
`@pytest.mark.ffmpeg` and skip cleanly when it is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_understanding.core.config import Settings
from video_understanding.core.models import (
    EditSuggestion,
    Highlight,
    Scene,
    TranscriptSegment,
    VideoMetadata,
    VideoUnderstanding,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed"
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely at a temp directory, with mock providers."""
    config = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        transcription_provider="mock",
        vision_provider="mock",
        llm_provider="mock",
        worker_count=1,
        video_max_duration_seconds=600,
        keep_intermediate_files=False,
    )
    config.ensure_directories()
    return config


def _make_clip(path: Path, source: str, seconds: float, *, audio: bool, freq: int = 440) -> None:
    """Render a clip from an lavfi source, with an optional sine tone.

    The sources used below are *textured* (test patterns, not flat colours) on
    purpose. Solid-colour clips produce content deltas well below what real
    footage yields at a cut, so a fixture built from them would only pass with
    an artificially lowered detection threshold — the test would stop
    exercising the production defaults it is supposed to protect.
    """
    args = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"{source}=s=320x240:r=25",
        "-t", str(seconds),
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    subprocess.run(args, check=True, capture_output=True)


def _concat(parts: list[Path], destination: Path) -> None:
    listing = destination.parent / "concat_list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", listing.name,
            "-c", "copy", destination.name,
        ],
        check=True,
        capture_output=True,
        cwd=destination.parent,
    )


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """A 9-second video with three visually distinct scenes and an audio track."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")

    directory = tmp_path_factory.mktemp("media")
    parts = []
    for index, (source, freq) in enumerate(
        [("testsrc2", 440), ("smptebars", 660), ("mandelbrot", 880)]
    ):
        part = directory / f"part{index}.mp4"
        _make_clip(part, source, 3.0, audio=True, freq=freq)
        parts.append(part)

    destination = directory / "sample.mp4"
    _concat(parts, destination)
    return destination


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> Path:
    """A 3-second video with no audio track at all."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not installed")

    directory = tmp_path_factory.mktemp("media_silent")
    destination = directory / "silent.mp4"
    _make_clip(destination, "testsrc2", 3.0, audio=False)
    return destination


@pytest.fixture
def metadata() -> VideoMetadata:
    return VideoMetadata(
        duration=30.0,
        width=1080,
        height=1920,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        has_audio=True,
        size_bytes=1024,
        container="mov",
    )


@pytest.fixture
def transcript_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=1.0, end=4.0, text="Hello and welcome."),
        TranscriptSegment(start=4.5, end=8.0, text="Let me show you the kitchen."),
        TranscriptSegment(start=12.0, end=16.0, text="And now we head outside."),
    ]


@pytest.fixture
def scenes() -> list[Scene]:
    return [
        Scene(
            id=1, start=0.0, end=10.0, description="Person enters the kitchen.",
            people=["one adult"], objects=["counter"], actions=["walking"],
            importance=0.6, keyframes=[2.0, 5.0], transcript_text="Hello and welcome.",
        ),
        Scene(
            id=2, start=10.0, end=20.0, description="Person opens the refrigerator.",
            people=["one adult"], objects=["refrigerator", "drink"], actions=["opening"],
            importance=0.85, keyframes=[12.0, 15.0], transcript_text="And now we head outside.",
        ),
        Scene(
            id=3, start=20.0, end=30.0, description="Person walks outside.",
            people=["one adult"], objects=["door"], actions=["walking"],
            importance=0.4, keyframes=[25.0],
        ),
    ]


@pytest.fixture
def understanding(metadata, transcript_segments, scenes) -> VideoUnderstanding:
    return VideoUnderstanding(
        duration=30.0,
        summary="A person makes a drink and goes outside.",
        title="Kitchen Trip",
        metadata=metadata,
        transcript=transcript_segments,
        scenes=scenes,
        highlights=[Highlight(start=10.0, end=20.0, reason="Main action.", score=0.85)],
        edit_suggestions=[
            EditSuggestion(start=0.0, end=1.0, action="remove", reason="Silence."),
            EditSuggestion(start=10.0, end=20.0, action="caption", reason="Key speech."),
        ],
    )
