"""Scene boundary detection.

Three detectors, tried in order of quality when `SCENE_DETECTOR=auto`:

1. **pyscenedetect** - content-aware detection (needs the `scenes` extra).
2. **ffmpeg** - the built-in `scene` filter. No extra dependencies, decent on
   hard cuts, weaker on slow pans. This is the Docker default.
3. **uniform** - fixed-length chunks. Always works, never fails, and is the
   guaranteed floor so the pipeline never dies at this stage.

Whatever detector runs, the output is post-processed identically: boundaries
become contiguous non-overlapping scenes, short scenes are merged into their
neighbour, and the last scene is clamped to the video duration.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from video_understanding.core.config import Settings
from video_understanding.core.errors import FFmpegError
from video_understanding.core.models import SceneBoundary
from video_understanding.video import ffmpeg_utils

logger = logging.getLogger(__name__)

_PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


def detect_scenes(
    video_path: str | Path,
    duration: float,
    settings: Settings,
) -> list[SceneBoundary]:
    """Detect scenes in a video, degrading gracefully between detectors."""
    video_path = Path(video_path)
    if duration <= 0:
        return []

    order: list[str]
    if settings.scene_detector == "auto":
        order = ["pyscenedetect", "ffmpeg", "uniform"]
    else:
        # An explicit choice still falls back to uniform rather than failing.
        order = [settings.scene_detector]
        if settings.scene_detector != "uniform":
            order.append("uniform")

    for detector in order:
        try:
            cuts = _run_detector(detector, video_path, duration, settings)
        except Exception as exc:  # noqa: BLE001 - detector choice must never be fatal
            logger.warning("scene detector '%s' failed (%s); trying next", detector, exc)
            continue

        scenes = build_scenes(cuts, duration, settings.scene_min_duration, detector)
        if scenes:
            logger.info("detected %d scene(s) using '%s'", len(scenes), detector)
            return scenes

    # Unreachable in practice: `uniform` cannot return empty for duration > 0.
    return build_scenes([], duration, settings.scene_min_duration, "uniform")


def _run_detector(
    name: str, video_path: Path, duration: float, settings: Settings
) -> list[float]:
    if name == "pyscenedetect":
        return _detect_pyscenedetect(video_path, settings)
    if name == "ffmpeg":
        return _detect_ffmpeg(video_path, settings)
    if name == "uniform":
        return _detect_uniform(duration, settings)
    raise ValueError(f"unknown scene detector: {name}")


def _detect_pyscenedetect(video_path: Path, settings: Settings) -> list[float]:
    """Content-aware detection via PySceneDetect (optional dependency)."""
    from scenedetect import ContentDetector, SceneManager, open_video  # noqa: PLC0415

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=settings.scene_threshold,
            min_scene_len=max(1, int(settings.scene_min_duration * (video.frame_rate or 30))),
        )
    )
    manager.detect_scenes(video, show_progress=False)
    scene_list = manager.get_scene_list()
    # Keep only the cut points; `build_scenes` reconstructs the ranges.
    return [scene[0].get_seconds() for scene in scene_list if scene[0].get_seconds() > 0]


def _detect_ffmpeg(video_path: Path, settings: Settings) -> list[float]:
    """Use ffmpeg's `scene` filter and parse timestamps out of showinfo."""
    proc = ffmpeg_utils.run(
        [
            "-i", str(video_path),
            "-filter:v", f"select='gt(scene,{settings.scene_ffmpeg_threshold})',showinfo",
            "-f", "null",
            "-",
        ],
        check=False,
    )
    # showinfo writes to stderr; a non-zero exit with usable output still counts.
    if proc.returncode != 0 and "pts_time" not in proc.stderr:
        raise FFmpegError(
            "ffmpeg scene detection failed",
            stderr="\n".join(proc.stderr.strip().splitlines()[-8:]),
        )
    return [float(m) for m in _PTS_RE.findall(proc.stderr)]


def _detect_uniform(duration: float, settings: Settings) -> list[float]:
    """Fixed-length chunking. The detector that cannot fail."""
    step = max(settings.scene_uniform_duration, settings.scene_min_duration)
    cuts: list[float] = []
    t = step
    while t < duration - 0.5:
        cuts.append(round(t, 3))
        t += step
    return cuts


def build_scenes(
    cuts: list[float],
    duration: float,
    min_duration: float,
    detector: str,
) -> list[SceneBoundary]:
    """Turn a list of cut timestamps into contiguous, merged scene ranges."""
    if duration <= 0:
        return []

    # Sort, drop out-of-range and duplicate cuts.
    points = sorted({round(c, 3) for c in cuts if 0 < c < duration})
    boundaries = [0.0, *points, round(duration, 3)]

    raw: list[tuple[float, float]] = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]
    if not raw:
        raw = [(0.0, round(duration, 3))]

    merged = _merge_short(raw, min_duration)

    return [
        SceneBoundary(id=index, start=round(start, 3), end=round(end, 3), detector=detector)
        for index, (start, end) in enumerate(merged, start=1)
    ]


def _merge_short(
    ranges: list[tuple[float, float]], min_duration: float
) -> list[tuple[float, float]]:
    """Fold scenes shorter than `min_duration` into the previous one.

    A too-short first scene is folded forward into the second instead, so the
    timeline always starts at 0.
    """
    if len(ranges) <= 1:
        return ranges

    merged: list[list[float]] = [list(ranges[0])]
    for start, end in ranges[1:]:
        previous = merged[-1]
        if (end - start) < min_duration or (previous[1] - previous[0]) < min_duration:
            previous[1] = end
        else:
            merged.append([start, end])

    # The loop can still leave a short final scene; fold it back.
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_duration:
        tail = merged.pop()
        merged[-1][1] = tail[1]

    return [(start, end) for start, end in merged]
