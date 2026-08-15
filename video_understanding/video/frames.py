"""Representative frame sampling.

This module is where the central cost optimisation lives. A 60-second phone
video at 30 fps is 1,800 frames; sending those to a local VLM would take hours.
Instead we detect scenes and sample a handful of frames from each:

    60s video -> 10 scenes -> 3 frames/scene -> 30 frames

`MAX_FRAMES` is a hard ceiling on the whole video. When the per-scene budget
would exceed it, frames are redistributed in proportion to scene duration so
long scenes keep more coverage, while every scene retains at least one frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

from video_understanding.core.config import Settings
from video_understanding.core.errors import FFmpegError
from video_understanding.core.models import Frame, SceneBoundary
from video_understanding.video import ffmpeg_utils

logger = logging.getLogger(__name__)


def plan_frame_budget(
    scenes: list[SceneBoundary],
    frames_per_scene: int,
    max_frames: int,
) -> dict[int, int]:
    """Decide how many frames each scene gets.

    Returns a mapping of scene id -> frame count. Scenes omitted from the
    mapping get no frames (only possible when there are more scenes than
    `max_frames`, in which case the longest scenes win).
    """
    if not scenes or max_frames <= 0 or frames_per_scene <= 0:
        return {}

    # More scenes than the total budget: one frame each for the longest scenes.
    if len(scenes) > max_frames:
        longest = sorted(scenes, key=lambda s: s.duration, reverse=True)[:max_frames]
        logger.info(
            "%d scenes exceed the %d frame budget; sampling the %d longest",
            len(scenes), max_frames, len(longest),
        )
        return {scene.id: 1 for scene in longest}

    desired = {scene.id: frames_per_scene for scene in scenes}
    total = sum(desired.values())
    if total <= max_frames:
        return desired

    # Over budget: one frame guaranteed per scene, the rest shared out by duration.
    budget = {scene.id: 1 for scene in scenes}
    remaining = max_frames - len(scenes)
    total_duration = sum(s.duration for s in scenes) or 1.0

    if remaining > 0:
        extras = sorted(
            ((s.id, s.duration / total_duration * remaining) for s in scenes),
            key=lambda pair: pair[1],
            reverse=True,
        )
        for scene_id, share in extras:
            if remaining <= 0:
                break
            grant = min(int(share), frames_per_scene - 1, remaining)
            budget[scene_id] += grant
            remaining -= grant

        # Hand out any leftover from rounding, longest scene first.
        for scene_id, _ in extras:
            if remaining <= 0:
                break
            if budget[scene_id] < frames_per_scene:
                budget[scene_id] += 1
                remaining -= 1

    logger.info(
        "frame budget: %d frames across %d scenes (cap %d)",
        sum(budget.values()), len(scenes), max_frames,
    )
    return budget


def sample_timestamps(scene: SceneBoundary, count: int) -> list[float]:
    """Evenly spaced timestamps inside a scene.

    Points are placed at the centre of `count` equal sub-intervals, which keeps
    them away from the cut boundaries where frames are often mid-transition and
    motion-blurred.
    """
    if count <= 0 or scene.duration <= 0:
        return []
    step = scene.duration / count
    return [round(scene.start + step * (i + 0.5), 3) for i in range(count)]


def extract_frames(
    video_path: str | Path,
    scenes: list[SceneBoundary],
    output_dir: str | Path,
    settings: Settings,
) -> list[Frame]:
    """Extract representative JPEG frames for each scene."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budget = plan_frame_budget(scenes, settings.video_frames_per_scene, settings.video_max_frames)
    frames: list[Frame] = []

    for scene in scenes:
        count = budget.get(scene.id, 0)
        for index, timestamp in enumerate(sample_timestamps(scene, count)):
            destination = output_dir / f"scene_{scene.id:03d}_frame_{index:02d}.jpg"
            if extract_still(video_path, timestamp, destination, settings.video_frame_width):
                frames.append(
                    Frame(scene_id=scene.id, timestamp=timestamp, path=str(destination))
                )

    logger.info("extracted %d representative frame(s)", len(frames))
    return frames


def extract_still(
    video_path: str | Path, timestamp: float, destination: str | Path, width: int
) -> bool:
    """Extract a single frame. Returns False if ffmpeg produced nothing.

    A missing frame is not fatal — seeking past the last keyframe near the end
    of a file can legitimately yield no output, and the scene still has its
    other samples. Shared with the thumbnail generator, which needs exactly
    this behaviour for one frame.
    """
    video_path = Path(video_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg_utils.run(
            [
                "-y",
                "-ss", f"{timestamp:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                # -2 keeps the height even, which JPEG encoders require.
                "-vf", f"scale={width}:-2:flags=bicubic",
                "-q:v", "3",
                "-loglevel", "error",
                str(destination),
            ],
            timeout=120,
        )
    except FFmpegError as exc:
        logger.warning("frame extraction failed at %.2fs: %s", timestamp, exc)
        return False

    if not destination.is_file() or destination.stat().st_size == 0:
        logger.warning("frame extraction produced no output at %.2fs", timestamp)
        destination.unlink(missing_ok=True)
        return False
    return True


def frames_by_scene(frames: list[Frame]) -> dict[int, list[Frame]]:
    """Group extracted frames by their scene id, preserving timestamp order."""
    grouped: dict[int, list[Frame]] = {}
    for frame in frames:
        grouped.setdefault(frame.scene_id, []).append(frame)
    for scene_frames in grouped.values():
        scene_frames.sort(key=lambda f: f.timestamp)
    return grouped
