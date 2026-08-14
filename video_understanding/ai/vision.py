"""Vision analysis stage.

Sends each scene's representative frames to the vision provider and turns the
result into `Scene` objects fused with the transcript for that time window.
"""

from __future__ import annotations

import logging

from video_understanding.core.errors import ProviderError, ProviderUnavailableError
from video_understanding.core.models import (
    Frame,
    Scene,
    SceneAnalysis,
    SceneBoundary,
    Transcript,
)
from video_understanding.video.frames import frames_by_scene

logger = logging.getLogger(__name__)

ProgressCallback = "callable taking (completed: int, total: int) -> None"


class VisionAnalyzer:
    """Runs the vision provider over detected scenes."""

    def __init__(self, provider, *, strict: bool = False) -> None:
        self.provider = provider
        # strict=True turns a single scene failure into a job failure.
        self.strict = strict

    def analyze(
        self,
        boundaries: list[SceneBoundary],
        frames: list[Frame],
        transcript: Transcript,
        on_progress=None,
    ) -> tuple[list[Scene], list[str]]:
        """Analyse every scene. Returns (scenes, warnings)."""
        grouped = frames_by_scene(frames)
        scenes: list[Scene] = []
        warnings: list[str] = []
        backend_down = False

        for position, boundary in enumerate(boundaries, start=1):
            scene_frames = grouped.get(boundary.id, [])
            speech = transcript.text_between(boundary.start, boundary.end)

            if backend_down or not scene_frames:
                analysis = SceneAnalysis(scene_id=boundary.id)
                if not scene_frames and not backend_down:
                    logger.info("scene %d has no frames; skipping vision", boundary.id)
            else:
                try:
                    analysis = self._analyze_scene(boundary, scene_frames, speech)
                except ProviderUnavailableError as exc:
                    # The backend is down: every remaining scene would fail the
                    # same way, and each failure costs a full timeout. Stop.
                    if self.strict:
                        raise
                    logger.warning("vision backend unavailable; skipping remaining scenes: %s", exc)
                    warnings.append(f"Visual analysis unavailable: {exc}")
                    backend_down = True
                    analysis = SceneAnalysis(scene_id=boundary.id)
                except ProviderError as exc:
                    if self.strict:
                        raise
                    logger.warning("vision failed for scene %d: %s", boundary.id, exc)
                    warnings.append(f"Scene {boundary.id}: visual analysis failed ({exc})")
                    analysis = SceneAnalysis(scene_id=boundary.id)

            scenes.append(_fuse(boundary, analysis, scene_frames, speech))

            if on_progress is not None:
                on_progress(position, len(boundaries))

        return scenes, warnings

    def _analyze_scene(
        self, boundary: SceneBoundary, scene_frames: list[Frame], speech: str
    ) -> SceneAnalysis:
        paths = [frame.path for frame in scene_frames]
        analyze = self.provider.analyze_frames
        try:
            # Providers in this repo accept scene context as keyword arguments;
            # a bare Protocol implementation only has to accept (frames, context).
            return analyze(
                paths,
                speech,
                scene_id=boundary.id,
                start=boundary.start,
                end=boundary.end,
            )
        except TypeError:
            analysis = analyze(paths, speech)
            analysis.scene_id = boundary.id
            return analysis


def _fuse(
    boundary: SceneBoundary,
    analysis: SceneAnalysis,
    scene_frames: list[Frame],
    speech: str,
) -> Scene:
    """Combine boundary, visual analysis and speech into one `Scene`."""
    description = analysis.description.strip()
    if not description:
        description = (
            f"Scene from {boundary.start:.1f}s to {boundary.end:.1f}s (no visual analysis)."
        )

    return Scene(
        id=boundary.id,
        start=boundary.start,
        end=boundary.end,
        description=description,
        people=_dedupe(analysis.people),
        objects=_dedupe(analysis.objects),
        actions=_dedupe(analysis.actions),
        environment=analysis.environment,
        importance=analysis.significance,
        keyframes=[frame.timestamp for frame in scene_frames],
        transcript_text=speech,
    )


def _dedupe(values: list[str]) -> list[str]:
    """Case-insensitive de-duplication that preserves order and original casing."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def aggregate_entities(scenes: list[Scene]) -> tuple[list[str], list[str], list[str]]:
    """Roll per-scene people/objects/actions up to the video level."""
    people = _dedupe([p for scene in scenes for p in scene.people])
    objects = _dedupe([o for scene in scenes for o in scene.objects])
    actions = _dedupe([a for scene in scenes for a in scene.actions])
    return people, objects, actions
