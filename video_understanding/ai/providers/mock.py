"""Deterministic stand-in providers.

Two jobs:

1. **Tests.** Unit tests must never need a running model, so every AI stage can
   be driven by these.
2. **`PROVIDER=mock` demo mode.** Someone evaluating the project can run the
   entire pipeline — real FFmpeg, real scene detection, real frame sampling —
   before installing a single model, and see the exact JSON shape they will get.

Output is derived from the inputs and is stable across runs, so tests can assert
on it. It travels through the same parsing and validation path as a real model's
output, which is what makes it a useful test double rather than a shortcut.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from video_understanding.core.models import SceneAnalysis, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"Duration:\s*([0-9]+\.?[0-9]*)\s*s", re.IGNORECASE)
_SCENE_LINE_RE = re.compile(r"Scene\s+(\d+)\s*\[([0-9.]+)s?\s*-\s*([0-9.]+)s?\]", re.IGNORECASE)


def _seed(value: str) -> float:
    """Stable pseudo-random float in [0, 1) derived from a string."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockTranscriptionProvider:
    """Returns a fixed transcript spread across the audio's duration."""

    name = "mock"

    def __init__(self, settings=None) -> None:
        self.model = "mock-whisper"
        self.settings = settings

    def transcribe(self, audio_path: str) -> Transcript:
        from video_understanding.video.audio import audio_duration  # noqa: PLC0415

        duration = audio_duration(audio_path) or 10.0
        lines = [
            "This is mock transcription output.",
            "No speech model was used to produce it.",
            "Set TRANSCRIPTION_PROVIDER=whisper for real transcription.",
        ]
        step = duration / len(lines)
        segments = [
            TranscriptSegment(
                start=round(index * step, 3),
                end=round(min((index + 1) * step, duration), 3),
                text=text,
                confidence=0.99,
            )
            for index, text in enumerate(lines)
        ]
        logger.info("mock transcription: %d segment(s) over %.1fs", len(segments), duration)
        return Transcript(
            language="en", segments=segments, provider=self.name, model=self.model
        )

    def health(self) -> tuple[bool, str]:
        return True, "mock transcription provider (no model required)"


class MockVisionProvider:
    """Produces plausible, deterministic scene analysis from frame filenames."""

    name = "mock"

    def __init__(self, settings=None) -> None:
        self.model = "mock-vision"
        self.settings = settings

    def analyze_frames(
        self,
        frames: list[str],
        context: str = "",
        *,
        scene_id: int = 0,
        start: float = 0.0,
        end: float = 0.0,
    ) -> SceneAnalysis:
        key = f"{scene_id}:{len(frames)}:{Path(frames[0]).name if frames else 'none'}"
        significance = round(0.35 + _seed(key) * 0.5, 2)

        description = (
            f"Mock analysis of scene {scene_id} "
            f"({start:.1f}s-{end:.1f}s) from {len(frames)} sampled frame(s)."
        )
        if context.strip():
            description += " Speech was present during this scene."

        return SceneAnalysis(
            scene_id=scene_id,
            description=description,
            people=["one adult"] if significance > 0.5 else [],
            objects=["mock object"],
            actions=["talking"] if context.strip() else ["moving"],
            environment="mock environment",
            notable_events=[f"mock event in scene {scene_id}"] if significance > 0.7 else [],
            significance=significance,
            mood="neutral",
        )

    def health(self) -> tuple[bool, str]:
        return True, "mock vision provider (no model required)"


class MockLLMProvider:
    """Returns schema-valid JSON so the real parsing path is exercised."""

    name = "mock"

    def __init__(self, settings=None) -> None:
        self.model = "mock-llm"
        self.settings = settings

    def generate(self, prompt: str, *, system: str = "", json_mode: bool = False) -> str:
        duration = 10.0
        match = _DURATION_RE.search(prompt)
        if match:
            try:
                duration = float(match.group(1))
            except ValueError:
                pass

        scenes = [
            (int(sid), float(start), float(end))
            for sid, start, end in _SCENE_LINE_RE.findall(prompt)
        ]

        highlights = []
        edits = []
        if scenes:
            # Highlight the longest scene; that is a defensible mock heuristic.
            longest = max(scenes, key=lambda s: s[2] - s[1])
            highlights.append(
                {
                    "start": longest[1],
                    "end": longest[2],
                    "reason": "Longest continuous scene in the video (mock reasoning).",
                    "score": 0.75,
                    "title": f"Scene {longest[0]}",
                }
            )
            first = scenes[0]
            if first[1] < 1.0 and (first[2] - first[1]) > 2.0:
                edits.append(
                    {
                        "start": 0.0,
                        "end": round(min(1.0, first[2]), 3),
                        "action": "trim",
                        "reason": "Tighten the opening (mock reasoning).",
                        "confidence": 0.5,
                    }
                )
            edits.append(
                {
                    "start": longest[1],
                    "end": longest[2],
                    "action": "caption",
                    "reason": "Add captions over the main moment (mock reasoning).",
                    "confidence": 0.6,
                }
            )
        else:
            highlights.append(
                {
                    "start": 0.0,
                    "end": round(duration, 3),
                    "reason": "Whole video (mock reasoning).",
                    "score": 0.5,
                    "title": "Full clip",
                }
            )

        return json.dumps(
            {
                "title": "Mock Video Analysis",
                "summary": (
                    "This summary was produced by the mock LLM provider, not a real model. "
                    f"The video is {duration:.1f} seconds long and contains {len(scenes)} "
                    "detected scene(s). Set LLM_PROVIDER=ollama for real reasoning."
                ),
                "topics": ["mock", "demo"],
                "highlights": highlights,
                "edit_suggestions": edits,
            }
        )

    def health(self) -> tuple[bool, str]:
        return True, "mock LLM provider (no model required)"


# --- Null providers: explicitly skip a stage --------------------------------


class NullTranscriptionProvider:
    """Skips transcription entirely (`TRANSCRIPTION_PROVIDER=null`)."""

    name = "null"

    def __init__(self, settings=None) -> None:
        self.model = "none"

    def transcribe(self, audio_path: str) -> Transcript:
        return Transcript(provider=self.name, model=self.model)

    def health(self) -> tuple[bool, str]:
        return True, "transcription disabled"


class NullVisionProvider:
    """Skips visual analysis (`VISION_PROVIDER=null`)."""

    name = "null"

    def __init__(self, settings=None) -> None:
        self.model = "none"

    def analyze_frames(
        self,
        frames: list[str],
        context: str = "",
        *,
        scene_id: int = 0,
        start: float = 0.0,
        end: float = 0.0,
    ) -> SceneAnalysis:
        return SceneAnalysis(scene_id=scene_id, description="", significance=0.5)

    def health(self) -> tuple[bool, str]:
        return True, "vision analysis disabled"


class NullLLMProvider:
    """Skips LLM reasoning (`LLM_PROVIDER=null`); heuristics take over."""

    name = "null"

    def __init__(self, settings=None) -> None:
        self.model = "none"

    def generate(self, prompt: str, *, system: str = "", json_mode: bool = False) -> str:
        return ""

    def health(self) -> tuple[bool, str]:
        return True, "LLM reasoning disabled (heuristic fallback in use)"
