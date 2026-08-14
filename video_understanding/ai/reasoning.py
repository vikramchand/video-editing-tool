"""Reasoning stage.

The reasoning model never sees pixels. It receives the *structured*
representation — metadata, transcript, per-scene visual analysis — and reasons
over it to produce the summary, highlights and editing recommendations.

That separation is deliberate: it keeps the expensive multimodal work confined
to a bounded number of sampled frames, and it means the reasoning model can be
swapped for a stronger text model without touching the vision path.

Every field the model returns is re-validated and clamped here. A local 8B model
will confidently emit a highlight ending at 900s in a 47s video, an action of
"delete", or a score of 85; none of that reaches the API.
"""

from __future__ import annotations

import logging
from typing import Any

from video_understanding.ai.prompts import (
    REASONING_SYSTEM_PROMPT,
    build_reasoning_prompt,
)
from video_understanding.ai.providers.base import extract_json
from video_understanding.core.errors import ProviderError, ResponseParseError
from video_understanding.core.models import (
    EditSuggestion,
    Highlight,
    Scene,
    Transcript,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

MAX_HIGHLIGHTS = 5
MAX_EDITS = 12
MAX_TRANSCRIPT_CHARS = 6000

VALID_ACTIONS = {"keep", "remove", "trim", "zoom", "caption", "broll"}

# Local models reach for these words; map them onto the closed action set
# rather than discarding an otherwise sensible suggestion.
ACTION_SYNONYMS = {
    "delete": "remove",
    "cut": "remove",
    "drop": "remove",
    "discard": "remove",
    "shorten": "trim",
    "tighten": "trim",
    "speed_up": "trim",
    "subtitle": "caption",
    "subtitles": "caption",
    "captions": "caption",
    "text": "caption",
    "b-roll": "broll",
    "b_roll": "broll",
    "broll_insert": "broll",
    "punch_in": "zoom",
    "zoom_in": "zoom",
    "crop": "zoom",
    "retain": "keep",
}


class ReasoningResult:
    """Everything the reasoning stage produces."""

    def __init__(
        self,
        *,
        title: str | None = None,
        summary: str = "",
        topics: list[str] | None = None,
        highlights: list[Highlight] | None = None,
        edit_suggestions: list[EditSuggestion] | None = None,
        warnings: list[str] | None = None,
        source: str = "llm",
    ) -> None:
        self.title = title
        self.summary = summary
        self.topics = topics or []
        self.highlights = highlights or []
        self.edit_suggestions = edit_suggestions or []
        self.warnings = warnings or []
        self.source = source


# --- Structured representation ---------------------------------------------


def format_metadata(metadata: VideoMetadata, filename: str = "") -> str:
    lines = [
        f"Filename: {filename or 'unknown'}",
        f"Duration: {metadata.duration:.1f} s",
        f"Resolution: {metadata.width}x{metadata.height} ({metadata.orientation})",
        f"Frame rate: {metadata.fps:.2f} fps",
        f"Has audio: {'yes' if metadata.has_audio else 'no'}",
    ]
    return "\n".join(lines)


def format_transcript(transcript: Transcript, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Timestamped transcript, truncated to protect the model's context window."""
    if not transcript.segments:
        return ""

    lines = [
        f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}"
        for segment in transcript.segments
    ]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text

    # Keep the beginning and the end; the middle of a rambling clip matters least.
    head_budget = int(max_chars * 0.6)
    tail_budget = max_chars - head_budget
    head: list[str] = []
    length = 0
    for line in lines:
        if length + len(line) > head_budget:
            break
        head.append(line)
        length += len(line) + 1

    tail: list[str] = []
    length = 0
    for line in reversed(lines[len(head) :]):
        if length + len(line) > tail_budget:
            break
        tail.append(line)
        length += len(line) + 1

    return "\n".join([*head, "[... transcript truncated ...]", *reversed(tail)])


def format_scenes(scenes: list[Scene]) -> str:
    """Render scenes in the shape the prompt (and the mock provider) expect."""
    blocks = []
    for scene in scenes:
        parts = [f"Scene {scene.id} [{scene.start:.1f}s - {scene.end:.1f}s]"]
        if scene.description:
            parts.append(f"  Description: {scene.description}")
        if scene.people:
            parts.append(f"  People: {', '.join(scene.people)}")
        if scene.objects:
            parts.append(f"  Objects: {', '.join(scene.objects)}")
        if scene.actions:
            parts.append(f"  Actions: {', '.join(scene.actions)}")
        if scene.environment:
            parts.append(f"  Environment: {scene.environment}")
        parts.append(f"  Visual importance: {scene.importance:.2f}")
        if scene.transcript_text:
            speech = scene.transcript_text
            if len(speech) > 400:
                speech = speech[:400].rsplit(" ", 1)[0] + "..."
            parts.append(f'  Speech: "{speech}"')
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def build_structured_representation(
    metadata: VideoMetadata,
    transcript: Transcript,
    scenes: list[Scene],
    filename: str = "",
) -> dict[str, str]:
    """The three text blocks handed to the reasoning model."""
    return {
        "metadata": format_metadata(metadata, filename),
        "transcript": format_transcript(transcript),
        "scenes": format_scenes(scenes),
    }


# --- Engine -----------------------------------------------------------------


class ReasoningEngine:
    """Turns the structured representation into editorial output."""

    def __init__(self, provider, *, strict: bool = False) -> None:
        self.provider = provider
        self.strict = strict

    def analyze(
        self,
        metadata: VideoMetadata,
        transcript: Transcript,
        scenes: list[Scene],
        filename: str = "",
    ) -> ReasoningResult:
        representation = build_structured_representation(metadata, transcript, scenes, filename)
        prompt = build_reasoning_prompt(
            metadata=representation["metadata"],
            transcript=representation["transcript"],
            scenes=representation["scenes"],
            duration=metadata.duration,
            max_highlights=MAX_HIGHLIGHTS,
            max_edits=MAX_EDITS,
        )

        try:
            raw = self.provider.generate(
                prompt, system=REASONING_SYSTEM_PROMPT, json_mode=True
            )
            if not raw or not raw.strip():
                raise ResponseParseError("reasoning model returned an empty response")
            data = extract_json(raw)
            if not isinstance(data, dict):
                raise ResponseParseError(
                    f"reasoning model returned {type(data).__name__}, expected an object"
                )
        except ProviderError as exc:
            if self.strict:
                raise
            logger.warning("reasoning failed (%s); falling back to heuristics", exc)
            result = heuristic_analysis(metadata, transcript, scenes)
            result.warnings.append(f"LLM reasoning unavailable, used heuristics instead: {exc}")
            return result

        result = _parse_reasoning(data, metadata.duration)

        # A model that answers with valid JSON but no usable content is no more
        # useful than one that failed; fall back rather than return an empty shell.
        if not result.summary.strip():
            fallback = heuristic_analysis(metadata, transcript, scenes)
            result.summary = fallback.summary
            result.warnings.append("Model returned no summary; used a heuristic summary.")
        if not result.highlights:
            result.highlights = heuristic_analysis(metadata, transcript, scenes).highlights
            result.warnings.append(
                "Model returned no highlights; derived them from scene importance."
            )

        return result


def _parse_reasoning(data: dict[str, Any], duration: float) -> ReasoningResult:
    warnings: list[str] = []

    highlights = _parse_highlights(data.get("highlights"), duration, warnings)
    edits = _parse_edits(data.get("edit_suggestions"), duration, warnings)

    topics = data.get("topics") or []
    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split(",") if t.strip()]
    topics = [str(t).strip() for t in topics if str(t).strip()][:10]

    title = data.get("title")
    summary = data.get("summary") or ""
    if isinstance(summary, list):  # some models answer with a list of sentences
        summary = " ".join(str(s) for s in summary)

    return ReasoningResult(
        title=str(title).strip() if title else None,
        summary=str(summary).strip(),
        topics=topics,
        highlights=highlights,
        edit_suggestions=edits,
        warnings=warnings,
        source="llm",
    )


def _clamp_range(item: dict, duration: float) -> tuple[float, float] | None:
    """Validate a start/end pair against the real video duration."""
    try:
        start = float(item.get("start", 0))
        end = float(item.get("end", 0))
    except (TypeError, ValueError):
        return None

    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end <= start:
        return None
    return round(start, 3), round(end, 3)


def _clamp_unit(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1.0:
        number = number / 100.0 if number > 10.0 else number / 10.0
    return round(max(0.0, min(1.0, number)), 3)


def _parse_highlights(raw: Any, duration: float, warnings: list[str]) -> list[Highlight]:
    if not isinstance(raw, list):
        return []

    highlights: list[Highlight] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        window = _clamp_range(item, duration)
        if window is None:
            warnings.append(f"Dropped a highlight with an invalid time range: {item!r:.120}")
            continue
        highlights.append(
            Highlight(
                start=window[0],
                end=window[1],
                reason=str(item.get("reason") or "Notable moment.").strip(),
                score=_clamp_unit(item.get("score"), 0.5),
                title=str(item["title"]).strip() if item.get("title") else None,
            )
        )

    highlights.sort(key=lambda h: h.score, reverse=True)
    return highlights[:MAX_HIGHLIGHTS]


def _parse_edits(raw: Any, duration: float, warnings: list[str]) -> list[EditSuggestion]:
    if not isinstance(raw, list):
        return []

    edits: list[EditSuggestion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        window = _clamp_range(item, duration)
        if window is None:
            warnings.append(f"Dropped an edit suggestion with an invalid time range: {item!r:.120}")
            continue

        action = str(item.get("action") or "").strip().lower().replace(" ", "_")
        action = ACTION_SYNONYMS.get(action, action)
        if action not in VALID_ACTIONS:
            warnings.append(f"Dropped an edit suggestion with unknown action {action!r}")
            continue

        edits.append(
            EditSuggestion(
                start=window[0],
                end=window[1],
                action=action,  # type: ignore[arg-type]
                reason=str(item.get("reason") or "No reason given.").strip(),
                confidence=_clamp_unit(item.get("confidence"), 0.5),
            )
        )

    edits.sort(key=lambda e: e.start)
    return edits[:MAX_EDITS]


# --- Heuristic fallback -----------------------------------------------------


def heuristic_analysis(
    metadata: VideoMetadata,
    transcript: Transcript,
    scenes: list[Scene],
) -> ReasoningResult:
    """Produce a usable result with no LLM at all.

    Used when the reasoning model is unreachable or disabled. It is intentionally
    simple and deterministic: rank scenes by visual importance and speech
    density, and flag long silent stretches for removal.
    """
    duration = metadata.duration

    summary = _heuristic_summary(metadata, transcript, scenes)
    highlights = _heuristic_highlights(scenes, duration)
    edits = _heuristic_edits(transcript, scenes, duration)

    return ReasoningResult(
        title=None,
        summary=summary,
        topics=[],
        highlights=highlights,
        edit_suggestions=edits,
        source="heuristic",
    )


def _heuristic_summary(
    metadata: VideoMetadata, transcript: Transcript, scenes: list[Scene]
) -> str:
    parts = [
        f"A {metadata.duration:.0f}-second {metadata.orientation} video "
        f"with {len(scenes)} detected scene{'s' if len(scenes) != 1 else ''}."
    ]
    described = [s for s in scenes if s.description and "no visual analysis" not in s.description]
    if described:
        best = max(described, key=lambda s: s.importance)
        parts.append(f"The most prominent moment: {best.description}")
    if transcript.segments:
        opening = transcript.text[:200].strip()
        parts.append(f'Speech begins: "{opening}{"..." if len(transcript.text) > 200 else ""}"')
    else:
        parts.append("No speech was detected.")
    return " ".join(parts)


def _heuristic_highlights(scenes: list[Scene], duration: float) -> list[Highlight]:
    if not scenes:
        return []

    scored: list[tuple[float, Scene]] = []
    for scene in scenes:
        # Visual importance, nudged by whether anything is said and by length.
        score = scene.importance
        if scene.transcript_text:
            score += 0.15
        if scene.duration >= 3.0:
            score += 0.05
        scored.append((min(1.0, score), scene))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        Highlight(
            start=scene.start,
            end=min(scene.end, duration),
            reason=(
                "High visual importance with accompanying speech."
                if scene.transcript_text
                else "Highest visual importance among the detected scenes."
            ),
            score=round(score, 3),
            title=f"Scene {scene.id}",
        )
        for score, scene in scored[:3]
    ]


def _heuristic_edits(
    transcript: Transcript, scenes: list[Scene], duration: float
) -> list[EditSuggestion]:
    """Flag silence gaps for removal and captionable speech for captions."""
    edits: list[EditSuggestion] = []
    min_silence = 2.0

    if transcript.segments:
        # Silence before the first word.
        first = transcript.segments[0]
        if first.start >= min_silence:
            edits.append(
                EditSuggestion(
                    start=0.0,
                    end=round(first.start, 3),
                    action="remove",
                    reason=f"{first.start:.1f}s of silence before the first word.",
                    confidence=0.8,
                )
            )

        # Silence between utterances.
        for previous, current in zip(transcript.segments, transcript.segments[1:], strict=False):
            gap = current.start - previous.end
            if gap >= min_silence:
                edits.append(
                    EditSuggestion(
                        start=round(previous.end, 3),
                        end=round(current.start, 3),
                        action="remove",
                        reason=f"{gap:.1f}s silent gap between spoken segments.",
                        confidence=0.7,
                    )
                )

        # Trailing silence.
        last = transcript.segments[-1]
        if duration - last.end >= min_silence:
            edits.append(
                EditSuggestion(
                    start=round(last.end, 3),
                    end=round(duration, 3),
                    action="remove",
                    reason=f"{duration - last.end:.1f}s of silence after the last word.",
                    confidence=0.8,
                )
            )

        edits.append(
            EditSuggestion(
                start=round(first.start, 3),
                end=round(last.end, 3),
                action="caption",
                reason="Speech is present; burned-in captions improve silent-autoplay retention.",
                confidence=0.6,
            )
        )

    if scenes:
        best = max(scenes, key=lambda s: s.importance)
        if best.importance >= 0.6:
            edits.append(
                EditSuggestion(
                    start=best.start,
                    end=min(best.end, duration),
                    action="zoom",
                    reason="Highest-importance scene; a punch-in adds emphasis.",
                    confidence=0.5,
                )
            )

    edits.sort(key=lambda e: e.start)
    return edits[:MAX_EDITS]
