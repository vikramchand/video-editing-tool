"""Prompt templates for the vision and reasoning stages.

Two rules shaped these prompts, both learned from small local models:

1. **Flat schemas beat nested ones.** A 7B VLM reliably fills a flat object with
   six keys; it drifts badly on nested structures.
2. **Show the exact JSON skeleton.** Describing fields in prose produces
   inventive key names. Showing the literal shape does not.

Parsing is still defensive (see `providers.base.extract_json` and the coercing
validators on `SceneAnalysis`) because "still" is doing a lot of work in
"models still get this wrong".
"""

from __future__ import annotations

VISION_SYSTEM_PROMPT = """You are a precise visual analyst for a video understanding system.
You look at still frames sampled from one continuous scene of a video and report what is
literally visible. You never speculate about things you cannot see, and you never invent
names for people. You always answer with a single valid JSON object and nothing else."""


VISION_USER_PROMPT = """These {frame_count} image(s) are frames sampled from ONE scene of a video.
The scene runs from {start:.1f}s to {end:.1f}s.{context}

Describe what is happening in this scene.

Answer with EXACTLY this JSON object and no other text:

{{
  "description": "one or two sentences describing what happens in this scene",
  "people": ["short descriptions such as 'one adult in a blue shirt'"],
  "objects": ["visible physical objects"],
  "actions": ["actions being performed, such as 'walking', 'talking'"],
  "environment": "where this takes place, such as 'indoor kitchen'",
  "notable_events": ["specific visual events worth remembering, if any"],
  "significance": 0.5,
  "mood": "the overall tone, such as 'calm' or 'energetic'"
}}

Rules:
- "significance" is a number from 0.0 to 1.0 rating how important this scene looks
  relative to a typical moment in a short video. Use 0.8+ only for a clear main action.
- Do not name people. Describe them.
- Use empty arrays when nothing applies. Never write null.
- Output only the JSON object."""


REASONING_SYSTEM_PROMPT = """You are a video editor's analytical assistant. You reason over a
structured description of a video - its metadata, its transcript, and a per-scene visual
analysis - and produce a summary, highlights, and concrete editing recommendations.

You never see the raw video. You work only from the structured information given to you, and
you never invent events that are not supported by it. You always answer with a single valid
JSON object and nothing else."""


REASONING_USER_PROMPT = """Analyse this video and produce an editorial assessment.

=== VIDEO METADATA ===
{metadata}

=== TRANSCRIPT ===
{transcript}

=== SCENE ANALYSIS ===
{scenes}

=== TASK ===
Produce EXACTLY this JSON object and no other text:

{{
  "title": "a short descriptive title for the video",
  "summary": "2-4 sentences describing what happens in the video, in order",
  "topics": ["the main subjects or themes"],
  "highlights": [
    {{"start": 0.0, "end": 5.0, "reason": "why this moment matters", "score": 0.8,
      "title": "short label"}}
  ],
  "edit_suggestions": [
    {{"start": 0.0, "end": 3.0, "action": "remove", "reason": "why", "confidence": 0.7}}
  ]
}}

Rules for highlights:
- Between 1 and {max_highlights} highlights, ordered by score descending.
- "score" is 0.0-1.0. Highlights must fall inside 0 to {duration:.1f} seconds.
- Prefer moments where the visual action and the speech agree that something happens.

Rules for edit_suggestions:
- "action" MUST be one of: "keep", "remove", "trim", "zoom", "caption", "broll".
- Use "remove" for dead air, long silences, and rambling; "trim" to tighten a scene that
  runs long; "caption" where on-screen text would help; "zoom" for a punch-in on a key
  moment; "broll" where supporting footage would cover a weak visual.
- Every suggestion needs a start and end inside 0 to {duration:.1f} seconds.
- Give at most {max_edits} suggestions, and only ones you can justify from the material.
- Do not suggest removing the single most important moment in the video.

Output only the JSON object."""


def build_vision_prompt(
    frame_count: int,
    start: float,
    end: float,
    transcript_context: str = "",
) -> str:
    """Render the per-scene vision prompt."""
    context = ""
    if transcript_context.strip():
        # Speech is a strong disambiguator for what a frame is showing.
        snippet = transcript_context.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
        context = f'\n\nSomeone says this during the scene: "{snippet}"'

    return VISION_USER_PROMPT.format(
        frame_count=frame_count,
        start=start,
        end=end,
        context=context,
    )


def build_reasoning_prompt(
    metadata: str,
    transcript: str,
    scenes: str,
    duration: float,
    max_highlights: int = 5,
    max_edits: int = 12,
) -> str:
    """Render the reasoning prompt from the structured representation."""
    return REASONING_USER_PROMPT.format(
        metadata=metadata,
        transcript=transcript or "(no speech detected in this video)",
        scenes=scenes or "(no scene analysis available)",
        duration=duration,
        max_highlights=max_highlights,
        max_edits=max_edits,
    )
