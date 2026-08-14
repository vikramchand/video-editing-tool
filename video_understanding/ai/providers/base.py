"""Provider interfaces.

The whole point of this layer: nothing downstream of a provider knows whether
it is talking to a local Whisper model on a Mac CPU, an Ollama VLM, or a hosted
GPU endpoint added in a later version. Implement the Protocol, register it, and
the pipeline picks it up from configuration.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from video_understanding.core.errors import ResponseParseError
from video_understanding.core.models import SceneAnalysis, Transcript


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Speech-to-text over an audio file."""

    name: str
    model: str

    def transcribe(self, audio_path: str) -> Transcript: ...

    def health(self) -> tuple[bool, str]:
        """Return (ok, human-readable detail) without doing heavy work."""
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Visual analysis over representative frames."""

    name: str
    model: str

    def analyze_frames(self, frames: list[str], context: str = "") -> SceneAnalysis: ...

    def health(self) -> tuple[bool, str]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Text generation / reasoning."""

    name: str
    model: str

    def generate(self, prompt: str, *, system: str = "", json_mode: bool = False) -> str: ...

    def health(self) -> tuple[bool, str]: ...


# --- Shared JSON handling ---------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove `<think>` blocks emitted by reasoning models such as qwen3."""
    cleaned = _THINK_RE.sub("", text)
    # An unterminated <think> means the model never got to the answer.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Small local models wrap JSON in prose, markdown fences, or reasoning tags
    no matter how firmly the prompt says not to. This walks through the
    plausible shapes rather than trusting the model to have behaved.
    """
    if not text or not text.strip():
        raise ResponseParseError("model returned an empty response")

    cleaned = strip_reasoning(text)

    candidates: list[str] = [cleaned]
    candidates.extend(match.strip() for match in _FENCE_RE.findall(cleaned))
    balanced = _first_balanced(cleaned)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Trailing commas are the single most common malformation.
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue

    preview = cleaned[:300].replace("\n", " ")
    raise ResponseParseError(f"could not parse JSON from model response: {preview!r}")


def _first_balanced(text: str) -> str | None:
    """Return the first balanced {...} or [...] block, ignoring braces in strings."""
    start = None
    opener = closer = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            closer = "}" if char == "{" else "]"
            break
    if start is None:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
