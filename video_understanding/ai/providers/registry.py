"""Provider registry.

Maps the configured provider names onto implementations. Adding a hosted GPU
backend later means writing a class that satisfies the Protocol and adding one
line to the relevant table — no pipeline code changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from video_understanding.ai.providers.mock import (
    MockLLMProvider,
    MockTranscriptionProvider,
    MockVisionProvider,
    NullLLMProvider,
    NullTranscriptionProvider,
    NullVisionProvider,
)
from video_understanding.core.config import Settings
from video_understanding.core.errors import ProviderError

logger = logging.getLogger(__name__)


def _ollama_llm(settings: Settings):
    from video_understanding.ai.providers.ollama import OllamaLLMProvider  # noqa: PLC0415

    return OllamaLLMProvider(settings)


def _ollama_vision(settings: Settings):
    from video_understanding.ai.providers.ollama import OllamaVisionProvider  # noqa: PLC0415

    return OllamaVisionProvider(settings)


def _whisper(settings: Settings):
    from video_understanding.ai.providers.whisper import (  # noqa: PLC0415
        WhisperTranscriptionProvider,
    )

    return WhisperTranscriptionProvider(settings)


TRANSCRIPTION_PROVIDERS: dict[str, Callable[[Settings], Any]] = {
    "whisper": _whisper,
    "mock": MockTranscriptionProvider,
    "null": NullTranscriptionProvider,
}

VISION_PROVIDERS: dict[str, Callable[[Settings], Any]] = {
    "ollama": _ollama_vision,
    "mock": MockVisionProvider,
    "null": NullVisionProvider,
}

LLM_PROVIDERS: dict[str, Callable[[Settings], Any]] = {
    "ollama": _ollama_llm,
    "mock": MockLLMProvider,
    "null": NullLLMProvider,
}


def _build(table: dict[str, Callable[[Settings], Any]], name: str, settings: Settings, kind: str):
    factory = table.get(name)
    if factory is None:
        raise ProviderError(
            f"unknown {kind} provider '{name}'. Available: {', '.join(sorted(table))}"
        )
    return factory(settings)


def get_transcription_provider(settings: Settings):
    return _build(
        TRANSCRIPTION_PROVIDERS, settings.transcription_provider, settings, "transcription"
    )


def get_vision_provider(settings: Settings):
    return _build(VISION_PROVIDERS, settings.vision_provider, settings, "vision")


def get_llm_provider(settings: Settings):
    return _build(LLM_PROVIDERS, settings.llm_provider, settings, "llm")


def health_report(settings: Settings) -> dict[str, str]:
    """Probe each configured provider for the `/health` endpoint."""
    report: dict[str, str] = {}
    probes = (
        ("transcription", get_transcription_provider),
        ("vision", get_vision_provider),
        ("llm", get_llm_provider),
    )
    for kind, getter in probes:
        try:
            ok, detail = getter(settings).health()
            report[kind] = ("ok: " if ok else "unavailable: ") + detail
        except Exception as exc:  # noqa: BLE001 - health must never raise
            report[kind] = f"error: {exc}"
    return report
