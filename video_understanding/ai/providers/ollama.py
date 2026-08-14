"""Ollama-backed LLM and vision providers.

Ollama runs on the *host* Mac so it can reach the GPU via Metal. When this API
runs in Docker, `OLLAMA_HOST` points at `http://host.docker.internal:11434`;
running natively it is `http://localhost:11434`. Nothing else changes.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from video_understanding.ai.prompts import (
    VISION_SYSTEM_PROMPT,
    build_vision_prompt,
)
from video_understanding.ai.providers.base import extract_json
from video_understanding.core.config import Settings
from video_understanding.core.errors import (
    ProviderError,
    ProviderUnavailableError,
    ResponseParseError,
)
from video_understanding.core.models import SceneAnalysis

logger = logging.getLogger(__name__)


class OllamaClient:
    """Minimal Ollama HTTP client covering the two endpoints we need."""

    def __init__(self, settings: Settings) -> None:
        self.host = settings.ollama_host.rstrip("/")
        self.timeout = settings.ollama_timeout
        self.num_ctx = settings.ollama_num_ctx
        self.temperature = settings.ollama_temperature

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            # Ollama constrains sampling to valid JSON, which removes most
            # (not all) of the malformed-response problem.
            payload["format"] = "json"

        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=timeout or self.timeout,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"cannot reach Ollama at {self.host}. Is `ollama serve` running? "
                "In Docker, OLLAMA_HOST should be http://host.docker.internal:11434",
                provider="ollama",
                model=model,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama timed out after {timeout or self.timeout:.0f}s using model '{model}'. "
                "Try a smaller model or raise OLLAMA_TIMEOUT.",
                provider="ollama",
                model=model,
            ) from exc

        if response.status_code == 404:
            raise ProviderUnavailableError(
                f"model '{model}' is not installed in Ollama. Run: ollama pull {model}",
                provider="ollama",
                model=model,
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}",
                provider="ollama",
                model=model,
            )

        data = response.json()
        if "error" in data:
            raise ProviderError(f"Ollama error: {data['error']}", provider="ollama", model=model)
        return (data.get("message") or {}).get("content", "")

    def list_models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"cannot reach Ollama at {self.host}: {exc}", provider="ollama"
            ) from exc
        return [m.get("name", "") for m in response.json().get("models", [])]

    def health(self, model: str) -> tuple[bool, str]:
        try:
            installed = self.list_models()
        except ProviderUnavailableError as exc:
            return False, str(exc)

        # Ollama reports names as `family:tag`; accept a bare family match too.
        base = model.split(":")[0]
        if model in installed or any(name.split(":")[0] == base for name in installed):
            return True, f"ollama at {self.host}, model '{model}' available"
        return False, (
            f"ollama is running at {self.host} but model '{model}' is not installed "
            f"(run: ollama pull {model})"
        )


class OllamaLLMProvider:
    """Text reasoning through Ollama."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self.client = OllamaClient(settings)

    def generate(self, prompt: str, *, system: str = "", json_mode: bool = False) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.info("reasoning with ollama model '%s' (%d chars)", self.model, len(prompt))
        return self.client.chat(self.model, messages, json_mode=json_mode)

    def health(self) -> tuple[bool, str]:
        return self.client.health(self.model)


class OllamaVisionProvider:
    """Frame analysis through an Ollama vision-language model."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.vision_model
        self.client = OllamaClient(settings)

    def analyze_frames(
        self,
        frames: list[str],
        context: str = "",
        *,
        scene_id: int = 0,
        start: float = 0.0,
        end: float = 0.0,
    ) -> SceneAnalysis:
        if not frames:
            return SceneAnalysis(
                scene_id=scene_id, description="No frames available for this scene."
            )

        images = [_encode_image(path) for path in frames]
        images = [img for img in images if img]
        if not images:
            return SceneAnalysis(scene_id=scene_id, description="Frames could not be read.")

        prompt = build_vision_prompt(
            frame_count=len(images), start=start, end=end, transcript_context=context
        )
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": images},
        ]

        logger.info(
            "analysing scene %d with %d frame(s) via '%s'", scene_id, len(images), self.model
        )
        raw = self.client.chat(self.model, messages, json_mode=True)

        try:
            data = extract_json(raw)
        except ResponseParseError as exc:
            raise ResponseParseError(
                f"vision model '{self.model}' returned invalid JSON "
                f"for scene {scene_id}: {exc}",
                provider="ollama",
                model=self.model,
            ) from exc

        if not isinstance(data, dict):
            raise ResponseParseError(
                f"vision model returned {type(data).__name__}, expected an object",
                provider="ollama",
                model=self.model,
            )

        data["scene_id"] = scene_id
        return SceneAnalysis.model_validate(data)

    def health(self) -> tuple[bool, str]:
        return self.client.health(self.model)


def _encode_image(path: str) -> str | None:
    """Base64-encode an image for the Ollama API."""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("could not read frame %s: %s", path, exc)
        return None
