"""Ollama provider behaviour, with the HTTP layer stubbed.

These tests never contact a real Ollama instance. They exist because the Ollama
providers are the only code that talks to a network service, and the failure
modes (backend down, model not pulled, timeout) are exactly the ones a new user
hits first — so the error messages they produce need to be verified, not assumed.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from video_understanding.ai.providers.ollama import (
    OllamaClient,
    OllamaLLMProvider,
    OllamaVisionProvider,
)
from video_understanding.core.config import Settings
from video_understanding.core.errors import (
    ProviderError,
    ProviderUnavailableError,
    ResponseParseError,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ollama_host="http://localhost:11434",
        vision_model="qwen2.5vl:7b",
        llm_model="qwen3:8b",
        ollama_timeout=30.0,
    )


class _Response:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]


def _capture(monkeypatch, response, *, method: str = "post") -> dict:
    """Patch httpx and record the outgoing request."""
    captured: dict = {}

    def fake(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(httpx, method, fake)
    return captured


class TestOllamaClientChat:
    def test_returns_message_content(self, settings, monkeypatch):
        _capture(monkeypatch, _Response(200, {"message": {"content": "hello"}}))
        client = OllamaClient(settings)
        assert client.chat("m", [{"role": "user", "content": "hi"}]) == "hello"

    def test_sends_the_expected_payload(self, settings, monkeypatch):
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "ok"}}))
        OllamaClient(settings).chat("my-model", [{"role": "user", "content": "hi"}])

        assert captured["url"].endswith("/api/chat")
        body = captured["json"]
        assert body["model"] == "my-model"
        assert body["stream"] is False
        assert body["options"]["num_ctx"] == settings.ollama_num_ctx

    def test_json_mode_sets_the_format_field(self, settings, monkeypatch):
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "{}"}}))
        OllamaClient(settings).chat("m", [], json_mode=True)
        assert captured["json"]["format"] == "json"

    def test_json_mode_is_off_by_default(self, settings, monkeypatch):
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "x"}}))
        OllamaClient(settings).chat("m", [])
        assert "format" not in captured["json"]

    def test_trailing_slash_in_host_is_normalised(self, monkeypatch):
        settings = Settings(ollama_host="http://localhost:11434/")
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "x"}}))
        OllamaClient(settings).chat("m", [])
        assert "//api/chat" not in captured["url"]


class TestOllamaClientErrors:
    def test_connection_refused_explains_how_to_fix_it(self, settings, monkeypatch):
        _capture(monkeypatch, httpx.ConnectError("refused"))
        with pytest.raises(ProviderUnavailableError) as info:
            OllamaClient(settings).chat("m", [])

        message = str(info.value)
        assert "ollama serve" in message
        assert "host.docker.internal" in message  # the Docker case is the common trap

    def test_missing_model_tells_you_to_pull_it(self, settings, monkeypatch):
        _capture(monkeypatch, _Response(404, {"error": "model not found"}))
        with pytest.raises(ProviderUnavailableError, match="ollama pull my-model"):
            OllamaClient(settings).chat("my-model", [])

    def test_timeout_suggests_a_remedy(self, settings, monkeypatch):
        _capture(monkeypatch, httpx.TimeoutException("too slow"))
        with pytest.raises(ProviderError, match="timed out"):
            OllamaClient(settings).chat("m", [])

    def test_server_error_is_surfaced(self, settings, monkeypatch):
        _capture(monkeypatch, _Response(500, {}, text="internal error"))
        with pytest.raises(ProviderError, match="HTTP 500"):
            OllamaClient(settings).chat("m", [])

    def test_error_field_in_a_200_response(self, settings, monkeypatch):
        # Ollama can return 200 with an error body.
        _capture(monkeypatch, _Response(200, {"error": "out of memory"}))
        with pytest.raises(ProviderError, match="out of memory"):
            OllamaClient(settings).chat("m", [])


class TestOllamaHealth:
    def test_reports_ok_for_an_installed_model(self, settings, monkeypatch):
        _capture(
            monkeypatch,
            _Response(200, {"models": [{"name": "qwen3:8b"}]}),
            method="get",
        )
        ok, detail = OllamaClient(settings).health("qwen3:8b")
        assert ok is True and "available" in detail

    def test_matches_on_the_model_family(self, settings, monkeypatch):
        # A different tag of the same family counts as installed.
        _capture(
            monkeypatch,
            _Response(200, {"models": [{"name": "qwen3:14b"}]}),
            method="get",
        )
        assert OllamaClient(settings).health("qwen3:8b")[0] is True

    def test_reports_a_missing_model(self, settings, monkeypatch):
        _capture(monkeypatch, _Response(200, {"models": [{"name": "llama3:8b"}]}), method="get")
        ok, detail = OllamaClient(settings).health("qwen3:8b")
        assert ok is False
        assert "ollama pull qwen3:8b" in detail

    def test_unreachable_backend_is_reported_not_raised(self, settings, monkeypatch):
        _capture(monkeypatch, httpx.ConnectError("refused"), method="get")
        ok, detail = OllamaClient(settings).health("qwen3:8b")
        assert ok is False and "cannot reach" in detail


class TestOllamaLLMProvider:
    def test_generate_includes_the_system_prompt(self, settings, monkeypatch):
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "answer"}}))
        provider = OllamaLLMProvider(settings)

        assert provider.generate("question", system="be terse") == "answer"
        roles = [m["role"] for m in captured["json"]["messages"]]
        assert roles == ["system", "user"]

    def test_generate_without_a_system_prompt(self, settings, monkeypatch):
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "a"}}))
        OllamaLLMProvider(settings).generate("question")
        assert [m["role"] for m in captured["json"]["messages"]] == ["user"]

    def test_uses_the_configured_model(self, settings, monkeypatch):
        captured = _capture(monkeypatch, _Response(200, {"message": {"content": "a"}}))
        OllamaLLMProvider(settings).generate("q")
        assert captured["json"]["model"] == "qwen3:8b"


class TestOllamaVisionProvider:
    @pytest.fixture
    def frame(self, tmp_path):
        path = tmp_path / "frame.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg bytes")
        return path

    def _payload(self) -> dict:
        return {
            "message": {
                "content": json.dumps(
                    {
                        "description": "A person in a kitchen.",
                        "people": ["one adult"],
                        "objects": ["refrigerator"],
                        "actions": ["opening"],
                        "environment": "indoor kitchen",
                        "notable_events": [],
                        "significance": 0.8,
                        "mood": "calm",
                    }
                )
            }
        }

    def test_parses_a_well_formed_response(self, settings, monkeypatch, frame):
        _capture(monkeypatch, _Response(200, self._payload()))
        analysis = OllamaVisionProvider(settings).analyze_frames(
            [str(frame)], "", scene_id=3, start=1.0, end=5.0
        )

        assert analysis.scene_id == 3
        assert analysis.description == "A person in a kitchen."
        assert analysis.objects == ["refrigerator"]
        assert analysis.significance == 0.8

    def test_frames_are_base64_encoded(self, settings, monkeypatch, frame):
        captured = _capture(monkeypatch, _Response(200, self._payload()))
        OllamaVisionProvider(settings).analyze_frames([str(frame)], "", scene_id=1)

        images = captured["json"]["messages"][1]["images"]
        assert base64.b64decode(images[0]) == frame.read_bytes()

    def test_transcript_context_reaches_the_prompt(self, settings, monkeypatch, frame):
        captured = _capture(monkeypatch, _Response(200, self._payload()))
        OllamaVisionProvider(settings).analyze_frames(
            [str(frame)], "I am opening the fridge", scene_id=1
        )
        assert "opening the fridge" in captured["json"]["messages"][1]["content"]

    def test_no_frames_returns_a_placeholder(self, settings):
        analysis = OllamaVisionProvider(settings).analyze_frames([], "", scene_id=2)
        assert analysis.scene_id == 2
        assert "No frames" in analysis.description

    def test_unreadable_frames_do_not_call_the_model(self, settings, tmp_path):
        analysis = OllamaVisionProvider(settings).analyze_frames(
            [str(tmp_path / "missing.jpg")], "", scene_id=1
        )
        assert "could not be read" in analysis.description

    def test_non_json_response_raises_a_parse_error(self, settings, monkeypatch, frame):
        _capture(monkeypatch, _Response(200, {"message": {"content": "I see a kitchen."}}))
        with pytest.raises(ResponseParseError, match="scene 1"):
            OllamaVisionProvider(settings).analyze_frames([str(frame)], "", scene_id=1)

    def test_response_wrapped_in_prose_still_parses(self, settings, monkeypatch, frame):
        wrapped = {
            "message": {
                "content": "Here is the JSON:\n```json\n"
                + json.dumps({"description": "ok", "significance": 90})
                + "\n```"
            }
        }
        _capture(monkeypatch, _Response(200, wrapped))
        analysis = OllamaVisionProvider(settings).analyze_frames([str(frame)], "", scene_id=1)

        assert analysis.description == "ok"
        assert analysis.significance == 0.9  # rescaled from a 0-100 answer

    def test_a_json_array_response_is_rejected(self, settings, monkeypatch, frame):
        _capture(monkeypatch, _Response(200, {"message": {"content": "[1, 2, 3]"}}))
        with pytest.raises(ResponseParseError, match="expected an object"):
            OllamaVisionProvider(settings).analyze_frames([str(frame)], "", scene_id=1)

    def test_json_mode_is_requested(self, settings, monkeypatch, frame):
        captured = _capture(monkeypatch, _Response(200, self._payload()))
        OllamaVisionProvider(settings).analyze_frames([str(frame)], "", scene_id=1)
        assert captured["json"]["format"] == "json"
