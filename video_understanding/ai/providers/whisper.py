"""Local speech-to-text via faster-whisper.

faster-whisper (CTranslate2) is the practical choice on Apple Silicon today:
there is no Metal backend for it, but int8 CPU inference on an M-series chip
still transcribes a 60-second clip in a handful of seconds with the `small`
model, and it avoids the PyTorch install that openai-whisper drags in.

The import is deferred to first use so the package stays importable — and the
test suite stays runnable — without the optional `whisper` extra installed.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from video_understanding.core.config import Settings, get_hardware
from video_understanding.core.errors import ProviderError, ProviderUnavailableError
from video_understanding.core.models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperTranscriptionProvider:
    """Speech transcription with faster-whisper."""

    name = "whisper"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.whisper_model
        self._model = None
        self._lock = threading.Lock()

    def _resolve_runtime(self) -> tuple[str, str]:
        """Pick device and compute type, honouring explicit configuration."""
        device = self.settings.whisper_device
        compute_type = self.settings.whisper_compute_type
        auto_device, auto_compute = get_hardware().whisper_defaults()
        return (
            auto_device if device == "auto" else device,
            auto_compute if compute_type == "auto" else compute_type,
        )

    def _load(self):
        """Load the model once, lazily, and share it across worker threads."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "faster-whisper is not installed. Install it with: "
                    "pip install 'video-understanding[whisper]'",
                    provider="whisper",
                    model=self.model,
                ) from exc

            device, compute_type = self._resolve_runtime()
            logger.info(
                "loading whisper model '%s' (device=%s, compute_type=%s)",
                self.model, device, compute_type,
            )
            try:
                self._model = WhisperModel(self.model, device=device, compute_type=compute_type)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as ProviderError
                raise ProviderError(
                    f"could not load whisper model '{self.model}': {exc}",
                    provider="whisper",
                    model=self.model,
                ) from exc
            return self._model

    def transcribe(self, audio_path: str) -> Transcript:
        path = Path(audio_path)
        if not path.is_file():
            raise ProviderError(
                f"audio file not found: {path}", provider="whisper", model=self.model
            )

        model = self._load()
        logger.info("transcribing %s", path.name)

        try:
            segments, info = model.transcribe(
                str(path),
                language=self.settings.whisper_language,
                beam_size=5,
                vad_filter=True,  # drops silence, which also improves timestamps
                vad_parameters={"min_silence_duration_ms": 500},
            )
            parsed = [
                TranscriptSegment(
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    text=segment.text.strip(),
                    confidence=_confidence_of(segment),
                )
                for segment in segments
                if segment.text and segment.text.strip()
            ]
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as ProviderError
            raise ProviderError(
                f"transcription failed: {exc}", provider="whisper", model=self.model
            ) from exc

        transcript = Transcript(
            language=getattr(info, "language", None),
            segments=parsed,
            provider=self.name,
            model=self.model,
        )
        logger.info(
            "transcribed %d segment(s), language=%s", len(parsed), transcript.language
        )
        return transcript

    def health(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401,PLC0415
        except ImportError:
            return False, (
                "faster-whisper is not installed "
                "(pip install 'video-understanding[whisper]')"
            )
        device, compute_type = self._resolve_runtime()
        loaded = "loaded" if self._model is not None else "not yet loaded"
        return True, f"faster-whisper '{self.model}' ({device}/{compute_type}, {loaded})"


def _confidence_of(segment: object) -> float | None:
    """faster-whisper reports avg_logprob; map it into a rough 0-1 confidence."""
    logprob = getattr(segment, "avg_logprob", None)
    if logprob is None:
        return None
    try:
        import math  # noqa: PLC0415

        return round(max(0.0, min(1.0, math.exp(float(logprob)))), 3)
    except (TypeError, ValueError, OverflowError):
        return None
