"""Exception hierarchy.

Everything the pipeline raises deliberately derives from `VideoUnderstandingError`
so the job processor can distinguish an expected, reportable failure (bad file,
model unreachable) from a genuine bug, which it lets propagate as-is.
"""

from __future__ import annotations


class VideoUnderstandingError(Exception):
    """Base class for all expected, reportable errors."""


class ValidationError(VideoUnderstandingError):
    """The uploaded file is missing, too large, or not a usable video."""


class FFmpegError(VideoUnderstandingError):
    """An ffmpeg/ffprobe invocation failed or is unavailable."""

    def __init__(self, message: str, *, command: list[str] | None = None, stderr: str = "") -> None:
        self.command = command or []
        self.stderr = stderr
        super().__init__(message)


class ProviderError(VideoUnderstandingError):
    """An AI provider was unreachable or returned something unusable."""

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class ProviderUnavailableError(ProviderError):
    """The provider's backend is not running or the model is not installed."""


class ResponseParseError(ProviderError):
    """A model replied, but not with the JSON the prompt required."""


class JobNotFoundError(VideoUnderstandingError):
    """No job exists with the requested id."""


class RenderError(VideoUnderstandingError):
    """The edit could not be rendered."""
