"""Configuration.

Precedence, lowest to highest:

    built-in defaults  <  config.yaml  <  environment variables / .env

No model name is ever hard-coded outside this module. Every stage receives the
provider and model it should use from `Settings`, so swapping a local Ollama
model for a hosted GPU endpoint is a configuration change, not a code change.
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_FILE = "config.yaml"


def _load_yaml_defaults() -> dict[str, Any]:
    """Flatten an optional YAML config into env-style keys.

    The README documents a nested YAML shape:

        video:
          max_duration_seconds: 180
          frames_per_scene: 3

    which becomes VIDEO_MAX_DURATION_SECONDS / VIDEO_FRAMES_PER_SCENE. Any key
    already present in the real environment wins, so YAML only supplies
    defaults.
    """
    path = Path(os.getenv("CONFIG_FILE", DEFAULT_CONFIG_FILE))
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}

    flat: dict[str, Any] = {}
    for section, value in raw.items():
        if isinstance(value, dict):
            for key, inner in value.items():
                flat[f"{section}_{key}".upper()] = inner
        else:
            flat[str(section).upper()] = value
    return flat


class Settings(BaseSettings):
    """Application settings, driven entirely by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # --- Server -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "*"

    # --- Storage ------------------------------------------------------------
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    database_path: Path | None = None
    keep_intermediate_files: bool = Field(
        default=False,
        description="Keep extracted audio and frames after processing (useful for debugging)",
    )

    # --- Job queue ----------------------------------------------------------
    worker_count: int = Field(default=1, ge=1, le=8)
    job_timeout_seconds: int = 3600

    # --- Video limits and sampling -----------------------------------------
    video_max_duration_seconds: float = Field(default=180.0, gt=0)
    video_max_upload_mb: float = Field(default=512.0, gt=0)
    video_frames_per_scene: int = Field(default=3, ge=1, le=10)
    video_max_frames: int = Field(default=100, ge=1, le=1000)
    video_frame_width: int = Field(
        default=768, description="Frames are downscaled to this width before the vision model"
    )
    video_allowed_extensions: str = ".mp4,.mov,.m4v,.avi,.mkv,.webm"

    # --- Scene detection ----------------------------------------------------
    scene_detector: Literal["auto", "pyscenedetect", "ffmpeg", "uniform"] = "auto"
    scene_threshold: float = Field(
        default=27.0, gt=0, description="PySceneDetect content threshold"
    )
    scene_ffmpeg_threshold: float = Field(
        default=0.25,
        gt=0,
        lt=1,
        description=(
            "FFmpeg `scene` filter score threshold. Lower catches more cuts. "
            "Real cuts typically score 0.2-0.6; short scenes are merged afterwards, "
            "so erring low costs little."
        ),
    )
    scene_min_duration: float = Field(
        default=1.5, gt=0, description="Merge scenes shorter than this"
    )
    scene_uniform_duration: float = Field(
        default=6.0, gt=0, description="Chunk length for the uniform fallback detector"
    )

    # --- Providers ----------------------------------------------------------
    transcription_provider: Literal["whisper", "mock", "null"] = "whisper"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str | None = None

    vision_provider: Literal["ollama", "mock", "null"] = "ollama"
    vision_model: str = "qwen2.5vl:7b"

    llm_provider: Literal["ollama", "mock", "null"] = "ollama"
    llm_model: str = "qwen3:8b"

    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = Field(default=300.0, gt=0)
    ollama_num_ctx: int = Field(default=8192, ge=512)
    ollama_temperature: float = Field(default=0.2, ge=0, le=2)

    # --- Resilience ---------------------------------------------------------
    fail_on_provider_error: bool = Field(
        default=False,
        description=(
            "When False a failing AI stage degrades gracefully (recorded in "
            "`warnings`) instead of failing the whole job."
        ),
    )

    # --- Derived ------------------------------------------------------------

    @field_validator("data_dir", "output_dir", mode="before")
    @classmethod
    def _expand(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def db_path(self) -> Path:
        return self.database_path or (self.data_dir / "videos.db")

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in self.video_allowed_extensions.split(",")
            if e.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return int(self.video_max_upload_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.work_dir,
            self.renders_dir,
            self.thumbnails_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def provider_summary(self) -> dict[str, str]:
        return {
            "transcription": f"{self.transcription_provider}:{self.whisper_model}",
            "vision": f"{self.vision_provider}:{self.vision_model}",
            "llm": f"{self.llm_provider}:{self.llm_model}",
        }


class HardwareInfo:
    """Best-effort host detection, used for defaults and for `/health`."""

    def __init__(self) -> None:
        self.system = platform.system()
        self.machine = platform.machine()

    @property
    def is_mac(self) -> bool:
        return self.system == "Darwin"

    @property
    def is_apple_silicon(self) -> bool:
        return self.is_mac and self.machine in ("arm64", "aarch64")

    @property
    def is_intel_mac(self) -> bool:
        return self.is_mac and self.machine in ("x86_64", "AMD64")

    @property
    def in_container(self) -> bool:
        return Path("/.dockerenv").exists() or os.getenv("IN_DOCKER") == "1"

    def describe(self) -> str:
        if self.is_apple_silicon:
            return f"Apple Silicon ({self.machine})"
        if self.is_intel_mac:
            return "Intel Mac"
        return f"{self.system} ({self.machine})"

    def whisper_defaults(self) -> tuple[str, str]:
        """Return (device, compute_type) appropriate for this host.

        faster-whisper has no Metal backend, so Apple Silicon runs on CPU with
        int8 quantization — which is still comfortably faster than realtime for
        the `small` model.
        """
        if self.is_apple_silicon:
            return "cpu", "int8"
        return "cpu", "int8"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    yaml_defaults = _load_yaml_defaults()
    # Environment always wins over the YAML file.
    overrides = {k.lower(): v for k, v in yaml_defaults.items() if k not in os.environ}
    settings = Settings(**overrides)
    settings.ensure_directories()
    return settings


@functools.lru_cache(maxsize=1)
def get_hardware() -> HardwareInfo:
    return HardwareInfo()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def reset_settings_cache() -> None:
    """Test helper: drop the cached settings so env changes take effect."""
    get_settings.cache_clear()
