"""Configuration loading and precedence."""

from __future__ import annotations

import os

import pytest

from video_understanding.core.config import (
    HardwareInfo,
    Settings,
    _load_yaml_defaults,
    get_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestDefaults:
    def test_sampling_defaults_match_the_documented_example(self):
        settings = Settings()
        assert settings.video_max_duration_seconds == 180
        assert settings.video_frames_per_scene == 3
        assert settings.video_max_frames == 100

    def test_no_model_name_is_empty(self):
        settings = Settings()
        assert settings.whisper_model and settings.vision_model and settings.llm_model

    def test_worker_count_defaults_to_one(self):
        # Local inference is memory-bound; parallel jobs make things worse.
        assert Settings().worker_count == 1


class TestDerivedPaths:
    def test_subdirectories_live_under_data_dir(self, tmp_path):
        settings = Settings(data_dir=tmp_path / "d")
        assert settings.uploads_dir.parent == settings.data_dir
        assert settings.db_path == settings.data_dir / "videos.db"

    def test_explicit_database_path_wins(self, tmp_path):
        settings = Settings(data_dir=tmp_path, database_path=tmp_path / "custom.db")
        assert settings.db_path.name == "custom.db"

    def test_ensure_directories_creates_them(self, tmp_path):
        settings = Settings(data_dir=tmp_path / "new", output_dir=tmp_path / "out")
        settings.ensure_directories()
        assert settings.uploads_dir.is_dir()
        assert settings.renders_dir.is_dir()
        assert settings.output_dir.is_dir()


class TestParsedLists:
    def test_extensions_are_normalised(self):
        settings = Settings(video_allowed_extensions="mp4, .MOV ,webm")
        assert settings.allowed_extensions == {".mp4", ".mov", ".webm"}

    def test_max_upload_bytes(self):
        assert Settings(video_max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024

    def test_cors_origins_are_split(self):
        assert Settings(cors_origins="http://a.com, http://b.com").cors_origin_list == [
            "http://a.com",
            "http://b.com",
        ]


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("video_frames_per_scene", 0),
            ("video_max_frames", 0),
            ("video_max_duration_seconds", -1),
            ("worker_count", 0),
            ("scene_ffmpeg_threshold", 1.5),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field, value):
        with pytest.raises(ValueError):
            Settings(**{field: value})

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError):
            Settings(llm_provider="openai")


class TestYamlLoading:
    def test_nested_keys_are_flattened(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text(
            "video:\n  frames_per_scene: 7\n  max_frames: 42\nscene:\n  detector: uniform\n"
        )
        monkeypatch.setenv("CONFIG_FILE", str(config))

        flat = _load_yaml_defaults()
        assert flat["VIDEO_FRAMES_PER_SCENE"] == 7
        assert flat["SCENE_DETECTOR"] == "uniform"

    def test_yaml_values_reach_settings(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("video:\n  frames_per_scene: 7\nllm_model: from-yaml\n")
        monkeypatch.setenv("CONFIG_FILE", str(config))
        monkeypatch.delenv("VIDEO_FRAMES_PER_SCENE", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)

        settings = get_settings()
        assert settings.video_frames_per_scene == 7
        assert settings.llm_model == "from-yaml"

    def test_environment_beats_yaml(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("video:\n  frames_per_scene: 7\n")
        monkeypatch.setenv("CONFIG_FILE", str(config))
        monkeypatch.setenv("VIDEO_FRAMES_PER_SCENE", "2")
        monkeypatch.chdir(tmp_path)

        assert get_settings().video_frames_per_scene == 2

    def test_missing_file_is_not_an_error(self, monkeypatch):
        monkeypatch.setenv("CONFIG_FILE", "/nonexistent/config.yaml")
        assert _load_yaml_defaults() == {}

    def test_malformed_yaml_is_ignored_rather_than_fatal(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("video:\n  - this: [is not\n    valid yaml")
        monkeypatch.setenv("CONFIG_FILE", str(config))
        assert _load_yaml_defaults() == {}

    def test_shipped_example_is_loadable(self, monkeypatch):
        example = os.path.join(os.path.dirname(__file__), "..", "config.yaml.example")
        monkeypatch.setenv("CONFIG_FILE", example)
        flat = _load_yaml_defaults()
        assert flat["VIDEO_FRAMES_PER_SCENE"] == 3
        # Every key the example produces must be a real setting.
        for key in flat:
            assert key.lower() in Settings.model_fields, f"unknown setting in example: {key}"


class TestProviderSummary:
    def test_includes_every_layer(self):
        summary = Settings().provider_summary()
        assert set(summary) == {"transcription", "vision", "llm"}


class TestHardwareInfo:
    def test_describe_is_non_empty(self):
        assert HardwareInfo().describe()

    def test_whisper_defaults_are_returned(self):
        device, compute = HardwareInfo().whisper_defaults()
        assert device in ("cpu", "cuda")
        assert compute

    def test_apple_silicon_detection(self, monkeypatch):
        info = HardwareInfo()
        monkeypatch.setattr(info, "system", "Darwin")
        monkeypatch.setattr(info, "machine", "arm64")
        assert info.is_apple_silicon is True
        assert info.is_intel_mac is False
        assert "Apple Silicon" in info.describe()

    def test_intel_mac_detection(self, monkeypatch):
        info = HardwareInfo()
        monkeypatch.setattr(info, "system", "Darwin")
        monkeypatch.setattr(info, "machine", "x86_64")
        assert info.is_intel_mac is True
        assert info.is_apple_silicon is False
