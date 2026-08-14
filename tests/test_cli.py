"""CLI argument handling and report rendering."""

from __future__ import annotations

import json

import pytest

from tests.conftest import requires_ffmpeg
from video_understanding.cli.main import (
    _wrap,
    build_parser,
    main,
    normalize_argv,
    print_report,
)


class TestNormalizeArgv:
    def test_bare_video_becomes_a_process_command(self):
        # `video-understand my_video.mp4` is the documented entry point.
        assert normalize_argv(["my_video.mp4"]) == ["process", "my_video.mp4"]

    def test_flags_after_a_bare_video_are_preserved(self):
        assert normalize_argv(["v.mp4", "--render"]) == ["process", "v.mp4", "--render"]

    @pytest.mark.parametrize("command", ["check", "serve", "process"])
    def test_explicit_subcommands_are_left_alone(self, command):
        assert normalize_argv([command]) == [command]

    @pytest.mark.parametrize("flag", ["-h", "--help", "--version"])
    def test_global_flags_are_left_alone(self, flag):
        assert normalize_argv([flag]) == [flag]

    def test_empty_argv(self):
        assert normalize_argv([]) == []


class TestParser:
    def test_bare_form_parses(self):
        args = build_parser().parse_args(normalize_argv(["clip.mp4"]))
        assert args.video == "clip.mp4"

    def test_render_flags(self):
        args = build_parser().parse_args(
            normalize_argv(["clip.mp4", "--render", "--captions", "--highlights-only"])
        )
        assert args.render and args.captions and args.highlights_only

    def test_output_option(self):
        args = build_parser().parse_args(normalize_argv(["clip.mp4", "-o", "out.json"]))
        assert args.output == "out.json"

    def test_serve_options(self):
        args = build_parser().parse_args(["serve", "--port", "9000"])
        assert args.port == 9000


class TestMainErrors:
    def test_no_arguments_prints_help(self, capsys):
        assert main([]) == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_missing_file_is_an_error(self, capsys, tmp_path):
        assert main([str(tmp_path / "nope.mp4")]) == 2
        assert "not found" in capsys.readouterr().err.lower()


class TestReport:
    def test_prints_every_section(self, understanding, capsys, tmp_path):
        print_report(understanding, tmp_path / "video.mp4")
        out = capsys.readouterr().out

        for heading in ("SUMMARY", "SCENES", "HIGHLIGHTS", "EDIT SUGGESTIONS", "TRANSCRIPT"):
            assert heading in out
        assert understanding.summary in out.replace("\n  ", " ")

    def test_handles_an_empty_result(self, capsys, tmp_path):
        from video_understanding.core.models import VideoUnderstanding

        print_report(VideoUnderstanding(duration=0.0), tmp_path / "empty.mp4")
        assert "no speech detected" in capsys.readouterr().out.lower()

    def test_shows_warnings(self, understanding, capsys, tmp_path):
        understanding.warnings = ["Ollama was unreachable"]
        print_report(understanding, tmp_path / "v.mp4")
        assert "Ollama was unreachable" in capsys.readouterr().out


class TestWrap:
    def test_wraps_at_the_requested_width(self):
        lines = _wrap("word " * 40, 30)
        assert all(len(line) <= 30 for line in lines)

    def test_empty_text(self):
        assert _wrap("", 20) == [""]


@requires_ffmpeg
class TestEndToEnd:
    def test_writes_json_and_prints_a_report(self, sample_video, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "mock")
        monkeypatch.setenv("VISION_PROVIDER", "mock")
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

        from video_understanding.core.config import reset_settings_cache

        reset_settings_cache()

        output = tmp_path / "result.json"
        assert main([str(sample_video), "-o", str(output)]) == 0

        data = json.loads(output.read_text())
        assert data["duration"] > 0
        assert data["scenes"] and data["summary"]
        assert "SCENES" in capsys.readouterr().out

        reset_settings_cache()

    def test_quiet_mode_only_reports_the_saved_path(
        self, sample_video, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "mock")
        monkeypatch.setenv("VISION_PROVIDER", "mock")
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

        from video_understanding.core.config import reset_settings_cache

        reset_settings_cache()

        output = tmp_path / "quiet.json"
        assert main([str(sample_video), "-o", str(output), "--quiet"]) == 0

        out = capsys.readouterr().out
        assert "SCENES" not in out
        assert str(output) in out

        reset_settings_cache()
