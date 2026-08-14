"""Metadata extraction, scene detection, frame sampling, and audio extraction."""

from __future__ import annotations

import pytest

from tests.conftest import requires_ffmpeg
from video_understanding.core.errors import ValidationError
from video_understanding.core.models import SceneBoundary
from video_understanding.video import scenes as scenes_module
from video_understanding.video.audio import extract_audio, has_audio_stream
from video_understanding.video.frames import (
    extract_frames,
    frames_by_scene,
    plan_frame_budget,
    sample_timestamps,
)
from video_understanding.video.metadata import (
    _parse_fraction,
    extract_metadata,
    validate_upload,
)


class TestParseFraction:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("30/1", 30.0), ("30000/1001", 29.97), ("25", 25.0), ("0/0", 0.0), (None, 0.0), ("", 0.0)],
    )
    def test_parse(self, value, expected):
        assert _parse_fraction(value) == pytest.approx(expected, abs=0.01)


@requires_ffmpeg
class TestMetadata:
    def test_extracts_core_fields(self, sample_video):
        meta = extract_metadata(sample_video)
        assert meta.duration == pytest.approx(9.0, abs=0.5)
        assert (meta.width, meta.height) == (320, 240)
        assert meta.fps == pytest.approx(25.0, abs=0.5)
        assert meta.has_audio is True
        assert meta.video_codec == "h264"
        assert meta.size_bytes > 0

    def test_detects_missing_audio(self, silent_video):
        assert extract_metadata(silent_video).has_audio is False

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="file not found"):
            extract_metadata(tmp_path / "nope.mp4")


@requires_ffmpeg
class TestValidateUpload:
    def test_accepts_a_valid_video(self, sample_video, settings):
        assert validate_upload(sample_video, settings).duration > 0

    def test_rejects_unsupported_extension(self, tmp_path, settings, sample_video):
        bad = tmp_path / "video.txt"
        bad.write_bytes(sample_video.read_bytes())
        with pytest.raises(ValidationError, match="unsupported file type"):
            validate_upload(bad, settings)

    def test_rejects_empty_file(self, tmp_path, settings):
        empty = tmp_path / "empty.mp4"
        empty.touch()
        with pytest.raises(ValidationError, match="empty"):
            validate_upload(empty, settings)

    def test_rejects_video_over_the_duration_limit(self, sample_video, settings):
        settings.video_max_duration_seconds = 2.0
        with pytest.raises(ValidationError, match="exceeds"):
            validate_upload(sample_video, settings)

    def test_rejects_file_over_the_size_limit(self, sample_video, settings):
        settings.video_max_upload_mb = 0.0001
        with pytest.raises(ValidationError, match="MB limit"):
            validate_upload(sample_video, settings)

    def test_rejects_non_video_content(self, tmp_path, settings):
        fake = tmp_path / "fake.mp4"
        fake.write_text("this is definitely not a video" * 100)
        with pytest.raises(ValidationError):
            validate_upload(fake, settings)


@requires_ffmpeg
class TestAudio:
    def test_extracts_audio(self, sample_video, tmp_path):
        out = extract_audio(sample_video, tmp_path / "audio.wav")
        assert out is not None and out.stat().st_size > 0

    def test_silent_video_returns_none(self, silent_video, tmp_path):
        # A silent video is a valid input, not an error.
        assert extract_audio(silent_video, tmp_path / "audio.wav") is None

    def test_has_audio_stream(self, sample_video, silent_video):
        assert has_audio_stream(sample_video) is True
        assert has_audio_stream(silent_video) is False


class TestBuildScenes:
    """Post-processing is detector-independent, so it is tested independently."""

    def test_no_cuts_yields_one_scene(self):
        result = scenes_module.build_scenes([], 20.0, 1.5, "test")
        assert len(result) == 1
        assert (result[0].start, result[0].end) == (0.0, 20.0)

    def test_cuts_become_contiguous_scenes(self):
        result = scenes_module.build_scenes([5.0, 12.0], 20.0, 1.5, "test")
        assert [(s.start, s.end) for s in result] == [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)]

    def test_scenes_are_numbered_from_one(self):
        assert [s.id for s in scenes_module.build_scenes([5.0], 10.0, 1.0, "t")] == [1, 2]

    def test_short_scenes_are_merged(self):
        # Cuts at 5.0 and 5.5 would make a 0.5s scene; it merges into its neighbour.
        result = scenes_module.build_scenes([5.0, 5.5], 20.0, 2.0, "test")
        assert all(scene.duration >= 2.0 for scene in result)

    def test_out_of_range_cuts_are_ignored(self):
        result = scenes_module.build_scenes([-5.0, 5.0, 999.0], 20.0, 1.0, "test")
        assert [(s.start, s.end) for s in result] == [(0.0, 5.0), (5.0, 20.0)]

    def test_duplicate_cuts_are_collapsed(self):
        assert len(scenes_module.build_scenes([5.0, 5.0, 5.0], 20.0, 1.0, "t")) == 2

    def test_scenes_are_contiguous_and_cover_the_video(self):
        result = scenes_module.build_scenes([3.0, 7.0, 11.0], 15.0, 1.0, "test")
        assert result[0].start == 0.0
        assert result[-1].end == 15.0
        for previous, current in zip(result, result[1:], strict=False):
            assert previous.end == current.start

    def test_zero_duration_yields_nothing(self):
        assert scenes_module.build_scenes([], 0.0, 1.0, "test") == []

    def test_all_short_scenes_collapse_to_one(self):
        result = scenes_module.build_scenes([1.0, 2.0, 3.0], 4.0, 10.0, "test")
        assert len(result) == 1
        assert (result[0].start, result[0].end) == (0.0, 4.0)


class TestUniformDetector:
    def test_chunks_the_timeline(self, settings):
        settings.scene_uniform_duration = 5.0
        assert scenes_module._detect_uniform(22.0, settings) == [5.0, 10.0, 15.0, 20.0]

    def test_short_video_gets_no_cuts(self, settings):
        settings.scene_uniform_duration = 6.0
        assert scenes_module._detect_uniform(4.0, settings) == []


@requires_ffmpeg
class TestSceneDetection:
    def test_detects_the_three_scenes(self, sample_video, settings):
        # Uses the production default thresholds; the fixture is textured so
        # its cuts produce realistic content deltas.
        result = scenes_module.detect_scenes(sample_video, 9.0, settings)
        assert len(result) == 3
        assert result[0].start == 0.0
        assert result[-1].end == pytest.approx(9.0, abs=0.5)
        assert [round(s.start) for s in result] == [0, 3, 6]

    def test_explicit_uniform_detector(self, sample_video, settings):
        settings.scene_detector = "uniform"
        settings.scene_uniform_duration = 3.0
        result = scenes_module.detect_scenes(sample_video, 9.0, settings)
        assert [s.detector for s in result] == ["uniform"] * len(result)

    def test_ffmpeg_detector(self, sample_video, settings):
        settings.scene_detector = "ffmpeg"
        result = scenes_module.detect_scenes(sample_video, 9.0, settings)
        assert len(result) >= 1

    def test_falls_back_when_the_chosen_detector_fails(self, sample_video, settings, monkeypatch):
        settings.scene_detector = "pyscenedetect"
        monkeypatch.setattr(
            scenes_module,
            "_detect_pyscenedetect",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = scenes_module.detect_scenes(sample_video, 9.0, settings)
        assert result and result[0].detector == "uniform"


class TestFrameBudget:
    def _boundaries(self, count: int, duration: float = 10.0) -> list[SceneBoundary]:
        return [
            SceneBoundary(id=i, start=i * duration, end=(i + 1) * duration)
            for i in range(1, count + 1)
        ]

    def test_under_budget_gets_the_full_allocation(self):
        budget = plan_frame_budget(self._boundaries(5), frames_per_scene=3, max_frames=100)
        assert sum(budget.values()) == 15
        assert set(budget.values()) == {3}

    def test_over_budget_is_capped(self):
        budget = plan_frame_budget(self._boundaries(50), frames_per_scene=4, max_frames=60)
        assert sum(budget.values()) <= 60

    def test_every_scene_keeps_at_least_one_frame_when_possible(self):
        budget = plan_frame_budget(self._boundaries(30), frames_per_scene=4, max_frames=40)
        assert len(budget) == 30
        assert all(count >= 1 for count in budget.values())

    def test_more_scenes_than_frames_keeps_the_longest(self):
        boundaries = [
            SceneBoundary(id=1, start=0, end=1),
            SceneBoundary(id=2, start=1, end=50),   # longest
            SceneBoundary(id=3, start=50, end=52),
        ]
        budget = plan_frame_budget(boundaries, frames_per_scene=3, max_frames=2)
        assert sum(budget.values()) == 2
        assert 2 in budget  # the long scene survives
        assert 1 not in budget

    def test_longer_scenes_get_more_frames(self):
        boundaries = [
            SceneBoundary(id=1, start=0, end=2),
            SceneBoundary(id=2, start=2, end=60),
        ]
        budget = plan_frame_budget(boundaries, frames_per_scene=5, max_frames=6)
        assert budget[2] > budget[1]

    def test_no_scenes_yields_no_budget(self):
        assert plan_frame_budget([], 3, 100) == {}

    def test_scenario_from_the_readme(self):
        # 10 scenes, 3 frames each, cap 100 -> 30 frames.
        budget = plan_frame_budget(self._boundaries(10, 6.0), frames_per_scene=3, max_frames=100)
        assert sum(budget.values()) == 30


class TestSampleTimestamps:
    def test_timestamps_are_inside_the_scene(self):
        scene = SceneBoundary(id=1, start=10.0, end=20.0)
        for timestamp in sample_timestamps(scene, 3):
            assert 10.0 < timestamp < 20.0

    def test_timestamps_avoid_the_boundaries(self):
        scene = SceneBoundary(id=1, start=0.0, end=10.0)
        timestamps = sample_timestamps(scene, 2)
        assert timestamps == [2.5, 7.5]

    def test_single_frame_is_centred(self):
        assert sample_timestamps(SceneBoundary(id=1, start=0.0, end=10.0), 1) == [5.0]

    def test_zero_count(self):
        assert sample_timestamps(SceneBoundary(id=1, start=0.0, end=10.0), 0) == []

    def test_timestamps_are_ordered(self):
        timestamps = sample_timestamps(SceneBoundary(id=1, start=5.0, end=25.0), 6)
        assert timestamps == sorted(timestamps)


@requires_ffmpeg
class TestExtractFrames:
    def test_extracts_files_that_exist(self, sample_video, settings, tmp_path):
        boundaries = [
            SceneBoundary(id=1, start=0.0, end=3.0),
            SceneBoundary(id=2, start=3.0, end=6.0),
        ]
        settings.video_frames_per_scene = 2
        frames = extract_frames(sample_video, boundaries, tmp_path / "frames", settings)

        assert len(frames) == 4
        for frame in frames:
            from pathlib import Path

            assert Path(frame.path).stat().st_size > 0

    def test_respects_the_max_frames_cap(self, sample_video, settings, tmp_path):
        boundaries = [SceneBoundary(id=i, start=i - 1, end=i) for i in range(1, 8)]
        settings.video_frames_per_scene = 3
        settings.video_max_frames = 5
        frames = extract_frames(sample_video, boundaries, tmp_path / "frames", settings)
        assert len(frames) <= 5

    def test_frames_are_downscaled(self, sample_video, settings, tmp_path):
        settings.video_frame_width = 160
        settings.video_frames_per_scene = 1
        frames = extract_frames(
            sample_video, [SceneBoundary(id=1, start=0.0, end=3.0)], tmp_path / "f", settings
        )
        assert extract_metadata(frames[0].path).width == 160


class TestFramesByScene:
    def test_groups_and_orders(self):
        from video_understanding.core.models import Frame

        frames = [
            Frame(scene_id=2, timestamp=5.0, path="b"),
            Frame(scene_id=1, timestamp=3.0, path="c"),
            Frame(scene_id=1, timestamp=1.0, path="a"),
        ]
        grouped = frames_by_scene(frames)
        assert set(grouped) == {1, 2}
        assert [f.timestamp for f in grouped[1]] == [1.0, 3.0]
