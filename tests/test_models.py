"""Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from video_understanding.core.models import (
    EditSuggestion,
    Highlight,
    ProcessingStage,
    SceneAnalysis,
    SceneBoundary,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
    VideoUnderstanding,
)


class TestTranscriptSegment:
    def test_valid_segment(self):
        segment = TranscriptSegment(start=1.0, end=2.5, text="  hello  ")
        assert segment.text == "hello"
        assert segment.duration == 1.5

    def test_end_before_start_is_rejected(self):
        with pytest.raises(PydanticValidationError, match="precedes start"):
            TranscriptSegment(start=5.0, end=2.0, text="backwards")

    def test_negative_start_is_rejected(self):
        with pytest.raises(PydanticValidationError):
            TranscriptSegment(start=-1.0, end=2.0, text="negative")

    def test_zero_length_segment_is_allowed(self):
        # Whisper legitimately emits these for very short interjections.
        assert TranscriptSegment(start=2.0, end=2.0, text="oh").duration == 0.0


class TestTranscript:
    def test_text_is_derived_from_segments(self, transcript_segments):
        assert Transcript(segments=transcript_segments).text.startswith("Hello and welcome.")

    def test_between_returns_overlapping_segments(self, transcript_segments):
        overlapping = Transcript(segments=transcript_segments).between(3.0, 6.0)
        assert len(overlapping) == 2

    def test_between_excludes_touching_boundaries(self, transcript_segments):
        # A segment ending exactly at the window start does not overlap it.
        assert Transcript(segments=transcript_segments).between(4.0, 4.4) == []

    def test_text_between(self, transcript_segments):
        assert "kitchen" in Transcript(segments=transcript_segments).text_between(4.0, 9.0)


class TestSceneAnalysis:
    """These coercions exist because small VLMs really do return these shapes."""

    def test_string_is_coerced_to_list(self):
        analysis = SceneAnalysis(people="one adult, one child", objects="table")
        assert analysis.people == ["one adult", "one child"]
        assert analysis.objects == ["table"]

    def test_none_and_placeholder_values_are_dropped(self):
        assert SceneAnalysis(people=None).people == []
        assert SceneAnalysis(objects="none").objects == []
        assert SceneAnalysis(actions="N/A").actions == []

    def test_significance_on_a_hundred_scale_is_rescaled(self):
        assert SceneAnalysis(significance=85).significance == 0.85

    def test_significance_on_a_ten_scale_is_rescaled(self):
        assert SceneAnalysis(significance=8).significance == 0.8

    def test_significance_garbage_falls_back_to_default(self):
        assert SceneAnalysis(significance="very high").significance == 0.5

    def test_significance_is_clamped(self):
        assert SceneAnalysis(significance=-3).significance == 0.0

    def test_numeric_list_items_become_strings(self):
        assert SceneAnalysis(objects=[1, 2]).objects == ["1", "2"]


class TestSceneBoundary:
    def test_zero_length_boundary_is_rejected(self):
        with pytest.raises(PydanticValidationError, match="must exceed start"):
            SceneBoundary(id=1, start=5.0, end=5.0)

    def test_duration(self):
        assert SceneBoundary(id=1, start=2.0, end=7.5).duration == 5.5


class TestEditSuggestion:
    def test_unknown_action_is_rejected(self):
        # The renderer switches on `action`; an unknown value must not get through.
        with pytest.raises(PydanticValidationError):
            EditSuggestion(start=0, end=1, action="explode", reason="no")

    @pytest.mark.parametrize(
        "action", ["keep", "remove", "trim", "zoom", "caption", "broll"]
    )
    def test_all_valid_actions_are_accepted(self, action):
        assert EditSuggestion(start=0, end=1, action=action, reason="ok").action == action


class TestHighlight:
    def test_score_above_one_is_rejected(self):
        with pytest.raises(PydanticValidationError):
            Highlight(start=0, end=1, reason="x", score=1.5)


class TestVideoMetadata:
    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (1080, 1920, "portrait"), (1920, 1080, "landscape"),
            (500, 500, "square"), (0, 0, "unknown"),
        ],
    )
    def test_orientation(self, width, height, expected):
        assert VideoMetadata(duration=1, width=width, height=height, fps=30).orientation == expected


class TestProcessingStage:
    def test_progress_is_monotonic(self):
        values = [stage.progress for stage in ProcessingStage]
        assert values == sorted(values)

    def test_endpoints(self):
        assert ProcessingStage.PENDING.progress == 0.0
        assert ProcessingStage.DONE.progress == 1.0


class TestVideoUnderstanding:
    def test_round_trips_through_json(self, understanding):
        restored = VideoUnderstanding.model_validate_json(understanding.model_dump_json())
        assert restored.duration == understanding.duration
        assert len(restored.scenes) == len(understanding.scenes)
        assert restored.scenes[1].objects == ["refrigerator", "drink"]

    def test_defaults_are_empty_not_none(self):
        empty = VideoUnderstanding(duration=1.0)
        assert empty.scenes == [] and empty.highlights == [] and empty.warnings == []
