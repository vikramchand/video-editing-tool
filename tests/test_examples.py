"""The shipped example JSON must stay valid as the models evolve.

These files are documentation, and documentation that no longer matches the
schema is worse than none. A schema change that breaks them fails the build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_understanding.core.models import VideoUnderstanding

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("name", ["sample_output.json", "mock_output.json"])
class TestShippedExamples:
    def _load(self, name: str) -> VideoUnderstanding:
        path = EXAMPLES / name
        if not path.is_file():
            pytest.skip(f"{name} is not present")
        return VideoUnderstanding.model_validate(json.loads(path.read_text()))

    def test_validates_against_the_schema(self, name):
        assert self._load(name).duration > 0

    def test_documented_keys_are_present(self, name):
        raw = json.loads((EXAMPLES / name).read_text())
        for key in (
            "duration", "summary", "transcript", "scenes", "highlights", "edit_suggestions",
        ):
            assert key in raw

    def test_all_time_ranges_are_within_the_video(self, name):
        result = self._load(name)
        limit = result.duration + 0.01
        for scene in result.scenes:
            assert 0 <= scene.start < scene.end <= limit
        for item in [*result.highlights, *result.edit_suggestions]:
            assert 0 <= item.start < item.end <= limit

    def test_scenes_are_contiguous(self, name):
        scenes = self._load(name).scenes
        for previous, current in zip(scenes, scenes[1:], strict=False):
            assert previous.end == current.start

    def test_transcript_is_ordered(self, name):
        segments = self._load(name).transcript
        for previous, current in zip(segments, segments[1:], strict=False):
            assert previous.start <= current.start
