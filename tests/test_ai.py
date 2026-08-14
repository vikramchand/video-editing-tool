"""Provider parsing, transcript handling, vision fusion, and reasoning.

No test here starts a model. Providers are mocks or fakes throughout.
"""

from __future__ import annotations

import pytest

from video_understanding.ai.providers.base import extract_json, strip_reasoning
from video_understanding.ai.providers.mock import (
    MockLLMProvider,
    MockTranscriptionProvider,
    MockVisionProvider,
)
from video_understanding.ai.reasoning import (
    ReasoningEngine,
    _parse_edits,
    _parse_highlights,
    format_scenes,
    format_transcript,
    heuristic_analysis,
)
from video_understanding.ai.transcription import normalize_transcript, to_srt
from video_understanding.ai.vision import VisionAnalyzer, aggregate_entities
from video_understanding.core.errors import (
    ProviderError,
    ProviderUnavailableError,
    ResponseParseError,
)
from video_understanding.core.models import (
    Frame,
    SceneAnalysis,
    SceneBoundary,
    Transcript,
    TranscriptSegment,
)


class TestExtractJson:
    """Every shape here is one a local model has actually produced."""

    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bare_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_around_the_json(self):
        assert extract_json('Sure! Here is the JSON:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_qwen_style_think_block(self):
        raw = '<think>Let me consider the frames...</think>\n{"a": 1}'
        assert extract_json(raw) == {"a": 1}

    def test_trailing_comma_is_repaired(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_nested_braces_are_balanced_correctly(self):
        assert extract_json('text {"a": {"b": [1, 2]}} more')["a"]["b"] == [1, 2]

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"a": "a } brace"}')["a"] == "a } brace"

    def test_escaped_quotes(self):
        assert extract_json(r'{"a": "say \"hi\""}')["a"] == 'say "hi"'

    def test_array_at_the_top_level(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_empty_response_raises(self):
        with pytest.raises(ResponseParseError, match="empty"):
            extract_json("")

    def test_unparseable_response_raises(self):
        with pytest.raises(ResponseParseError, match="could not parse"):
            extract_json("I am afraid I cannot do that.")

    def test_unterminated_think_block_raises(self):
        with pytest.raises(ResponseParseError):
            extract_json("<think>still thinking and never finished")


class TestStripReasoning:
    def test_removes_think_block(self):
        assert strip_reasoning("<think>hmm</think>answer") == "answer"

    def test_leaves_normal_text_alone(self):
        assert strip_reasoning("just an answer") == "just an answer"


class TestNormalizeTranscript:
    def test_sorts_segments(self):
        transcript = Transcript(segments=[
            TranscriptSegment(start=5, end=6, text="second"),
            TranscriptSegment(start=1, end=2, text="first"),
        ])
        assert [s.text for s in normalize_transcript(transcript).segments] == ["first", "second"]

    def test_clamps_to_duration(self):
        transcript = Transcript(segments=[TranscriptSegment(start=1, end=99, text="long")])
        assert normalize_transcript(transcript, duration=10.0).segments[0].end == 10.0

    def test_resolves_overlaps(self):
        transcript = Transcript(segments=[
            TranscriptSegment(start=0, end=5, text="a"),
            TranscriptSegment(start=3, end=8, text="b"),
        ])
        segments = normalize_transcript(transcript).segments
        assert segments[1].start >= segments[0].end

    def test_drops_empty_text(self):
        transcript = Transcript(segments=[
            TranscriptSegment(start=0, end=1, text="   "),
            TranscriptSegment(start=1, end=2, text="real"),
        ])
        assert len(normalize_transcript(transcript).segments) == 1

    def test_rebuilds_full_text(self):
        transcript = Transcript(segments=[
            TranscriptSegment(start=0, end=1, text="hello"),
            TranscriptSegment(start=1, end=2, text="world"),
        ])
        assert normalize_transcript(transcript).text == "hello world"


class TestSrt:
    def test_format(self):
        srt = to_srt([TranscriptSegment(start=1.5, end=3.25, text="hello")])
        assert srt.startswith("1\n00:00:01,500 --> 00:00:03,250\nhello")

    def test_hours(self):
        srt = to_srt([TranscriptSegment(start=3661.0, end=3662.0, text="x")])
        assert "01:01:01,000" in srt

    def test_numbering_is_sequential(self):
        srt = to_srt([
            TranscriptSegment(start=0, end=1, text="a"),
            TranscriptSegment(start=1, end=2, text="b"),
        ])
        assert srt.startswith("1\n") and "\n2\n" in srt

    def test_empty_input(self):
        assert to_srt([]) == ""


class TestMockProviders:
    def test_transcription_is_deterministic(self, tmp_path):
        provider = MockTranscriptionProvider()
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        assert provider.transcribe(str(audio)).segments == provider.transcribe(str(audio)).segments

    def test_vision_output_validates(self):
        analysis = MockVisionProvider().analyze_frames(["f.jpg"], "", scene_id=1, start=0, end=5)
        assert isinstance(analysis, SceneAnalysis)
        assert 0.0 <= analysis.significance <= 1.0

    def test_llm_returns_parseable_json(self):
        raw = MockLLMProvider().generate("Duration: 20.0 s\nScene 1 [0.0s - 20.0s]")
        data = extract_json(raw)
        assert "summary" in data and isinstance(data["highlights"], list)

    def test_all_providers_report_healthy(self):
        for provider in (MockTranscriptionProvider(), MockVisionProvider(), MockLLMProvider()):
            assert provider.health()[0] is True


class _FailingVision:
    """A vision provider that always fails, in a configurable way."""

    name = "failing"
    model = "none"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def analyze_frames(self, frames, context="", **kwargs):
        self.calls += 1
        raise self.error

    def health(self):
        return False, "always fails"


class TestVisionAnalyzer:
    def _inputs(self):
        boundaries = [
            SceneBoundary(id=1, start=0.0, end=5.0),
            SceneBoundary(id=2, start=5.0, end=10.0),
        ]
        frames = [
            Frame(scene_id=1, timestamp=2.0, path="a.jpg"),
            Frame(scene_id=2, timestamp=7.0, path="b.jpg"),
        ]
        transcript = Transcript(segments=[TranscriptSegment(start=1, end=4, text="hello there")])
        return boundaries, frames, transcript

    def test_produces_one_scene_per_boundary(self):
        boundaries, frames, transcript = self._inputs()
        analyzer = VisionAnalyzer(MockVisionProvider())
        scenes, warnings = analyzer.analyze(boundaries, frames, transcript)
        assert len(scenes) == 2
        assert warnings == []

    def test_attaches_the_matching_transcript_text(self):
        boundaries, frames, transcript = self._inputs()
        scenes, _ = VisionAnalyzer(MockVisionProvider()).analyze(boundaries, frames, transcript)
        assert scenes[0].transcript_text == "hello there"
        assert scenes[1].transcript_text == ""

    def test_reports_progress(self):
        boundaries, frames, transcript = self._inputs()
        seen = []
        VisionAnalyzer(MockVisionProvider()).analyze(
            boundaries, frames, transcript, on_progress=lambda d, t: seen.append((d, t))
        )
        assert seen == [(1, 2), (2, 2)]

    def test_scene_failure_degrades_without_killing_the_run(self):
        boundaries, frames, transcript = self._inputs()
        analyzer = VisionAnalyzer(_FailingVision(ProviderError("model confused")))
        scenes, warnings = analyzer.analyze(boundaries, frames, transcript)
        assert len(scenes) == 2
        assert len(warnings) == 2

    def test_unavailable_backend_stops_retrying(self):
        # Each retry against a dead backend costs a full timeout, so the
        # analyzer must give up after the first failure rather than per scene.
        boundaries, frames, transcript = self._inputs()
        provider = _FailingVision(ProviderUnavailableError("ollama is not running"))
        scenes, warnings = VisionAnalyzer(provider).analyze(boundaries, frames, transcript)
        assert provider.calls == 1
        assert len(scenes) == 2
        assert len(warnings) == 1

    def test_strict_mode_propagates(self):
        boundaries, frames, transcript = self._inputs()
        analyzer = VisionAnalyzer(_FailingVision(ProviderError("nope")), strict=True)
        with pytest.raises(ProviderError):
            analyzer.analyze(boundaries, frames, transcript)

    def test_scene_without_frames_is_still_returned(self):
        boundaries = [SceneBoundary(id=1, start=0.0, end=5.0)]
        scenes, _ = VisionAnalyzer(MockVisionProvider()).analyze(boundaries, [], Transcript())
        assert len(scenes) == 1
        assert "no visual analysis" in scenes[0].description


class TestAggregateEntities:
    def test_deduplicates_case_insensitively(self, scenes):
        scenes[0].people = ["One Adult"]
        scenes[1].people = ["one adult"]
        people, _, _ = aggregate_entities(scenes)
        assert len(people) == 1


class TestReasoningParsing:
    def test_highlights_out_of_range_are_dropped(self):
        warnings = []
        highlights = _parse_highlights(
            [{"start": 0, "end": 5, "reason": "ok", "score": 0.9},
             {"start": 500, "end": 900, "reason": "hallucinated", "score": 0.8}],
            duration=30.0, warnings=warnings,
        )
        assert len(highlights) == 1
        assert warnings

    def test_highlight_end_is_clamped_to_duration(self):
        highlights = _parse_highlights(
            [{"start": 10, "end": 999, "reason": "x", "score": 0.5}], 30.0, []
        )
        assert highlights[0].end == 30.0

    def test_highlights_are_sorted_by_score(self):
        highlights = _parse_highlights(
            [{"start": 0, "end": 5, "reason": "a", "score": 0.3},
             {"start": 5, "end": 10, "reason": "b", "score": 0.9}],
            30.0, [],
        )
        assert [h.score for h in highlights] == [0.9, 0.3]

    def test_score_on_a_hundred_scale_is_rescaled(self):
        highlights = _parse_highlights(
            [{"start": 0, "end": 5, "reason": "x", "score": 85}], 30.0, []
        )
        assert highlights[0].score == 0.85

    @pytest.mark.parametrize(
        ("given", "expected"),
        [("delete", "remove"), ("cut", "remove"), ("subtitle", "caption"),
         ("b-roll", "broll"), ("punch_in", "zoom"), ("REMOVE", "remove"), ("Zoom In", "zoom")],
    )
    def test_action_synonyms_are_mapped(self, given, expected):
        edits = _parse_edits([{"start": 0, "end": 5, "action": given, "reason": "x"}], 30.0, [])
        assert edits[0].action == expected

    def test_unknown_action_is_dropped_with_a_warning(self):
        warnings = []
        edits = _parse_edits(
            [{"start": 0, "end": 5, "action": "explode", "reason": "x"}], 30.0, warnings
        )
        assert edits == [] and warnings

    def test_inverted_range_is_dropped(self):
        inverted = [{"start": 10, "end": 5, "action": "remove", "reason": "x"}]
        assert _parse_edits(inverted, 30.0, []) == []

    def test_edits_are_sorted_by_start(self):
        edits = _parse_edits(
            [{"start": 20, "end": 25, "action": "keep", "reason": "b"},
             {"start": 1, "end": 5, "action": "remove", "reason": "a"}],
            30.0, [],
        )
        assert [e.start for e in edits] == [1.0, 20.0]

    def test_non_list_input_is_tolerated(self):
        assert _parse_highlights("not a list", 30.0, []) == []
        assert _parse_edits(None, 30.0, []) == []

    def test_garbage_list_items_are_skipped(self):
        assert _parse_edits(["nonsense", 42], 30.0, []) == []


class TestStructuredRepresentation:
    def test_scene_format_is_machine_readable(self, scenes):
        text = format_scenes(scenes)
        # The mock LLM parses this format back out, so the shape is a contract.
        assert "Scene 1 [0.0s - 10.0s]" in text
        assert "refrigerator" in text

    def test_transcript_includes_timestamps(self, transcript_segments):
        assert "[1.0s - 4.0s]" in format_transcript(Transcript(segments=transcript_segments))

    def test_long_transcripts_are_truncated(self):
        segments = [
            TranscriptSegment(start=i, end=i + 1, text=f"segment number {i} " * 12)
            for i in range(400)
        ]
        text = format_transcript(Transcript(segments=segments), max_chars=1500)
        assert len(text) < 2200
        assert "truncated" in text

    def test_empty_transcript(self):
        assert format_transcript(Transcript()) == ""


class _ScriptedLLM:
    name = "scripted"
    model = "test"

    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def generate(self, prompt, *, system="", json_mode=False):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def health(self):
        return True, "scripted"


class TestReasoningEngine:
    def test_parses_a_well_formed_response(self, metadata, scenes):
        response = """{"title": "Kitchen", "summary": "A person gets a drink.",
            "topics": ["kitchen"],
            "highlights": [{"start": 10, "end": 20, "reason": "main", "score": 0.9}],
            "edit_suggestions": [
                {"start": 0, "end": 2, "action": "remove", "reason": "silence"}]}"""
        result = ReasoningEngine(_ScriptedLLM(response)).analyze(metadata, Transcript(), scenes)

        assert result.title == "Kitchen"
        assert result.summary == "A person gets a drink."
        assert result.highlights[0].score == 0.9
        assert result.edit_suggestions[0].action == "remove"
        assert result.source == "llm"

    def test_provider_failure_falls_back_to_heuristics(self, metadata, scenes):
        engine = ReasoningEngine(_ScriptedLLM(ProviderUnavailableError("ollama down")))
        result = engine.analyze(metadata, Transcript(), scenes)

        assert result.source == "heuristic"
        assert result.summary
        assert result.highlights
        assert any("heuristics" in w for w in result.warnings)

    def test_strict_mode_propagates_provider_failure(self, metadata, scenes):
        engine = ReasoningEngine(_ScriptedLLM(ProviderUnavailableError("down")), strict=True)
        with pytest.raises(ProviderError):
            engine.analyze(metadata, Transcript(), scenes)

    def test_unparseable_response_falls_back(self, metadata, scenes):
        result = ReasoningEngine(_ScriptedLLM("I cannot help with that.")).analyze(
            metadata, Transcript(), scenes
        )
        assert result.source == "heuristic"

    def test_valid_json_with_no_summary_gets_a_heuristic_one(self, metadata, scenes):
        result = ReasoningEngine(_ScriptedLLM('{"summary": "", "highlights": []}')).analyze(
            metadata, Transcript(), scenes
        )
        assert result.summary
        assert result.highlights
        assert len(result.warnings) == 2


class TestHeuristicAnalysis:
    def test_produces_a_usable_result_without_any_model(
        self, metadata, scenes, transcript_segments
    ):
        result = heuristic_analysis(metadata, Transcript(segments=transcript_segments), scenes)
        assert result.summary
        assert result.highlights
        assert result.source == "heuristic"

    def test_ranks_the_most_important_scene_first(self, metadata, scenes):
        result = heuristic_analysis(metadata, Transcript(), scenes)
        assert result.highlights[0].start == 10.0  # the 0.85-importance scene

    def test_flags_leading_silence_for_removal(self, metadata, scenes):
        transcript = Transcript(
            segments=[TranscriptSegment(start=6.0, end=10.0, text="late start")]
        )
        removals = [e for e in heuristic_analysis(metadata, transcript, scenes).edit_suggestions
                    if e.action == "remove"]
        assert any(e.start == 0.0 and e.end == 6.0 for e in removals)

    def test_flags_internal_silence_gaps(self, metadata, scenes):
        transcript = Transcript(segments=[
            TranscriptSegment(start=0.0, end=2.0, text="one"),
            TranscriptSegment(start=9.0, end=11.0, text="two"),
        ])
        removals = [e for e in heuristic_analysis(metadata, transcript, scenes).edit_suggestions
                    if e.action == "remove"]
        assert any(e.start == 2.0 and e.end == 9.0 for e in removals)

    def test_no_transcript_produces_no_silence_removals(self, metadata, scenes):
        result = heuristic_analysis(metadata, Transcript(), scenes)
        assert not any(e.action == "remove" for e in result.edit_suggestions)

    def test_all_suggestions_stay_within_the_video(self, metadata, scenes, transcript_segments):
        result = heuristic_analysis(metadata, Transcript(segments=transcript_segments), scenes)
        for edit in result.edit_suggestions:
            assert 0 <= edit.start < edit.end <= metadata.duration
