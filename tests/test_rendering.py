"""Edit decision maths and FFmpeg rendering."""

from __future__ import annotations

import pytest

from tests.conftest import requires_ffmpeg
from video_understanding.core.errors import RenderError
from video_understanding.core.models import EditSuggestion, Highlight, TranscriptSegment
from video_understanding.video import rendering
from video_understanding.video.metadata import extract_metadata


class TestMergeRanges:
    def test_merges_overlaps(self):
        assert rendering.merge_ranges([(0, 5), (3, 8)]) == [(0, 8)]

    def test_merges_touching_ranges(self):
        assert rendering.merge_ranges([(0, 5), (5, 10)]) == [(0, 10)]

    def test_keeps_separate_ranges_apart(self):
        assert rendering.merge_ranges([(0, 2), (5, 8)]) == [(0, 2), (5, 8)]

    def test_sorts_input(self):
        assert rendering.merge_ranges([(10, 12), (0, 2)]) == [(0, 2), (10, 12)]

    def test_empty(self):
        assert rendering.merge_ranges([]) == []


class TestSubtractRanges:
    def test_removes_the_middle(self):
        assert rendering.subtract_ranges([(0, 10)], [(3, 6)]) == [(0, 3), (6, 10)]

    def test_removes_the_start(self):
        assert rendering.subtract_ranges([(0, 10)], [(0, 4)]) == [(4, 10)]

    def test_removes_the_end(self):
        assert rendering.subtract_ranges([(0, 10)], [(7, 10)]) == [(0, 7)]

    def test_removing_everything_leaves_nothing(self):
        assert rendering.subtract_ranges([(0, 10)], [(0, 10)]) == []

    def test_non_overlapping_hole_changes_nothing(self):
        assert rendering.subtract_ranges([(0, 5)], [(8, 9)]) == [(0, 5)]

    def test_multiple_holes(self):
        result = rendering.subtract_ranges([(0, 20)], [(2, 4), (10, 12)])
        assert result == [(0, 2), (4, 10), (12, 20)]

    def test_slivers_are_discarded(self):
        # A 0.01s fragment is not worth a segment in the filter graph.
        assert rendering.subtract_ranges([(0, 10)], [(0.01, 10)]) == []


class TestComputeKeepSegments:
    def test_no_removals_keeps_everything(self):
        assert rendering.compute_keep_segments(30.0, []) == [(0.0, 30.0)]

    def test_applies_removals(self):
        edits = [EditSuggestion(start=0, end=5, action="remove", reason="silence")]
        assert rendering.compute_keep_segments(30.0, edits) == [(5.0, 30.0)]

    def test_ignores_non_removal_actions(self):
        edits = [EditSuggestion(start=0, end=5, action="caption", reason="speech")]
        assert rendering.compute_keep_segments(30.0, edits) == [(0.0, 30.0)]

    def test_removals_can_be_disabled(self):
        edits = [EditSuggestion(start=0, end=5, action="remove", reason="silence")]
        assert rendering.compute_keep_segments(30.0, edits, apply_removals=False) == [(0.0, 30.0)]

    def test_removal_past_the_end_is_clamped(self):
        edits = [EditSuggestion(start=25, end=999, action="remove", reason="x")]
        assert rendering.compute_keep_segments(30.0, edits) == [(0.0, 25.0)]

    def test_removing_everything_raises(self):
        edits = [EditSuggestion(start=0, end=30, action="remove", reason="all of it")]
        with pytest.raises(RenderError, match="entire video"):
            rendering.compute_keep_segments(30.0, edits)

    def test_highlights_only_keeps_the_highlights(self):
        highlights = [Highlight(start=10, end=20, reason="main", score=0.9)]
        keep = rendering.compute_keep_segments(
            30.0, [], highlights_only=True, highlights=highlights
        )
        assert keep == [(10.0, 20.0)]

    def test_highlights_only_merges_adjacent_highlights(self):
        highlights = [
            Highlight(start=5, end=10, reason="a", score=0.7),
            Highlight(start=10, end=15, reason="b", score=0.8),
        ]
        keep = rendering.compute_keep_segments(
            30.0, [], highlights_only=True, highlights=highlights
        )
        assert keep == [(5.0, 15.0)]

    def test_highlights_only_without_highlights_falls_back_to_everything(self):
        assert rendering.compute_keep_segments(30.0, [], highlights_only=True, highlights=[]) == [
            (0.0, 30.0)
        ]

    def test_zero_duration(self):
        assert rendering.compute_keep_segments(0.0, []) == []


class TestRemapTranscript:
    def test_unchanged_timeline_keeps_timings(self):
        segments = [TranscriptSegment(start=2, end=4, text="hello")]
        remapped = rendering.remap_transcript(segments, [(0.0, 10.0)])
        assert (remapped[0].start, remapped[0].end) == (2.0, 4.0)

    def test_shifts_after_a_cut(self):
        # Keeping only 5-10s means speech at 6s lands at 1s in the edit.
        segments = [TranscriptSegment(start=6, end=8, text="kept")]
        remapped = rendering.remap_transcript(segments, [(5.0, 10.0)])
        assert (remapped[0].start, remapped[0].end) == (1.0, 3.0)

    def test_drops_speech_inside_a_removed_range(self):
        segments = [TranscriptSegment(start=1, end=2, text="cut away")]
        assert rendering.remap_transcript(segments, [(5.0, 10.0)]) == []

    def test_clips_speech_straddling_a_cut(self):
        segments = [TranscriptSegment(start=4, end=7, text="straddles")]
        remapped = rendering.remap_transcript(segments, [(5.0, 10.0)])
        assert (remapped[0].start, remapped[0].end) == (0.0, 2.0)

    def test_multiple_kept_segments_are_concatenated(self):
        segments = [
            TranscriptSegment(start=1, end=2, text="first"),
            TranscriptSegment(start=11, end=12, text="second"),
        ]
        remapped = rendering.remap_transcript(segments, [(0.0, 5.0), (10.0, 15.0)])
        assert (remapped[0].start, remapped[0].end) == (1.0, 2.0)
        # Second kept range starts at 5.0 on the new timeline: 11 - 10 + 5 = 6.
        assert (remapped[1].start, remapped[1].end) == (6.0, 7.0)

    def test_output_is_ordered(self):
        segments = [
            TranscriptSegment(start=11, end=12, text="b"),
            TranscriptSegment(start=1, end=2, text="a"),
        ]
        remapped = rendering.remap_transcript(segments, [(0.0, 5.0), (10.0, 15.0)])
        assert [s.start for s in remapped] == sorted(s.start for s in remapped)

    def test_text_is_preserved(self):
        segments = [TranscriptSegment(start=6, end=8, text="exact words")]
        assert rendering.remap_transcript(segments, [(5.0, 10.0)])[0].text == "exact words"


class TestTotalDuration:
    def test_sums_segments(self):
        assert rendering.total_duration([(0, 5), (10, 15)]) == 10.0

    def test_empty(self):
        assert rendering.total_duration([]) == 0.0


@requires_ffmpeg
class TestRenderEdit:
    def test_renders_a_trimmed_video(self, sample_video, tmp_path):
        out = tmp_path / "out.mp4"
        rendering.render_edit(sample_video, out, [(0.0, 3.0)], source_duration=9.0)

        assert out.is_file()
        assert extract_metadata(out).duration == pytest.approx(3.0, abs=0.4)

    def test_concatenates_multiple_segments(self, sample_video, tmp_path):
        out = tmp_path / "out.mp4"
        rendering.render_edit(sample_video, out, [(0.0, 2.0), (6.0, 8.0)], source_duration=9.0)
        assert extract_metadata(out).duration == pytest.approx(4.0, abs=0.5)

    def test_preserves_the_audio_track(self, sample_video, tmp_path):
        out = tmp_path / "out.mp4"
        rendering.render_edit(sample_video, out, [(1.0, 4.0)], source_duration=9.0)
        assert extract_metadata(out).has_audio is True

    def test_renders_a_video_with_no_audio(self, silent_video, tmp_path):
        out = tmp_path / "out.mp4"
        rendering.render_edit(silent_video, out, [(0.0, 2.0)], source_duration=3.0)
        assert out.is_file()
        assert extract_metadata(out).has_audio is False

    def test_burns_captions(self, sample_video, tmp_path):
        out = tmp_path / "out.mp4"
        captions = [TranscriptSegment(start=0.5, end=2.0, text="burned in")]
        rendering.render_edit(
            sample_video, out, [(0.0, 3.0)], captions=captions, source_duration=9.0
        )
        assert out.is_file() and out.stat().st_size > 0

    def test_burns_captions_without_cutting(self, sample_video, tmp_path):
        out = tmp_path / "out.mp4"
        captions = [TranscriptSegment(start=0.5, end=2.0, text="full length")]
        rendering.render_edit(
            sample_video, out, [(0.0, 9.0)], captions=captions, source_duration=9.0
        )
        assert extract_metadata(out).duration == pytest.approx(9.0, abs=0.6)

    def test_cut_and_caption_together(self, sample_video, tmp_path):
        out = tmp_path / "out.mp4"
        keep = [(3.0, 6.0)]
        captions = rendering.remap_transcript(
            [TranscriptSegment(start=4.0, end=5.0, text="middle")], keep
        )
        rendering.render_edit(sample_video, out, keep, captions=captions, source_duration=9.0)
        assert extract_metadata(out).duration == pytest.approx(3.0, abs=0.4)

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(RenderError, match="not found"):
            rendering.render_edit(tmp_path / "nope.mp4", tmp_path / "o.mp4", [(0, 1)])

    def test_no_segments_raises(self, sample_video, tmp_path):
        with pytest.raises(RenderError, match="no segments"):
            rendering.render_edit(sample_video, tmp_path / "o.mp4", [])

    def test_creates_missing_output_directories(self, sample_video, tmp_path):
        out = tmp_path / "deep" / "nested" / "out.mp4"
        rendering.render_edit(sample_video, out, [(0.0, 2.0)], source_duration=9.0)
        assert out.is_file()


@requires_ffmpeg
class TestExportSrt:
    def test_writes_a_file(self, tmp_path):
        path = rendering.export_srt(
            [TranscriptSegment(start=0, end=1, text="hello")], tmp_path / "s.srt"
        )
        assert "hello" in path.read_text()


@requires_ffmpeg
class TestEndToEndEditing:
    """The full edit path: suggestions -> keep segments -> rendered file."""

    def test_removal_shortens_the_video(self, sample_video, tmp_path):
        edits = [
            EditSuggestion(start=0.0, end=3.0, action="remove", reason="dead air"),
            EditSuggestion(start=3.0, end=6.0, action="caption", reason="not a removal"),
        ]
        keep = rendering.compute_keep_segments(9.0, edits)
        out = tmp_path / "edited.mp4"
        rendering.render_edit(sample_video, out, keep, source_duration=9.0)

        assert extract_metadata(out).duration == pytest.approx(6.0, abs=0.5)
