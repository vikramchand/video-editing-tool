"""Editing MVP: turn edit decisions into an actual video file.

Scope is deliberately narrow — removing segments, concatenating what is left,
and burning in captions. That is enough to prove the understanding layer
produces *actionable* output rather than just a description.

B-roll, music and effects are explicitly out of scope for the MVP. The seam for
them is `compute_keep_segments` -> `render_edit`: a future version adds new
`EditAction` handling in the planner without changing the renderer's contract.

One subtlety worth stating: when segments are removed, caption timings must be
remapped onto the *edited* timeline. `remap_transcript` does that, and it is the
reason rendering is a two-pass operation (cut, then burn) rather than one.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from video_understanding.ai.transcription import to_srt
from video_understanding.core.errors import RenderError
from video_understanding.core.models import (
    EditSuggestion,
    Highlight,
    TranscriptSegment,
)
from video_understanding.video import ffmpeg_utils
from video_understanding.video.audio import has_audio_stream

logger = logging.getLogger(__name__)

Segment = tuple[float, float]

# Captions are burned for silent autoplay, so they need to be legible on a phone.
CAPTION_STYLE = (
    "FontName=Helvetica,FontSize=18,PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=40"
)


def merge_ranges(ranges: list[Segment], gap_tolerance: float = 0.05) -> list[Segment]:
    """Sort and merge overlapping or touching ranges."""
    if not ranges:
        return []
    ordered = sorted((min(a, b), max(a, b)) for a, b in ranges)
    merged: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def subtract_ranges(base: list[Segment], holes: list[Segment]) -> list[Segment]:
    """Remove `holes` from `base`, returning what remains."""
    remaining = list(base)
    for hole_start, hole_end in merge_ranges(holes):
        next_remaining: list[Segment] = []
        for start, end in remaining:
            if hole_end <= start or hole_start >= end:
                next_remaining.append((start, end))  # no overlap
                continue
            if hole_start > start:
                next_remaining.append((start, hole_start))
            if hole_end < end:
                next_remaining.append((hole_end, end))
        remaining = next_remaining
    return [(round(a, 3), round(b, 3)) for a, b in remaining if b - a > 0.05]


def compute_keep_segments(
    duration: float,
    edit_suggestions: list[EditSuggestion],
    *,
    apply_removals: bool = True,
    highlights_only: bool = False,
    highlights: list[Highlight] | None = None,
) -> list[Segment]:
    """Decide which parts of the source survive into the edit.

    `highlights_only` builds a highlight reel instead of a trim: keep only the
    highlighted windows, then still drop anything marked for removal inside them.
    """
    if duration <= 0:
        return []

    if highlights_only:
        windows = [(h.start, min(h.end, duration)) for h in (highlights or []) if h.end > h.start]
        keep = merge_ranges(windows)
        if not keep:
            logger.warning("highlights_only requested but no highlights exist; keeping everything")
            keep = [(0.0, round(duration, 3))]
    else:
        keep = [(0.0, round(duration, 3))]

    if apply_removals:
        removals = [
            (s.start, min(s.end, duration))
            for s in edit_suggestions
            if s.action == "remove" and s.end > s.start
        ]
        if removals:
            keep = subtract_ranges(keep, removals)

    if not keep:
        # Every frame was marked for removal. Refusing is more honest than
        # silently producing an empty file.
        raise RenderError(
            "the requested edit removes the entire video; nothing would remain to render"
        )
    return keep


def remap_transcript(
    segments: list[TranscriptSegment], keep: list[Segment]
) -> list[TranscriptSegment]:
    """Rewrite transcript timings onto the edited timeline.

    Each kept range is concatenated in order, so a segment's new start is its
    offset within its kept range plus the total duration of all preceding ranges.
    Speech that straddles a cut is clipped to the surviving part.
    """
    remapped: list[TranscriptSegment] = []
    elapsed = 0.0

    for keep_start, keep_end in keep:
        window = keep_end - keep_start
        for segment in segments:
            overlap_start = max(segment.start, keep_start)
            overlap_end = min(segment.end, keep_end)
            if overlap_end <= overlap_start:
                continue
            remapped.append(
                segment.model_copy(
                    update={
                        "start": round(elapsed + (overlap_start - keep_start), 3),
                        "end": round(elapsed + (overlap_end - keep_start), 3),
                    }
                )
            )
        elapsed += window

    remapped.sort(key=lambda s: s.start)
    return remapped


def render_edit(
    video_path: str | Path,
    output_path: str | Path,
    keep: list[Segment],
    *,
    captions: list[TranscriptSegment] | None = None,
    source_duration: float = 0.0,
) -> Path:
    """Render the edited video.

    Pass 1 cuts and concatenates; pass 2 burns captions, if requested.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise RenderError(f"source video not found: {video_path}")
    if not keep:
        raise RenderError("no segments to render")

    with tempfile.TemporaryDirectory(prefix="vu-render-") as tmp:
        tmp_dir = Path(tmp)
        needs_captions = bool(captions)
        cut_target = (tmp_dir / "cut.mp4") if needs_captions else output_path

        untouched = (
            len(keep) == 1
            and keep[0][0] <= 0.05
            and source_duration > 0
            and keep[0][1] >= source_duration - 0.05
        )
        if untouched and needs_captions:
            # Nothing to cut; burn captions straight from the source.
            cut_target = video_path
        else:
            _cut_and_concat(video_path, cut_target, keep)

        if needs_captions:
            _burn_captions(cut_target, output_path, captions or [], tmp_dir)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RenderError("rendering produced an empty file")

    logger.info("rendered %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def _cut_and_concat(video_path: Path, output_path: Path, keep: list[Segment]) -> None:
    """Single-pass trim + concat using a filter graph.

    Frame-accurate, and avoids the concat-demuxer's requirement that every
    segment start on a keyframe (which is what makes naive `-ss`/`-c copy`
    trimming drift).
    """
    with_audio = has_audio_stream(video_path)

    filters: list[str] = []
    concat_inputs: list[str] = []

    for index, (start, end) in enumerate(keep):
        filters.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if with_audio:
            filters.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")

    streams_per_segment = 2 if with_audio else 1
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(keep)}:v=1:a={1 if with_audio else 0}"
        + ("[outv][outa]" if with_audio else "[outv]")
    )

    args = [
        "-y",
        "-i", str(video_path),
        "-filter_complex", ";".join(filters),
        "-map", "[outv]",
    ]
    if with_audio:
        args += ["-map", "[outa]", "-c:a", "aac", "-b:a", "128k"]
    args += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",   # required for QuickTime/Safari playback
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(output_path),
    ]

    logger.info(
        "cutting %d segment(s) (%d stream(s) each)", len(keep), streams_per_segment
    )
    ffmpeg_utils.run(args)


def _burn_captions(
    video_path: Path,
    output_path: Path,
    captions: list[TranscriptSegment],
    tmp_dir: Path,
) -> None:
    """Burn captions into the video with the `subtitles` filter."""
    if not captions:
        raise RenderError("caption burning was requested but there are no caption segments")

    srt_path = tmp_dir / "captions.srt"
    srt_path.write_text(to_srt(captions), encoding="utf-8")

    # Run from tmp_dir so the filter sees a bare filename and needs no escaping.
    source = video_path.resolve()
    destination = output_path.resolve()

    logger.info("burning %d caption segment(s)", len(captions))
    ffmpeg_utils.run(
        [
            "-y",
            "-i", str(source),
            "-vf", f"subtitles=captions.srt:force_style='{CAPTION_STYLE}'",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            str(destination),
        ],
        cwd=str(tmp_dir),
    )


def export_srt(segments: list[TranscriptSegment], output_path: str | Path) -> Path:
    """Write transcript segments to an .srt sidecar file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_srt(segments), encoding="utf-8")
    return output_path


def total_duration(segments: list[Segment]) -> float:
    return round(sum(end - start for start, end in segments), 3)
