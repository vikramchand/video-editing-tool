"""Command-line interface.

    video-understand my_video.mp4

Runs the same pipeline the API runs, in-process and synchronously, then prints
a human-readable report and writes the full JSON to `output/my_video.json`.

Deliberately dependency-free (argparse + ANSI): the CLI is often the first
thing someone runs, and it should not be able to fail on a missing TUI library.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from video_understanding import __version__
from video_understanding.core.config import get_hardware, get_settings
from video_understanding.core.errors import VideoUnderstandingError
from video_understanding.core.models import ProcessingStage, VideoUnderstanding
from video_understanding.jobs.pipeline import VideoPipeline
from video_understanding.video import ffmpeg_utils

# --- Terminal helpers -------------------------------------------------------

_COLOR = sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(text: str) -> str:
    return _paint(text, "1")


def dim(text: str) -> str:
    return _paint(text, "2")


def green(text: str) -> str:
    return _paint(text, "32")


def yellow(text: str) -> str:
    return _paint(text, "33")


def red(text: str) -> str:
    return _paint(text, "31")


def cyan(text: str) -> str:
    return _paint(text, "36")


def _format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:d}:{secs:02d}"


class ProgressReporter:
    """Renders pipeline stages as they happen."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.started = time.monotonic()
        self._last_stage: ProcessingStage | None = None

    def __call__(self, stage: ProcessingStage, detail: str) -> None:
        if self.quiet:
            return

        # Close off the previous stage's line before starting a new one, so each
        # stage leaves exactly one finished row behind and updates within a
        # stage overwrite in place.
        if self._last_stage is not None and stage is not self._last_stage:
            sys.stdout.write("\n")
        self._last_stage = stage

        elapsed = time.monotonic() - self.started
        percent = int(stage.progress * 100)
        bar_width = 24
        filled = int(bar_width * stage.progress)
        bar = "#" * filled + "-" * (bar_width - filled)

        sys.stdout.write(
            f"\r  [{cyan(bar)}] {percent:3d}%  "
            f"{bold(stage.value.replace('_', ' ')):<28} "
            f"{detail[:44]:<44} {dim(f'{elapsed:5.1f}s')}"
        )
        sys.stdout.flush()

    def finish(self) -> None:
        if not self.quiet:
            sys.stdout.write("\n")
            sys.stdout.flush()


# --- Report rendering -------------------------------------------------------


def print_report(result: VideoUnderstanding, source: Path) -> None:
    """Human-readable summary of the analysis."""
    rule = "=" * 78
    print()
    print(bold(rule))
    print(bold(f"  {result.title or source.name}"))
    print(bold(rule))

    meta = result.metadata
    if meta:
        print(
            f"  {dim('Duration')} {_format_time(result.duration)} "
            f"({result.duration:.1f}s)   "
            f"{dim('Resolution')} {meta.width}x{meta.height} ({meta.orientation})   "
            f"{dim('FPS')} {meta.fps:.1f}"
        )
    if result.providers:
        models = ", ".join(f"{k}={v}" for k, v in result.providers.items())
        print(f"  {dim('Models')} " + dim(models))

    print()
    print(bold("SUMMARY"))
    for line in _wrap(result.summary or "(none)", 74):
        print(f"  {line}")

    if result.topics:
        print(f"\n  {dim('Topics:')} {', '.join(result.topics)}")

    # Scenes ---------------------------------------------------------------
    print()
    print(bold(f"SCENES ({len(result.scenes)})"))
    for scene in result.scenes:
        marker = _importance_marker(scene.importance)
        print(
            f"  {cyan(f'{scene.start:6.1f}s - {scene.end:6.1f}s')} "
            f"{marker} {dim(f'{scene.importance:.2f}')}  {scene.description[:70]}"
        )
        details = []
        if scene.people:
            details.append(f"people: {', '.join(scene.people[:3])}")
        if scene.objects:
            details.append(f"objects: {', '.join(scene.objects[:4])}")
        if scene.actions:
            details.append(f"actions: {', '.join(scene.actions[:4])}")
        for detail in details:
            print(f"           {dim(detail)}")

    # Highlights -----------------------------------------------------------
    if result.highlights:
        print()
        print(bold(f"HIGHLIGHTS ({len(result.highlights)})"))
        for highlight in result.highlights:
            print(
                f"  {green(f'{highlight.start:6.1f}s - {highlight.end:6.1f}s')} "
                f"{dim(f'score {highlight.score:.2f}')}  {highlight.reason[:64]}"
            )

    # Edit suggestions -----------------------------------------------------
    if result.edit_suggestions:
        print()
        print(bold(f"EDIT SUGGESTIONS ({len(result.edit_suggestions)})"))
        for edit in result.edit_suggestions:
            colorize = red if edit.action == "remove" else yellow
            print(
                f"  {f'{edit.start:6.1f}s - {edit.end:6.1f}s':>18} "
                f"{colorize(f'{edit.action:<8}')} {edit.reason[:60]}"
            )

    # Transcript -----------------------------------------------------------
    if result.transcript:
        print()
        print(bold(f"TRANSCRIPT ({len(result.transcript)} segments)"))
        for segment in result.transcript[:8]:
            print(f"  {dim(f'{segment.start:6.1f}s')}  {segment.text[:66]}")
        if len(result.transcript) > 8:
            print(dim(f"  ... {len(result.transcript) - 8} more segments"))
    else:
        print(f"\n{bold('TRANSCRIPT')}\n  {dim('(no speech detected)')}")

    if result.warnings:
        print()
        print(yellow(bold(f"WARNINGS ({len(result.warnings)})")))
        for warning in result.warnings:
            print(f"  {yellow('!')} {warning}")

    print()


def _importance_marker(importance: float) -> str:
    if importance >= 0.75:
        return green("***")
    if importance >= 0.5:
        return yellow("** ")
    return dim("*  ")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return lines or [""]


# --- Commands ---------------------------------------------------------------


def cmd_process(args: argparse.Namespace) -> int:
    settings = get_settings()
    source = Path(args.video).expanduser()

    if not source.is_file():
        print(red(f"error: file not found: {source}"), file=sys.stderr)
        return 2

    if not ffmpeg_utils.available():
        print(
            red("error: ffmpeg/ffprobe not found on PATH.\n")
            + "  macOS:  brew install ffmpeg\n"
            + "  Ubuntu: sudo apt install ffmpeg",
            file=sys.stderr,
        )
        return 3

    if not args.quiet:
        print(
            f"\n{bold('video-understand')} {dim(f'v{__version__}')}  "
            f"{dim(get_hardware().describe())}"
        )
        print(f"  {dim('source')}  {source}")
        summary = ", ".join(f"{k}={v}" for k, v in settings.provider_summary().items())
        print(f"  {dim('models')}  " + summary)
        print()

    reporter = ProgressReporter(quiet=args.quiet)
    pipeline = VideoPipeline(settings)

    try:
        result = pipeline.process(source, on_stage=reporter)
    except VideoUnderstandingError as exc:
        reporter.finish()
        print(red(f"\nerror: {exc}"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        reporter.finish()
        print(yellow("\ninterrupted"), file=sys.stderr)
        return 130
    reporter.finish()

    if not args.quiet:
        print_report(result, source)

    # Save JSON --------------------------------------------------------------
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else settings.output_dir / f"{source.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"{green('saved')} {output_path}")

    # Optional render --------------------------------------------------------
    if args.render or args.captions:
        rendered = _render(result, source, args, settings)
        if rendered is not None:
            print(f"{green('rendered')} {rendered}")

    return 0


def _render(
    result: VideoUnderstanding, source: Path, args: argparse.Namespace, settings
) -> Path | None:
    from video_understanding.video import rendering  # noqa: PLC0415

    try:
        keep = rendering.compute_keep_segments(
            result.duration,
            result.edit_suggestions,
            apply_removals=args.render,
            highlights_only=args.highlights_only,
            highlights=result.highlights,
        )
        captions = (
            rendering.remap_transcript(result.transcript, keep)
            if (args.captions and result.transcript)
            else None
        )
        if args.captions and not captions:
            print(yellow("warning: --captions requested but no transcript is available"))

        output_path = settings.output_dir / f"{source.stem}_edit.mp4"
        rendering.render_edit(
            source, output_path, keep, captions=captions, source_duration=result.duration
        )
    except VideoUnderstandingError as exc:
        print(red(f"error: rendering failed: {exc}"), file=sys.stderr)
        return None

    print(
        dim(
            f"  {result.duration:.1f}s -> {rendering.total_duration(keep):.1f}s "
            f"across {len(keep)} segment(s)"
        )
    )
    return output_path


def cmd_check(args: argparse.Namespace) -> int:
    """Verify the local environment before processing anything."""
    from video_understanding.ai.providers import registry  # noqa: PLC0415

    settings = get_settings()
    hardware = get_hardware()

    print(f"\n{bold('video-understand check')} {dim(f'v{__version__}')}\n")
    print(f"  {'hardware':<16} {hardware.describe()}")
    print(f"  {'in container':<16} {str(hardware.in_container).lower()}")

    ffmpeg_version = ffmpeg_utils.version()
    ok = ffmpeg_version is not None
    print(f"  {'ffmpeg':<16} {green(ffmpeg_version) if ok else red('NOT FOUND')}")

    print(f"  {'data dir':<16} {settings.data_dir.resolve()}")
    print(f"  {'ollama host':<16} {settings.ollama_host}")
    print()

    all_ok = ok
    for kind, detail in registry.health_report(settings).items():
        healthy = detail.startswith("ok")
        all_ok = all_ok and healthy
        marker = green("ok") if healthy else red("!!")
        print(f"  {marker} {kind:<14} {detail}")

    print()
    if all_ok:
        print(green("everything is ready."))
        return 0
    print(yellow("some components are unavailable; see the README for setup steps."))
    return 1


def cmd_reset(args: argparse.Namespace) -> int:
    """Delete the database and the artefacts it points at, then recreate the tree.

    Rows and files have to go together. A job row without its upload is a dead
    link in the library; an upload without its row is a file nothing will ever
    clean up. So this clears both, and leaves the directory structure in the
    state `ensure_directories` would have created on a fresh install.

    Paths come from `Settings`, never from a hard-coded `data/`, so a relocated
    DATA_DIR or DATABASE_PATH is reset rather than silently skipped.
    """
    settings = get_settings()

    db = settings.db_path
    # WAL mode means the database is up to three files. Removing only the main
    # one can leave committed rows behind in the -wal sidecar.
    db_files = [db, db.with_name(f"{db.name}-wal"), db.with_name(f"{db.name}-shm")]
    directories = [
        settings.uploads_dir,
        settings.work_dir,
        settings.renders_dir,
        settings.thumbnails_dir,
    ]

    doomed = [p for p in db_files if p.exists()] + [d for d in directories if d.is_dir()]
    if not doomed:
        print(dim("nothing to delete."))
        return 0

    print(f"\n{bold('this permanently deletes:')}")
    for path in doomed:
        print(f"  {red('x')} {path.resolve()}")
    print()

    if not args.yes:
        try:
            answer = input("type 'yes' to continue: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "yes":
            print(yellow("aborted, nothing was deleted."))
            return 1

    for path in db_files:
        path.unlink(missing_ok=True)
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)

    settings.ensure_directories()
    print(green("reset complete.") + dim(f"  ({settings.data_dir.resolve()})"))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API server."""
    import uvicorn  # noqa: PLC0415

    settings = get_settings()
    uvicorn.run(
        "video_understanding.api.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


# --- Entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-understand",
        description="Understand a video locally: transcript, scenes, highlights, edits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  video-understand my_video.mp4\n"
            "  video-understand my_video.mp4 --render --captions\n"
            "  video-understand my_video.mp4 --highlights-only --render\n"
            "  video-understand check\n"
            "  video-understand serve --port 8000\n"
            "  video-understand reset\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="verify ffmpeg and model availability")
    check.set_defaults(func=cmd_check)

    serve = subparsers.add_parser("serve", help="run the REST API server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    serve.set_defaults(func=cmd_serve)

    reset = subparsers.add_parser("reset", help="delete the database and all stored videos")
    reset.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    reset.set_defaults(func=cmd_reset)

    process = subparsers.add_parser("process", help="analyse a video file (default command)")
    _add_process_arguments(process)
    process.set_defaults(func=cmd_process)

    parser.set_defaults(func=cmd_process, video=None)

    return parser


SUBCOMMANDS = frozenset({"check", "serve", "process", "reset"})


def normalize_argv(argv: list[str]) -> list[str]:
    """Make `process` the implicit default subcommand.

    Argparse cannot have both subparsers and a top-level positional, but
    `video-understand my_video.mp4` is the documented main entry point. So when
    the first argument is not a subcommand, insert `process` ahead of it.
    """
    if not argv:
        return argv
    first = argv[0]
    if first in SUBCOMMANDS:
        return argv
    if first in ("-h", "--help", "--version"):
        return argv
    return ["process", *argv]


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", nargs="?", help="path to the video file")
    parser.add_argument("-o", "--output", help="where to write the JSON result")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the saved path")
    parser.add_argument(
        "--render", action="store_true", help="also render an edit with removals applied"
    )
    parser.add_argument(
        "--captions", action="store_true", help="burn captions into the rendered edit"
    )
    parser.add_argument(
        "--highlights-only",
        action="store_true",
        help="render a highlight reel instead of a trimmed original",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(normalize_argv(list(argv if argv is not None else sys.argv[1:])))

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if args.func is cmd_process and not getattr(args, "video", None):
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
