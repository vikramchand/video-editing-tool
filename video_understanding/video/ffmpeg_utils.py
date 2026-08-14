"""Thin, well-behaved wrapper around the ffmpeg/ffprobe binaries.

Every ffmpeg call in the project goes through `run()` so that failures carry
the command and stderr tail, which is the difference between a useful bug
report and "ffmpeg failed".
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from video_understanding.core.errors import FFmpegError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 900


def _resolve(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegError(
            f"`{binary}` was not found on PATH. Install FFmpeg "
            "(macOS: `brew install ffmpeg`) and try again."
        )
    return path


def run(
    args: list[str],
    *,
    binary: str = "ffmpeg",
    timeout: int = DEFAULT_TIMEOUT,
    check: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg/ffprobe with `args` and return the completed process.

    `cwd` exists for the `subtitles=` filter, whose own escaping rules make
    absolute paths containing `:` or `'` painful. Running from the file's
    directory and passing a bare filename sidesteps that entirely.
    """
    command = [_resolve(binary), *args]
    logger.debug("running: %s", " ".join(command))
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"{binary} timed out after {timeout}s", command=command
        ) from exc

    if check and proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise FFmpegError(
            f"{binary} exited with code {proc.returncode}: {tail or '(no stderr)'}",
            command=command,
            stderr=proc.stderr,
        )
    return proc


def run_ffprobe(args: list[str], *, timeout: int = 120) -> str:
    """Run ffprobe and return stdout."""
    return run(args, binary="ffprobe", timeout=timeout).stdout


def available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def version() -> str | None:
    if not available():
        return None
    try:
        first_line = run(["-version"], timeout=30).stdout.splitlines()[0]
    except (FFmpegError, IndexError):
        return None
    return first_line.strip()
