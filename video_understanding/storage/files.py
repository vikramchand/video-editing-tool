"""Locating the file behind a job.

`jobs.source_path` records an absolute path, which stops being true the moment
the same data directory is opened from somewhere else — the common case being a
video uploaded through the container (`/app/data/uploads/x.mp4`) and then
browsed from a server running on the host, or a data directory that was moved.

The upload filename is `<video_id><ext>` by construction, so the file can always
be found again from the id alone. Resolution tries the recorded path first and
falls back to the uploads directory, which keeps old rows playable instead of
turning them into dead links.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_source(
    source_path: str | None, video_id: str, uploads_dir: str | Path
) -> Path | None:
    """Return the video file for a job, or None if it is genuinely gone."""
    if source_path:
        recorded = Path(source_path)
        if recorded.is_file():
            return recorded

    uploads = Path(uploads_dir)

    # Same basename, current uploads directory.
    if source_path:
        relocated = uploads / Path(source_path).name
        if relocated.is_file():
            logger.debug("resolved %s to %s", source_path, relocated)
            return relocated

    # Last resort: any extension, matched on the id the file was named after.
    for candidate in sorted(uploads.glob(f"{video_id}.*")):
        if candidate.is_file():
            return candidate

    return None
