"""The library: everything ever uploaded, searchable and paginated.

Separate from `/v1/videos` because it answers a different question. That
endpoint is the record of one job; this one is the index the UI browses, and it
is shaped for that — denormalised cards, a total for the pager, and unfiltered
per-status counts so the filter chips can show what a filter would find before
it is applied.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from video_understanding.api.dependencies import get_config, get_database
from video_understanding.core.config import Settings
from video_understanding.core.models import JobStatus, LibraryResponse
from video_understanding.storage.database import Database
from video_understanding.video.rendering import render_path

router = APIRouter(prefix="/v1/library", tags=["library"])


@router.get("", response_model=LibraryResponse)
@router.get("/", response_model=LibraryResponse, include_in_schema=False)
async def get_library(
    q: str | None = Query(default=None, description="Search filename, title, summary and topics"),
    status: JobStatus | None = Query(default=None, description="Only videos in this status"),
    sort: str = Query(default="newest", pattern="^(newest|oldest|longest|shortest|name)$"),
    limit: int = Query(default=24, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_config),
) -> LibraryResponse:
    """Browse uploaded videos."""
    items = database.list_summaries(
        limit=limit, offset=offset, query=q, status=status, sort=sort
    )
    for item in items:
        item.has_render = render_path(settings.renders_dir, item.video_id).is_file()

    return LibraryResponse(
        items=items,
        total=database.count_jobs(query=q, status=status),
        limit=limit,
        offset=offset,
        counts=database.status_counts(),
    )
