"""Structured representation of a video.

These models are the contract between every stage of the pipeline and the REST
API. Stages communicate only through these types, which is what makes the
model layer swappable: a hosted GPU vision model and a local Ollama one both
have to produce a `SceneAnalysis`, and nothing downstream can tell them apart.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EditAction = Literal["keep", "remove", "trim", "zoom", "caption", "broll"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    """Lifecycle of an uploaded video."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(StrEnum):
    """Observable pipeline stages, reported through the status endpoint.

    The order of declaration is the order of execution and is used to compute
    a coarse progress percentage.
    """

    PENDING = "pending"
    VALIDATING = "validating"
    METADATA = "extracting_metadata"
    AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    SCENE_DETECTION = "detecting_scenes"
    FRAME_EXTRACTION = "extracting_frames"
    VISION = "analyzing_frames"
    REASONING = "reasoning"
    PERSISTING = "persisting"
    DONE = "done"

    @property
    def index(self) -> int:
        return list(ProcessingStage).index(self)

    @property
    def progress(self) -> float:
        """Fraction of the pipeline completed once this stage finishes."""
        stages = list(ProcessingStage)
        return round(self.index / (len(stages) - 1), 3)


class VideoMetadata(BaseModel):
    """Container/stream facts extracted with ffprobe."""

    duration: float = Field(ge=0, description="Duration in seconds")
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    fps: float = Field(ge=0)
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False
    size_bytes: int = Field(ge=0, default=0)
    container: str | None = None
    rotation: int = 0
    created_at: str | None = Field(
        default=None, description="Creation time reported by the container, if present"
    )

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def orientation(self) -> Literal["portrait", "landscape", "square", "unknown"]:
        if not self.width or not self.height:
            return "unknown"
        if self.width == self.height:
            return "square"
        return "portrait" if self.height > self.width else "landscape"


class TranscriptSegment(BaseModel):
    """One utterance-sized chunk of speech."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> TranscriptSegment:
        if self.end < self.start:
            raise ValueError(f"segment end ({self.end}) precedes start ({self.start})")
        return self

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def duration(self) -> float:
        return self.end - self.start


class Transcript(BaseModel):
    """Full speech transcription for a video."""

    language: str | None = None
    text: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _derive_text(self) -> Transcript:
        if not self.text and self.segments:
            self.text = " ".join(s.text for s in self.segments).strip()
        return self

    def between(self, start: float, end: float) -> list[TranscriptSegment]:
        """Segments overlapping the [start, end) window."""
        return [s for s in self.segments if s.end > start and s.start < end]

    def text_between(self, start: float, end: float) -> str:
        return " ".join(s.text for s in self.between(start, end)).strip()


class SceneBoundary(BaseModel):
    """A detected cut, before any AI analysis is attached."""

    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    detector: str = "unknown"

    @model_validator(mode="after")
    def _end_after_start(self) -> SceneBoundary:
        if self.end <= self.start:
            raise ValueError(f"scene {self.id}: end ({self.end}) must exceed start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class Frame(BaseModel):
    """A representative still sampled from a scene."""

    scene_id: int
    timestamp: float = Field(ge=0)
    path: str


class SceneAnalysis(BaseModel):
    """What the vision model saw in one scene's representative frames.

    This is the strict schema the vision prompt asks for. Field names are kept
    short and unambiguous because small local VLMs follow simple schemas far
    more reliably than nested ones.
    """

    model_config = ConfigDict(populate_by_name=True)

    scene_id: int = 0
    description: str = ""
    people: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    environment: str = ""
    notable_events: list[str] = Field(default_factory=list)
    significance: float = Field(default=0.5, ge=0, le=1)
    mood: str | None = None

    @field_validator("people", "objects", "actions", "notable_events", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        """Small VLMs frequently answer with a string where a list was asked for."""
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace(";", ",").split(",")]
            return [p for p in parts if p and p.lower() not in {"none", "n/a", "unknown"}]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v)]

    @field_validator("significance", mode="before")
    @classmethod
    def _coerce_significance(cls, v: object) -> float:
        if v is None:
            return 0.5
        try:
            value = float(v)
        except (TypeError, ValueError):
            return 0.5
        # Models sometimes answer on a 0-100 or 0-10 scale despite the prompt.
        if value > 1.0:
            value = value / 100.0 if value > 10.0 else value / 10.0
        return max(0.0, min(1.0, value))


class Scene(BaseModel):
    """A scene with boundaries, visual analysis and the speech inside it."""

    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    description: str = ""
    people: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    environment: str = ""
    importance: float = Field(default=0.5, ge=0, le=1)
    keyframes: list[float] = Field(default_factory=list)
    transcript_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class Highlight(BaseModel):
    """A moment worth keeping."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    reason: str
    score: float = Field(ge=0, le=1)
    title: str | None = None


class EditSuggestion(BaseModel):
    """A proposed edit over a time range.

    `action` is deliberately a closed set: the rendering layer switches on it,
    so an unknown action is a bug rather than a no-op.
    """

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    action: EditAction
    reason: str
    confidence: float = Field(default=0.5, ge=0, le=1)

    @property
    def duration(self) -> float:
        return self.end - self.start


class VideoUnderstanding(BaseModel):
    """The complete structured understanding of one video."""

    duration: float = Field(ge=0)
    summary: str = ""
    title: str | None = None
    topics: list[str] = Field(default_factory=list)
    metadata: VideoMetadata | None = None
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    transcript_language: str | None = None
    scenes: list[Scene] = Field(default_factory=list)
    highlights: list[Highlight] = Field(default_factory=list)
    edit_suggestions: list[EditSuggestion] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    providers: dict[str, str] = Field(
        default_factory=dict, description="Which provider/model produced each layer"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal degradations, e.g. skipped transcription"
    )


class VideoJob(BaseModel):
    """A processing job and, once finished, its result."""

    video_id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    stage: ProcessingStage = ProcessingStage.PENDING
    progress: float = Field(default=0.0, ge=0, le=1)
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    source_path: str | None = None
    result: VideoUnderstanding | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)


# --- API payloads -----------------------------------------------------------


class UploadResponse(BaseModel):
    video_id: str
    status: JobStatus
    filename: str


class StatusResponse(BaseModel):
    video_id: str
    status: JobStatus
    stage: ProcessingStage
    progress: float
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class VideoResponse(BaseModel):
    video_id: str
    status: JobStatus
    stage: ProcessingStage
    progress: float
    error: str | None = None
    filename: str
    created_at: datetime
    updated_at: datetime
    result: VideoUnderstanding | None = None
    has_render: bool = Field(
        default=False, description="Whether a rendered edit for this video is already on disk"
    )


class VideoSummary(BaseModel):
    """One row of the library.

    Everything here comes from the job's own columns, so a page of the library
    costs one query and no JSON parsing. It is intentionally a different shape
    from `VideoResponse`: a card needs counts, a detail view needs the data.
    """

    video_id: str
    filename: str
    status: JobStatus
    stage: ProcessingStage
    progress: float
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    summary: str = ""
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    scene_count: int = 0
    highlight_count: int = 0
    transcript_count: int = 0
    has_audio: bool = False
    topics: list[str] = Field(default_factory=list)
    has_render: bool = False


class LibraryResponse(BaseModel):
    """A page of the library, plus the totals its filter chips need."""

    items: list[VideoSummary]
    total: int = Field(description="Videos matching the current filters")
    limit: int
    offset: int
    counts: dict[str, int] = Field(
        default_factory=dict, description="Unfiltered totals per status, plus 'all'"
    )


class RenderRequest(BaseModel):
    """Ask the renderer to cut a video down using edit decisions."""

    apply_removals: bool = True
    burn_captions: bool = False
    highlights_only: bool = False
    edit_suggestions: list[EditSuggestion] | None = Field(
        default=None,
        description="Override the stored suggestions. Defaults to the ones from the pipeline.",
    )


class RenderResponse(BaseModel):
    video_id: str
    output_path: str
    download_url: str
    source_duration: float
    output_duration: float
    segments_kept: int
    segments_removed: int
    captions_burned: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    ffmpeg: bool
    database: bool
    providers: dict[str, str]
    checks: dict[str, str] = Field(default_factory=dict)
