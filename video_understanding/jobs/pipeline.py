"""The video understanding pipeline.

    video -> ffmpeg -> audio -> whisper ------------.
                    \\                                > structured -> LLM -> JSON
                     -> scenes -> frames -> vision -'

One class, one public method, stages in order. Both the API worker and the CLI
drive this same object; there is no second implementation of the pipeline.

Every stage reports through `on_stage`, which is what makes processing
observable rather than a black box between upload and result.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from video_understanding.ai.providers import registry
from video_understanding.ai.reasoning import ReasoningEngine
from video_understanding.ai.transcription import TranscriptionService
from video_understanding.ai.vision import VisionAnalyzer, aggregate_entities
from video_understanding.core.config import Settings, get_settings
from video_understanding.core.models import (
    ProcessingStage,
    Transcript,
    VideoUnderstanding,
)
from video_understanding.video import frames as frames_module
from video_understanding.video import scenes as scenes_module
from video_understanding.video.audio import extract_audio
from video_understanding.video.metadata import validate_upload

logger = logging.getLogger(__name__)

StageCallback = Callable[[ProcessingStage, str], None]


class VideoPipeline:
    """Runs a video through every understanding stage."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transcription_provider=None,
        vision_provider=None,
        llm_provider=None,
    ) -> None:
        self.settings = settings or get_settings()
        strict = self.settings.fail_on_provider_error

        # Providers are injectable so tests can drive the real pipeline with
        # mocks, and so a future GPU backend drops in without changes here.
        self.transcription = TranscriptionService(
            transcription_provider or registry.get_transcription_provider(self.settings)
        )
        self.vision = VisionAnalyzer(
            vision_provider or registry.get_vision_provider(self.settings), strict=strict
        )
        self.reasoning = ReasoningEngine(
            llm_provider or registry.get_llm_provider(self.settings), strict=strict
        )

    def process(
        self,
        video_path: str | Path,
        *,
        work_dir: str | Path | None = None,
        on_stage: StageCallback | None = None,
    ) -> VideoUnderstanding:
        """Process one video and return its structured understanding."""
        video_path = Path(video_path)
        started = time.monotonic()
        warnings: list[str] = []

        def report(stage: ProcessingStage, detail: str = "") -> None:
            logger.info("[%s] %s", stage.value, detail or "...")
            if on_stage is not None:
                on_stage(stage, detail)

        work = Path(work_dir) if work_dir else self.settings.work_dir / video_path.stem
        work.mkdir(parents=True, exist_ok=True)

        try:
            # 1-3. Validate and extract metadata -----------------------------
            report(ProcessingStage.VALIDATING, f"validating {video_path.name}")
            metadata = validate_upload(video_path, self.settings)

            report(
                ProcessingStage.METADATA,
                f"{metadata.duration:.1f}s, {metadata.width}x{metadata.height}, "
                f"{metadata.fps:.1f} fps, audio={'yes' if metadata.has_audio else 'no'}",
            )

            # 4. Extract audio ------------------------------------------------
            report(ProcessingStage.AUDIO, "extracting audio track")
            audio_path = extract_audio(video_path, work / "audio.wav")
            if audio_path is None:
                warnings.append("Video has no audio track; transcription was skipped.")

            # 5. Transcribe ---------------------------------------------------
            report(ProcessingStage.TRANSCRIBING, "transcribing speech")
            transcript = self._transcribe(audio_path, metadata.duration, warnings)
            report(
                ProcessingStage.TRANSCRIBING,
                f"{len(transcript.segments)} transcript segment(s)",
            )

            # 6. Detect scenes ------------------------------------------------
            report(ProcessingStage.SCENE_DETECTION, "detecting scene boundaries")
            boundaries = scenes_module.detect_scenes(
                video_path, metadata.duration, self.settings
            )
            report(
                ProcessingStage.SCENE_DETECTION,
                f"{len(boundaries)} scene(s) "
                f"via {boundaries[0].detector if boundaries else 'none'}",
            )

            # 7. Sample representative frames ---------------------------------
            report(ProcessingStage.FRAME_EXTRACTION, "sampling representative frames")
            frame_list = frames_module.extract_frames(
                video_path, boundaries, work / "frames", self.settings
            )
            report(
                ProcessingStage.FRAME_EXTRACTION,
                f"{len(frame_list)} frame(s) from {len(boundaries)} scene(s)",
            )

            # 8-9. Vision analysis, fused with the transcript ------------------
            report(ProcessingStage.VISION, f"analysing {len(frame_list)} frame(s)")

            def vision_progress(done: int, total: int) -> None:
                report(ProcessingStage.VISION, f"analysed scene {done}/{total}")

            scene_list, vision_warnings = self.vision.analyze(
                boundaries, frame_list, transcript, on_progress=vision_progress
            )
            warnings.extend(vision_warnings)

            # 10-11. Reason over the structured representation -----------------
            report(ProcessingStage.REASONING, "reasoning over the structured representation")
            reasoning = self.reasoning.analyze(
                metadata, transcript, scene_list, filename=video_path.name
            )
            warnings.extend(reasoning.warnings)

            people, objects, actions = aggregate_entities(scene_list)

            understanding = VideoUnderstanding(
                duration=round(metadata.duration, 3),
                summary=reasoning.summary,
                title=reasoning.title,
                topics=reasoning.topics,
                metadata=metadata,
                transcript=transcript.segments,
                transcript_language=transcript.language,
                scenes=scene_list,
                highlights=reasoning.highlights,
                edit_suggestions=reasoning.edit_suggestions,
                people=people,
                objects=objects,
                actions=actions,
                providers={
                    "transcription": f"{self.transcription.provider.name}:"
                    f"{getattr(self.transcription.provider, 'model', '?')}",
                    "vision": f"{self.vision.provider.name}:"
                    f"{getattr(self.vision.provider, 'model', '?')}",
                    "reasoning": f"{self.reasoning.provider.name}:"
                    f"{getattr(self.reasoning.provider, 'model', '?')}",
                    "reasoning_source": reasoning.source,
                    "scene_detector": boundaries[0].detector if boundaries else "none",
                },
                warnings=warnings,
            )

            report(
                ProcessingStage.DONE,
                f"completed in {time.monotonic() - started:.1f}s",
            )
            return understanding

        finally:
            if not self.settings.keep_intermediate_files:
                shutil.rmtree(work, ignore_errors=True)

    def _transcribe(
        self, audio_path: Path | None, duration: float, warnings: list[str]
    ) -> Transcript:
        """Transcribe, degrading to an empty transcript on provider failure."""
        try:
            return self.transcription.transcribe(audio_path, duration)
        except Exception as exc:  # noqa: BLE001 - degradation is the point
            if self.settings.fail_on_provider_error:
                raise
            logger.warning("transcription failed: %s", exc)
            warnings.append(f"Transcription failed: {exc}")
            return Transcript(
                provider=getattr(self.transcription.provider, "name", "unknown"), model="none"
            )
