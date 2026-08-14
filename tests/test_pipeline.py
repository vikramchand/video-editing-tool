"""The pipeline and the job processor, driven with mock providers."""

from __future__ import annotations

import pytest

from tests.conftest import requires_ffmpeg
from video_understanding.ai.providers.mock import (
    MockLLMProvider,
    MockTranscriptionProvider,
    MockVisionProvider,
)
from video_understanding.core.errors import ProviderUnavailableError, ValidationError
from video_understanding.core.models import JobStatus, ProcessingStage, VideoJob
from video_understanding.jobs.pipeline import VideoPipeline
from video_understanding.jobs.processor import JobProcessor
from video_understanding.storage.database import Database


@pytest.fixture
def pipeline(settings) -> VideoPipeline:
    return VideoPipeline(
        settings,
        transcription_provider=MockTranscriptionProvider(),
        vision_provider=MockVisionProvider(),
        llm_provider=MockLLMProvider(),
    )


@requires_ffmpeg
class TestPipeline:
    def test_produces_a_complete_result(self, pipeline, sample_video):
        result = pipeline.process(sample_video)

        assert result.duration == pytest.approx(9.0, abs=0.5)
        assert result.summary
        assert result.scenes
        assert result.transcript
        assert result.highlights
        assert result.metadata is not None

    def test_scenes_are_ordered_and_contiguous(self, pipeline, sample_video):
        scenes = pipeline.process(sample_video).scenes
        for previous, current in zip(scenes, scenes[1:], strict=False):
            assert previous.end == current.start

    def test_scenes_stay_within_the_duration(self, pipeline, sample_video):
        result = pipeline.process(sample_video)
        assert result.scenes[0].start == 0.0
        assert result.scenes[-1].end <= result.duration + 0.1

    def test_highlights_and_edits_are_in_range(self, pipeline, sample_video):
        result = pipeline.process(sample_video)
        for item in [*result.highlights, *result.edit_suggestions]:
            assert 0 <= item.start < item.end <= result.duration + 0.1

    def test_reports_every_stage(self, pipeline, sample_video):
        seen: list[ProcessingStage] = []
        pipeline.process(sample_video, on_stage=lambda stage, detail: seen.append(stage))

        for expected in (
            ProcessingStage.VALIDATING,
            ProcessingStage.METADATA,
            ProcessingStage.AUDIO,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.SCENE_DETECTION,
            ProcessingStage.FRAME_EXTRACTION,
            ProcessingStage.VISION,
            ProcessingStage.REASONING,
            ProcessingStage.DONE,
        ):
            assert expected in seen

    def test_records_which_providers_ran(self, pipeline, sample_video):
        providers = pipeline.process(sample_video).providers
        assert providers["transcription"].startswith("mock")
        assert providers["vision"].startswith("mock")
        assert "scene_detector" in providers

    def test_silent_video_is_processed_with_a_warning(self, pipeline, silent_video):
        result = pipeline.process(silent_video)
        assert result.transcript == []
        assert result.scenes
        assert any("no audio" in w.lower() for w in result.warnings)

    def test_rejects_a_video_over_the_duration_limit(self, pipeline, sample_video, settings):
        settings.video_max_duration_seconds = 2.0
        with pytest.raises(ValidationError, match="exceeds"):
            pipeline.process(sample_video)

    def test_cleans_up_intermediate_files(self, pipeline, sample_video, settings, tmp_path):
        work = tmp_path / "work"
        pipeline.process(sample_video, work_dir=work)
        assert not work.exists()

    def test_keeps_intermediate_files_when_configured(self, sample_video, settings, tmp_path):
        settings.keep_intermediate_files = True
        pipeline = VideoPipeline(
            settings,
            transcription_provider=MockTranscriptionProvider(),
            vision_provider=MockVisionProvider(),
            llm_provider=MockLLMProvider(),
        )
        work = tmp_path / "work"
        pipeline.process(sample_video, work_dir=work)

        assert work.exists()
        assert list((work / "frames").glob("*.jpg"))

    def test_cleans_up_even_when_a_stage_fails(self, settings, sample_video, tmp_path):
        settings.fail_on_provider_error = True

        class Boom:
            name, model = "boom", "none"

            def analyze_frames(self, frames, context="", **kwargs):
                raise ProviderUnavailableError("vision backend is down")

            def health(self):
                return False, "down"

        pipeline = VideoPipeline(
            settings,
            transcription_provider=MockTranscriptionProvider(),
            vision_provider=Boom(),
            llm_provider=MockLLMProvider(),
        )
        work = tmp_path / "work"
        with pytest.raises(ProviderUnavailableError):
            pipeline.process(sample_video, work_dir=work)
        assert not work.exists()


@requires_ffmpeg
class TestPipelineDegradation:
    """A failing model degrades the result; it does not destroy it."""

    def test_transcription_failure_is_recorded_and_survived(self, settings, sample_video):
        class BadTranscription:
            name, model = "bad", "none"

            def transcribe(self, audio_path):
                raise ProviderUnavailableError("whisper is not installed")

            def health(self):
                return False, "down"

        pipeline = VideoPipeline(
            settings,
            transcription_provider=BadTranscription(),
            vision_provider=MockVisionProvider(),
            llm_provider=MockLLMProvider(),
        )
        result = pipeline.process(sample_video)

        assert result.transcript == []
        assert result.scenes
        assert any("Transcription failed" in w for w in result.warnings)

    def test_vision_failure_still_yields_scenes(self, settings, sample_video):
        class BadVision:
            name, model = "bad", "none"

            def analyze_frames(self, frames, context="", **kwargs):
                raise ProviderUnavailableError("ollama is not running")

            def health(self):
                return False, "down"

        pipeline = VideoPipeline(
            settings,
            transcription_provider=MockTranscriptionProvider(),
            vision_provider=BadVision(),
            llm_provider=MockLLMProvider(),
        )
        result = pipeline.process(sample_video)

        assert result.scenes
        assert result.summary
        assert any("unavailable" in w.lower() for w in result.warnings)

    def test_llm_failure_falls_back_to_heuristics(self, settings, sample_video):
        class BadLLM:
            name, model = "bad", "none"

            def generate(self, prompt, *, system="", json_mode=False):
                raise ProviderUnavailableError("ollama is not running")

            def health(self):
                return False, "down"

        pipeline = VideoPipeline(
            settings,
            transcription_provider=MockTranscriptionProvider(),
            vision_provider=MockVisionProvider(),
            llm_provider=BadLLM(),
        )
        result = pipeline.process(sample_video)

        assert result.summary
        assert result.highlights
        assert result.providers["reasoning_source"] == "heuristic"

    def test_strict_mode_turns_degradation_into_failure(self, settings, sample_video):
        settings.fail_on_provider_error = True

        class BadLLM:
            name, model = "bad", "none"

            def generate(self, prompt, *, system="", json_mode=False):
                raise ProviderUnavailableError("down")

            def health(self):
                return False, "down"

        pipeline = VideoPipeline(
            settings,
            transcription_provider=MockTranscriptionProvider(),
            vision_provider=MockVisionProvider(),
            llm_provider=BadLLM(),
        )
        with pytest.raises(ProviderUnavailableError):
            pipeline.process(sample_video)


@requires_ffmpeg
class TestJobProcessor:
    def _processor(self, settings, tmp_path, pipeline) -> tuple[JobProcessor, Database]:
        database = Database(tmp_path / "jobs.db")
        return JobProcessor(database, settings, pipeline_factory=lambda: pipeline), database

    def test_runs_a_job_to_completion(self, settings, tmp_path, pipeline, sample_video):
        processor, database = self._processor(settings, tmp_path, pipeline)
        database.create_job(VideoJob(video_id="job1", filename="sample.mp4"))

        processor.submit("job1", sample_video)
        processor.wait_for("job1", timeout=180)

        stored = database.get_job("job1")
        assert stored.status is JobStatus.COMPLETED
        assert stored.progress == 1.0
        assert stored.result is not None
        processor.shutdown()

    def test_records_progress_along_the_way(self, settings, tmp_path, pipeline, sample_video):
        processor, database = self._processor(settings, tmp_path, pipeline)
        database.create_job(VideoJob(video_id="job2", filename="sample.mp4"))

        processor.submit("job2", sample_video)
        processor.wait_for("job2", timeout=180)

        assert database.get_job("job2").stage is ProcessingStage.DONE
        processor.shutdown()

    def test_failure_is_recorded_not_raised(self, settings, tmp_path, pipeline, tmp_path_factory):
        processor, database = self._processor(settings, tmp_path, pipeline)
        database.create_job(VideoJob(video_id="job3", filename="missing.mp4"))

        processor.submit("job3", tmp_path / "does_not_exist.mp4")
        processor.wait_for("job3", timeout=60)

        stored = database.get_job("job3")
        assert stored.status is JobStatus.FAILED
        assert stored.error
        processor.shutdown()

    def test_duplicate_submissions_are_ignored(self, settings, tmp_path, pipeline, sample_video):
        processor, database = self._processor(settings, tmp_path, pipeline)
        database.create_job(VideoJob(video_id="job4", filename="sample.mp4"))

        processor.submit("job4", sample_video)
        processor.submit("job4", sample_video)
        processor.wait_for("job4", timeout=180)

        assert database.get_job("job4").status is JobStatus.COMPLETED
        processor.shutdown()

    def test_submitting_after_shutdown_raises(self, settings, tmp_path, pipeline, sample_video):
        processor, _ = self._processor(settings, tmp_path, pipeline)
        processor.shutdown()
        with pytest.raises(RuntimeError, match="shutting down"):
            processor.submit("job5", sample_video)
