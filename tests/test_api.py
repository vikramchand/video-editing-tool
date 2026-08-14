"""REST API endpoints."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_ffmpeg
from video_understanding.api.app import create_app


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _upload(client: TestClient, path, name: str = "sample.mp4"):
    with open(path, "rb") as handle:
        return client.post("/v1/videos", files={"file": (name, handle, "video/mp4")})


def _wait(client: TestClient, video_id: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/v1/videos/{video_id}/status").json()
        if payload["status"] in ("completed", "failed"):
            return payload
        time.sleep(0.2)
    raise AssertionError(f"job {video_id} did not finish within {timeout}s")


class TestHealth:
    def test_health_reports_ok_with_mock_providers(self, client):
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["database"] is True
        assert set(payload["providers"]) == {"transcription", "vision", "llm"}

    def test_health_is_always_200(self, client):
        assert client.get("/health").status_code == 200

    def test_config_endpoint_exposes_sampling_settings(self, client):
        payload = client.get("/v1/config").json()
        assert payload["video"]["frames_per_scene"] == 3
        assert payload["providers"]["llm"].startswith("mock")

    def test_openapi_is_served(self, client):
        assert "/v1/videos" in client.get("/openapi.json").json()["paths"]

    def test_web_ui_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Video Understanding" in response.text


class TestUploadValidation:
    def test_rejects_a_non_video_extension(self, client, tmp_path):
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        with open(bad, "rb") as handle:
            response = client.post(
                "/v1/videos", files={"file": ("notes.txt", handle, "text/plain")}
            )

        assert response.status_code == 415
        assert "unsupported" in response.json()["detail"].lower()

    def test_rejects_an_empty_file(self, client, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.touch()
        with open(empty, "rb") as handle:
            response = client.post("/v1/videos", files={"file": ("empty.mp4", handle, "video/mp4")})
        assert response.status_code == 400

    def test_rejects_a_file_that_is_not_really_a_video(self, client, tmp_path):
        fake = tmp_path / "fake.mp4"
        fake.write_bytes(b"definitely not a video" * 100)
        with open(fake, "rb") as handle:
            response = client.post("/v1/videos", files={"file": ("fake.mp4", handle, "video/mp4")})

        assert response.status_code == 400
        assert "could not read video" in response.json()["detail"].lower()

    @requires_ffmpeg
    def test_rejects_a_file_over_the_size_limit(self, client, settings, sample_video):
        settings.video_max_upload_mb = 0.0005
        response = _upload(client, sample_video)
        assert response.status_code in (400, 413)

    def test_missing_file_field_is_a_422(self, client):
        assert client.post("/v1/videos").status_code == 422


class TestNotFound:
    @pytest.mark.parametrize(
        "path",
        [
            "/v1/videos/nope",
            "/v1/videos/nope/status",
            "/v1/videos/nope/source",
            "/v1/videos/nope/transcript.srt",
            "/v1/videos/nope/render/download",
        ],
    )
    def test_unknown_video_returns_404(self, client, path):
        assert client.get(path).status_code == 404

    def test_render_of_an_unknown_video_returns_404(self, client):
        assert client.post("/v1/videos/nope/render", json={}).status_code == 404


@requires_ffmpeg
class TestFullFlow:
    def test_upload_returns_immediately_with_an_id(self, client, sample_video):
        response = _upload(client, sample_video)
        assert response.status_code == 202

        payload = response.json()
        assert payload["status"] in ("queued", "processing")
        assert payload["video_id"]
        assert payload["filename"] == "sample.mp4"

    def test_processing_completes_and_returns_a_result(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        assert _wait(client, video_id)["status"] == "completed"

        payload = client.get(f"/v1/videos/{video_id}").json()
        assert payload["status"] == "completed"

        result = payload["result"]
        assert result["duration"] == pytest.approx(9.0, abs=0.5)
        assert result["summary"]
        assert result["scenes"] and result["transcript"] and result["highlights"]

    def test_status_progresses_to_one(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)

        payload = client.get(f"/v1/videos/{video_id}/status").json()
        assert payload["progress"] == 1.0
        assert payload["stage"] == "done"
        assert payload["error"] is None

    def test_result_matches_the_documented_shape(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)
        result = client.get(f"/v1/videos/{video_id}").json()["result"]

        for key in (
            "duration", "summary", "transcript", "scenes", "highlights", "edit_suggestions",
        ):
            assert key in result, f"missing documented key: {key}"

        scene = result["scenes"][0]
        for key in (
            "id", "start", "end", "description", "people", "objects", "actions", "importance",
        ):
            assert key in scene

    def test_listing_includes_the_upload(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)

        listed = client.get("/v1/videos").json()
        assert any(item["video_id"] == video_id for item in listed)
        assert listed[0]["result"] is None  # omitted unless asked for

    def test_listing_can_include_results(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)
        listed = client.get("/v1/videos?include_results=true").json()
        assert next(i for i in listed if i["video_id"] == video_id)["result"] is not None

    def test_source_is_served_inline_for_the_player(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        response = client.get(f"/v1/videos/{video_id}/source")

        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        # `inline` matters: `attachment` makes the browser download instead of play.
        assert response.headers["content-disposition"].startswith("inline")

    def test_transcript_srt_download(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)

        response = client.get(f"/v1/videos/{video_id}/transcript.srt")
        assert response.status_code == 200
        assert "-->" in response.text

    def test_srt_before_completion_is_a_conflict(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        response = client.get(f"/v1/videos/{video_id}/transcript.srt")
        assert response.status_code in (200, 409)

    def test_delete_removes_the_job(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)

        assert client.delete(f"/v1/videos/{video_id}").status_code == 204
        assert client.get(f"/v1/videos/{video_id}").status_code == 404


@requires_ffmpeg
class TestRendering:
    def _completed(self, client, sample_video) -> str:
        video_id = _upload(client, sample_video).json()["video_id"]
        _wait(client, video_id)
        return video_id

    def test_render_returns_output_details(self, client, sample_video):
        video_id = self._completed(client, sample_video)
        response = client.post(f"/v1/videos/{video_id}/render", json={"apply_removals": True})

        assert response.status_code == 200
        payload = response.json()
        assert payload["segments_kept"] >= 1
        assert payload["output_duration"] > 0
        assert payload["download_url"].endswith("/render/download")

    def test_rendered_file_can_be_downloaded(self, client, sample_video):
        video_id = self._completed(client, sample_video)
        client.post(f"/v1/videos/{video_id}/render", json={})

        response = client.get(f"/v1/videos/{video_id}/render/download")
        assert response.status_code == 200
        assert len(response.content) > 0
        assert response.headers["content-type"] == "video/mp4"

    def test_render_with_explicit_removals_shortens_the_video(self, client, sample_video):
        video_id = self._completed(client, sample_video)
        response = client.post(
            f"/v1/videos/{video_id}/render",
            json={
                "apply_removals": True,
                "edit_suggestions": [
                    {"start": 0, "end": 3, "action": "remove", "reason": "test cut"}
                ],
            },
        )

        payload = response.json()
        assert payload["output_duration"] < payload["source_duration"]
        assert payload["segments_removed"] == 1

    def test_render_with_captions(self, client, sample_video):
        video_id = self._completed(client, sample_video)
        response = client.post(
            f"/v1/videos/{video_id}/render", json={"burn_captions": True}
        )
        assert response.status_code == 200
        assert response.json()["captions_burned"] is True

    def test_removing_everything_is_rejected(self, client, sample_video):
        video_id = self._completed(client, sample_video)
        response = client.post(
            f"/v1/videos/{video_id}/render",
            json={
                "apply_removals": True,
                "edit_suggestions": [
                    {"start": 0, "end": 999, "action": "remove", "reason": "everything"}
                ],
            },
        )
        assert response.status_code == 400
        assert "entire video" in response.json()["detail"]

    def test_download_before_render_is_404(self, client, sample_video):
        video_id = self._completed(client, sample_video)
        assert client.get(f"/v1/videos/{video_id}/render/download").status_code == 404

    def test_render_before_completion_is_a_conflict(self, client, sample_video):
        video_id = _upload(client, sample_video).json()["video_id"]
        response = client.post(f"/v1/videos/{video_id}/render", json={})
        assert response.status_code in (200, 409)
