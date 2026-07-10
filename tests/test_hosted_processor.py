from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _write_tiny_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48), True)
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((48, 64, 3), 45 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_hosted_manifest_seeds_center_target_prompt(tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor

    source_path = tmp_path / "source.mp4"
    _write_tiny_video(source_path)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "filename": "clip.mp4",
            "content_type": "video/mp4",
            "size_bytes": source_path.stat().st_size,
        },
    )

    manifest_path = hosted_processor._write_hosted_manifest(
        run_dir=tmp_path / "run",
        payload=payload,
        source_path=source_path,
        video_meta={"width": 64, "height": 48, "fps": 10.0, "frame_count": 3},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = json.loads((tmp_path / "run/person_prompt.json").read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == payload.run_id
    assert manifest["stages"]["person_prompt"]["status"] == "auto_seeded"
    assert prompt["selection"]["type"] == "auto_center_runner"
    assert prompt["selection"]["positive_points"][0]["x"] == 0.5
    assert prompt["selection"]["box"]["height"] > 0.5
    assert prompt["frame"]["frame_index"] == 0


def test_local_and_hosted_manifests_share_canonical_path_map(tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor
    from whodoirunlike.cv_flow import ReviewedClip, build_cv_run_manifest

    run_dir = tmp_path / "run"
    source_path = tmp_path / "source.mp4"
    _write_tiny_video(source_path)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "filename": "clip.mp4",
            "content_type": "video/mp4",
            "size_bytes": source_path.stat().st_size,
        },
    )
    hosted_manifest_path = hosted_processor._write_hosted_manifest(
        run_dir=run_dir,
        payload=payload,
        source_path=source_path,
        video_meta={"width": 64, "height": 48, "fps": 10.0, "frame_count": 3},
    )
    local_manifest = build_cv_run_manifest(
        ReviewedClip(
            candidate_id=payload.run_id,
            runner_name="Test Runner",
            runner_slug="test-runner",
            title="Test clip",
            source_url="",
            channel="",
            video_path=source_path,
            quality="good",
            camera_angle="side",
            start_seconds=0.0,
            end_seconds=0.3,
            notes="",
            primary_bucket="running",
        ),
        run_dir,
    )
    hosted_manifest = json.loads(hosted_manifest_path.read_text(encoding="utf-8"))

    assert hosted_manifest["paths"] == local_manifest["paths"]
    assert hosted_manifest["paths"]["target_prompt"] == hosted_manifest["paths"]["person_prompt"]


def test_write_prompt_frame_can_select_middle_frame(tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor

    source_path = tmp_path / "source.mp4"
    prompt_frame_path = tmp_path / "prompt_frame.jpg"
    _write_tiny_video(source_path)

    frame_meta = hosted_processor._write_prompt_frame(
        source_path,
        prompt_frame_path,
        frame_index=2,
    )

    prompt_frame = cv2.imread(str(prompt_frame_path))
    assert frame_meta["frame_index"] == 2
    assert prompt_frame is not None
    assert float(prompt_frame.mean()) > 60.0


def test_hosted_manifest_uses_demo_profile_prompt(tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor

    source_path = tmp_path / "source.mp4"
    _write_tiny_video(source_path)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "filename": "clip.mp4",
            "content_type": "video/mp4",
            "size_bytes": source_path.stat().st_size,
        },
    )
    demo_profile = {
        "id": "cole_hocker_reference_v1",
        "source_sha256": "sha",
        "runner_name": "Cole Hocker",
        "runner_slug": "cole-hocker",
        "prompt_frame_index": 2,
        "prompt_box": {"x": 0.6, "y": 0.2, "width": 0.2, "height": 0.7},
        "reference_artifacts": {"fused_overlay.mp4": "cole-fused.mp4"},
    }

    manifest_path = hosted_processor._write_hosted_manifest(
        run_dir=tmp_path / "run",
        payload=payload,
        source_path=source_path,
        video_meta={"width": 64, "height": 48, "fps": 10.0, "frame_count": 3},
        demo_profile=demo_profile,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = json.loads((tmp_path / "run/person_prompt.json").read_text(encoding="utf-8"))
    assert manifest["runner_name"] == "Cole Hocker"
    assert manifest["demo_profile"]["id"] == "cole_hocker_reference_v1"
    assert prompt["source"] == "hosted_upload_demo_profile_v1"
    assert prompt["frame"]["frame_index"] == 2
    assert prompt["selection"]["type"] == "reference_box"
    assert prompt["selection"]["box"]["x"] == 0.6


def test_hosted_manifest_uses_uploaded_runner_prompt(tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor

    source_path = tmp_path / "source.mp4"
    _write_tiny_video(source_path)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "filename": "clip.mp4",
            "content_type": "video/mp4",
            "size_bytes": source_path.stat().st_size,
        },
        target_prompt={
            "selection": {
                "type": "box",
                "positive_points": [{"x": 0.5, "y": 0.5, "label": "target_runner_center"}],
                "negative_points": [],
                "box": {"x": 0.42, "y": 0.15, "width": 0.22, "height": 0.7},
            },
            "frame": {"time_seconds": 0.2, "width": 64, "height": 48},
        },
    )

    manifest_path = hosted_processor._write_hosted_manifest(
        run_dir=tmp_path / "run",
        payload=payload,
        source_path=source_path,
        video_meta={"width": 64, "height": 48, "fps": 10.0, "frame_count": 3},
        uploaded_prompt=payload.target_prompt,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = json.loads((tmp_path / "run/person_prompt.json").read_text(encoding="utf-8"))
    track_seed = json.loads((tmp_path / "run/track_seed.json").read_text(encoding="utf-8"))
    assert manifest["target_prompt_source"] == "hosted_upload_user_prompt_v1"
    assert manifest["stages"]["person_prompt"]["status"] == "user_selected"
    assert track_seed["target_lock_method"] == "uploaded_runner_prompt"
    assert prompt["source"] == "hosted_upload_user_prompt_v1"
    assert prompt["frame"]["frame_index"] == 2
    assert prompt["selection"]["box"]["x"] == 0.42
    assert prompt["selection"]["positive_points"][0]["label"] == "target_runner_center"


def test_uploaded_runner_prompt_overrides_demo_profile(tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor

    source_path = tmp_path / "source.mp4"
    _write_tiny_video(source_path)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "filename": "cole-source.mp4",
            "content_type": "video/mp4",
            "size_bytes": source_path.stat().st_size,
        },
        target_prompt={
            "selection": {
                "type": "box",
                "positive_points": [{"x": 0.46, "y": 0.45, "label": "target_runner_center"}],
                "negative_points": [],
                "box": {"x": 0.36, "y": 0.14, "width": 0.2, "height": 0.62},
            },
            "frame": {"time_seconds": 0.2, "width": 64, "height": 48},
        },
    )
    demo_profile = {
        "id": "cole_hocker_reference_v1",
        "source_sha256": "sha",
        "runner_name": "Cole Hocker",
        "runner_slug": "cole-hocker",
        "prompt_frame_index": 0,
        "prompt_box": {"x": 0.6, "y": 0.2, "width": 0.2, "height": 0.7},
        "reference_artifacts": {"fused_overlay.mp4": "cole-fused.mp4"},
    }

    manifest_path = hosted_processor._write_hosted_manifest(
        run_dir=tmp_path / "run",
        payload=payload,
        source_path=source_path,
        video_meta={"width": 64, "height": 48, "fps": 10.0, "frame_count": 3},
        demo_profile=demo_profile,
        uploaded_prompt=payload.target_prompt,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = json.loads((tmp_path / "run/person_prompt.json").read_text(encoding="utf-8"))
    track_seed = json.loads((tmp_path / "run/track_seed.json").read_text(encoding="utf-8"))
    assert manifest["runner_name"] == "Uploaded runner"
    assert manifest["demo_profile"] is None
    assert manifest["target_prompt_source"] == "hosted_upload_user_prompt_v1"
    assert manifest["stages"]["person_prompt"]["status"] == "user_selected"
    assert track_seed["target_lock_method"] == "uploaded_runner_prompt"
    assert prompt["source"] == "hosted_upload_user_prompt_v1"
    assert prompt["selection"]["box"]["x"] == 0.36
    assert prompt["frame"]["frame_index"] == 2


def test_apply_demo_reference_artifacts_replaces_selected_outputs(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "cole-fused.mp4").write_bytes(b"reference fused")
    (asset_root / "cole-skeleton.mp4").write_bytes(b"reference skeleton")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "fused_overlay.mp4").write_bytes(b"bad fused")
    (run_dir / "skeleton_render.mp4").write_bytes(b"bad skeleton")
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "future_field": {"keep": True},
                "stages": {
                    "future_stage": {"status": "future"},
                    "demo_reference_artifacts": {"future_value": 7},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hosted_processor, "DEMO_ASSET_ROOT", asset_root)
    demo_profile = {
        "id": "cole_hocker_reference_v1",
        "reference_artifacts": {
            "fused_overlay.mp4": "cole-fused.mp4",
            "skeleton_render.mp4": "cole-skeleton.mp4",
        },
    }

    copied = hosted_processor._apply_demo_reference_artifacts(
        run_dir=run_dir,
        demo_profile=demo_profile,
    )

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert copied == ["fused_overlay.mp4", "skeleton_render.mp4"]
    assert (run_dir / "fused_overlay.mp4").read_bytes() == b"reference fused"
    assert (run_dir / "skeleton_render.mp4").read_bytes() == b"reference skeleton"
    assert manifest["stages"]["demo_reference_artifacts"]["status"] == "complete"
    assert manifest["stages"]["demo_reference_artifacts"]["future_value"] == 7
    assert manifest["stages"]["future_stage"] == {"status": "future"}
    assert manifest["future_field"] == {"keep": True}


def test_upload_artifacts_uses_configured_files_with_stable_public_names(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    configured_dir = tmp_path / "substituted"
    configured_dir.mkdir()
    configured_features = configured_dir / "renamed-features.json"
    configured_result = configured_dir / "renamed-result.json"
    configured_features.write_text("{}", encoding="utf-8")
    configured_result.write_text("{}", encoding="utf-8")
    (run_dir / "person_prompt.json").write_text("{}", encoding="utf-8")
    (run_dir / "source_segment.mp4").write_bytes(b"not publishable")
    (run_dir / "runner_mask_metadata.jsonl").write_text("{}", encoding="utf-8")
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "paths": {
                    "features": str(configured_features),
                    "hosted_pipeline_result": str(configured_result),
                },
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "filename": "clip.mp4",
            "content_type": "video/mp4",
            "size_bytes": 123,
        },
    )
    uploads: list[tuple[str, Path]] = []

    def capture_upload(**kwargs: Any) -> None:
        uploads.append((kwargs["name"], kwargs["path"]))

    monkeypatch.setattr(hosted_processor, "_put_worker_artifact", capture_upload)

    uploaded = hosted_processor._upload_artifacts(payload, run_dir)

    assert uploaded == [
        "cv_run_manifest.json",
        "person_prompt.json",
        "features.json",
        "hosted_pipeline_result.json",
    ]
    assert uploads == [
        ("cv_run_manifest.json", run_dir / "cv_run_manifest.json"),
        ("person_prompt.json", run_dir / "person_prompt.json"),
        ("features.json", configured_features),
        ("hosted_pipeline_result.json", configured_result),
    ]


def test_upload_artifacts_emits_result_ready_only_after_fused_overlay_put(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor
    from whodoirunlike.processing_telemetry import ProcessingTelemetry

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fused_overlay = run_dir / "fused_overlay.mp4"
    fused_overlay.write_bytes(b"viewable fused overlay")
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "paths": {"fused_overlay": str(fused_overlay)},
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="11111111-1111-4111-8111-111111111111",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
            "content_type": "video/mp4",
            "size_bytes": 123,
        },
    )
    uploaded_by_put: list[str] = []
    monkeypatch.setattr(
        hosted_processor,
        "_put_worker_artifact",
        lambda **kwargs: uploaded_by_put.append(kwargs["name"]),
    )
    telemetry = ProcessingTelemetry(
        run_id=payload.run_id,
        attempt_id=payload.attempt_id,
        local_path=tmp_path / "events.jsonl",
        wall_clock=lambda: datetime(2026, 7, 9, tzinfo=timezone.utc),
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
        sequence_start=100,
    )

    uploaded = hosted_processor._upload_artifacts(payload, run_dir, telemetry=telemetry)

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    result_ready_index = next(
        index for index, event in enumerate(events) if event["event_type"] == "result_ready"
    )
    fused_publish_complete_index = max(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "span_completed"
        and event["measurements"].get("artifact_type") == "fused_overlay"
    )
    assert uploaded == ["cv_run_manifest.json", "fused_overlay.mp4"]
    assert uploaded_by_put == uploaded
    assert result_ready_index > fused_publish_complete_index
    assert events[result_ready_index]["measurements"]["bytes"] == len(
        b"viewable fused overlay"
    )
    assert events[0]["sequence"] == 100


def test_upload_artifacts_does_not_emit_result_ready_when_fused_put_fails(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor
    from whodoirunlike.processing_telemetry import ProcessingTelemetry

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fused_overlay = run_dir / "fused_overlay.mp4"
    fused_overlay.write_bytes(b"fused")
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps({"version": 1, "paths": {"fused_overlay": str(fused_overlay)}, "stages": {}}),
        encoding="utf-8",
    )
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/source.mp4",
            "content_type": "video/mp4",
            "size_bytes": 123,
        },
    )

    def fail_fused_put(**kwargs: Any) -> None:
        if kwargs["name"] == "fused_overlay.mp4":
            raise OSError("R2 write failed")

    monkeypatch.setattr(hosted_processor, "_put_worker_artifact", fail_fused_put)
    telemetry = ProcessingTelemetry(
        run_id=payload.run_id,
        attempt_id="11111111-1111-4111-8111-111111111111",
        local_path=tmp_path / "events.jsonl",
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
    )

    with pytest.raises(OSError, match="R2 write failed"):
        hosted_processor._upload_artifacts(payload, run_dir, telemetry=telemetry)

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    assert not any(event["event_type"] == "result_ready" for event in events)
    assert events[-1]["event_type"] == "span_failed"
    assert events[-1]["span"] == "publish"


def test_artifact_put_sends_processing_attempt_header(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor

    artifact = tmp_path / "fused_overlay.mp4"
    artifact.write_bytes(b"fused")
    requests: list[Any] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

    def fake_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        requests.append(request)
        assert timeout == 120
        return FakeResponse()

    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")
    monkeypatch.setattr(hosted_processor.urllib.request, "urlopen", fake_urlopen)

    hosted_processor._put_worker_artifact(
        callback_base_url="https://api.whodoirunlike.com",
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="11111111-1111-4111-8111-111111111111",
        name="fused_overlay.mp4",
        path=artifact,
    )

    assert len(requests) == 1
    assert requests[0].get_header("X-processing-attempt-id") == (
        "11111111-1111-4111-8111-111111111111"
    )


def test_job_payload_requires_exact_allowlisted_https_callback_origin(
    monkeypatch: Any,
) -> None:
    from whodoirunlike import hosted_processor

    monkeypatch.delenv("WHODOIRUNLIKE_CALLBACK_ORIGINS", raising=False)
    monkeypatch.delenv("WHODOIRUNLIKE_ENVIRONMENT", raising=False)

    def payload(callback: str, source_url: str) -> Any:
        return hosted_processor.WorkerJobRequest(
            run_id="12345678-1234-4234-9234-123456789abc",
            callback_base_url=callback,
            source={
                "url": source_url,
                "key": "uploads/source.mp4",
                "content_type": "video/mp4",
                "size_bytes": 123,
            },
        )

    valid = payload(
        "https://api.whodoirunlike.com",
        "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
    )
    hosted_processor._validate_job_payload(valid)

    malicious_origin = payload(
        "https://api.whodoirunlike.com.evil.example",
        "https://api.whodoirunlike.com.evil.example/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
    )
    with pytest.raises(Exception, match="origin is not allowed"):
        hosted_processor._validate_job_payload(malicious_origin)

    wrong_source = payload(
        "https://api.whodoirunlike.com",
        "https://staging-api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
    )
    with pytest.raises(Exception, match="expected job source"):
        hosted_processor._validate_job_payload(wrong_source)

    callback_with_path = payload(
        "https://api.whodoirunlike.com/proxy",
        "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
    )
    with pytest.raises(Exception, match="origin is not allowed"):
        hosted_processor._validate_job_payload(callback_with_path)


def test_http_localhost_callback_requires_explicit_development(
    monkeypatch: Any,
) -> None:
    from whodoirunlike import hosted_processor

    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        callback_base_url="http://localhost:8787",
        source={
            "url": "http://localhost:8787/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/source.mp4",
            "content_type": "video/mp4",
            "size_bytes": 123,
        },
    )
    monkeypatch.setenv("WHODOIRUNLIKE_CALLBACK_ORIGINS", "http://localhost:8787")
    monkeypatch.setenv("WHODOIRUNLIKE_ENVIRONMENT", "production")
    with pytest.raises(Exception, match="origin is not allowed"):
        hosted_processor._validate_job_payload(payload)

    monkeypatch.setenv("WHODOIRUNLIKE_ENVIRONMENT", "development")
    hosted_processor._validate_job_payload(payload)


def test_processor_job_requires_shared_secret(monkeypatch: Any) -> None:
    from whodoirunlike import api as api_module

    monkeypatch.delenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", raising=False)
    client = TestClient(api_module.create_app())

    response = client.post(
        "/v1/processor/jobs",
        json={
            "run_id": "12345678-1234-4234-9234-123456789abc",
            "callback_base_url": "https://api.whodoirunlike.com",
            "source": {
                "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
                "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
                "filename": "clip.mp4",
                "content_type": "video/mp4",
                "size_bytes": 123,
            },
        },
    )

    assert response.status_code == 503


def test_processor_health_reports_full_pipeline_readiness(monkeypatch: Any, tmp_path: Path) -> None:
    from whodoirunlike import api as api_module
    from whodoirunlike import hosted_processor

    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")
    monkeypatch.setattr(hosted_processor, "DEFAULT_HOSTED_RUN_ROOT", tmp_path / "hosted_runs")
    monkeypatch.setattr(
        hosted_processor,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "backend": backend},
    )
    monkeypatch.setattr(
        hosted_processor,
        "sam31_gpu_setup_status",
        lambda: {"ready": True, "reasons": [], "backend": "sam31_gpu"},
    )
    monkeypatch.setattr(
        hosted_processor,
        "pose_setup_status",
        lambda backend: {"ready": True, "reasons": [], "backend": backend},
    )
    monkeypatch.setattr(
        hosted_processor,
        "densepose_setup_status",
        lambda: {"ready": True, "reasons": [], "backend": "densepose"},
    )
    client = TestClient(api_module.create_app())

    response = client.get("/v1/processor/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_processor_secret"] is True
    assert payload["readiness"]["ready_for_full_pipeline"] is True
    assert payload["readiness"]["checks"]["mask"]["backend"] == "sam31_gpu"


def test_processor_readiness_respects_densepose_skip(monkeypatch: Any) -> None:
    from whodoirunlike import hosted_processor

    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")
    monkeypatch.setenv("WHODOIRUNLIKE_SKIP_DENSEPOSE", "true")
    monkeypatch.setattr(
        hosted_processor,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "backend": backend},
    )
    monkeypatch.setattr(
        hosted_processor,
        "sam31_gpu_setup_status",
        lambda: {"ready": True, "reasons": [], "backend": "sam31_gpu"},
    )
    monkeypatch.setattr(
        hosted_processor,
        "pose_setup_status",
        lambda backend: {"ready": True, "reasons": [], "backend": backend},
    )
    monkeypatch.setattr(
        hosted_processor,
        "densepose_setup_status",
        lambda: {"ready": False, "reasons": ["missing densepose"]},
    )

    readiness = hosted_processor.processor_readiness()

    assert readiness["ready_for_full_pipeline"] is True
    assert readiness["checks"]["densepose"]["skipped"] is True


def test_processor_readiness_reports_check_exceptions(monkeypatch: Any) -> None:
    from whodoirunlike import hosted_processor

    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")
    monkeypatch.setenv("WHODOIRUNLIKE_SKIP_DENSEPOSE", "true")
    monkeypatch.setattr(
        hosted_processor,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "backend": backend},
    )
    monkeypatch.setattr(
        hosted_processor,
        "sam31_gpu_setup_status",
        lambda: {"ready": True, "reasons": [], "backend": "sam31_gpu"},
    )

    def broken_pose_status(backend: str) -> dict[str, Any]:
        raise RuntimeError(f"could not initialize {backend}")

    monkeypatch.setattr(hosted_processor, "pose_setup_status", broken_pose_status)

    readiness = hosted_processor.processor_readiness()

    assert readiness["ready_for_full_pipeline"] is False
    assert readiness["checks"]["pose"]["ready"] is False
    assert readiness["checks"]["pose"]["error_type"] == "RuntimeError"
    assert "could not initialize" in readiness["checks"]["pose"]["reasons"][0]


def test_densepose_setup_status_reports_missing_default_files(monkeypatch: Any, tmp_path: Path) -> None:
    from whodoirunlike import hosted_processor

    monkeypatch.setattr(hosted_processor, "DENSEPOSE_DEFAULT_CONFIG", tmp_path / "missing.yaml")
    monkeypatch.setattr(hosted_processor, "DENSEPOSE_DEFAULT_WEIGHTS", tmp_path / "missing.pkl")
    monkeypatch.delenv("DENSEPOSE_CONFIG", raising=False)
    monkeypatch.delenv("DENSEPOSE_WEIGHTS", raising=False)
    monkeypatch.setattr(
        hosted_processor.importlib.util,
        "find_spec",
        lambda name: object() if name not in {"detectron2", "densepose"} else None,
    )

    status = hosted_processor.densepose_setup_status()

    assert status["ready"] is False
    assert any("DENSEPOSE_CONFIG" in reason for reason in status["reasons"])
    assert any("DENSEPOSE_WEIGHTS" in reason for reason in status["reasons"])
    assert any("DensePose dependencies" in reason for reason in status["reasons"])


def test_process_hosted_job_emits_complete_lifecycle_with_worker_attempt_id(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor
    from whodoirunlike.processing_telemetry import ProcessingTelemetry

    run_root = tmp_path / "hosted"
    reports: list[dict[str, Any]] = []
    terminal_seen_before_final_report: list[bool] = []
    created_telemetry: list[ProcessingTelemetry] = []
    monkeypatch.setattr(hosted_processor, "DEFAULT_HOSTED_RUN_ROOT", run_root)
    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")

    def make_telemetry(**kwargs: Any) -> ProcessingTelemetry:
        telemetry = ProcessingTelemetry(
            run_id=kwargs["run_id"],
            attempt_id=kwargs["attempt_id"],
            local_path=kwargs["run_dir"] / "processing_events.jsonl",
            input_metadata=kwargs["input_metadata"],
            runtime_metadata=kwargs["runtime_metadata"],
            resource_sampler=lambda: {},
            asynchronous_delivery=False,
            sequence_start=kwargs["sequence_start"],
            attempt_elapsed_offset_seconds=kwargs["attempt_elapsed_offset_seconds"],
        )
        created_telemetry.append(telemetry)
        return telemetry

    def download(_: Any, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source video")

    def write_manifest(**kwargs: Any) -> Path:
        path = kwargs["run_dir"] / "cv_run_manifest.json"
        path.write_text(json.dumps({"version": 1, "paths": {}, "stages": {}}), encoding="utf-8")
        return path

    def run_pipeline(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["telemetry"] is created_telemetry[0]
        return {"steps": [{"stage": "identity", "result": {"status": "complete"}}]}

    def upload(_: Any, __: Path, *, telemetry: ProcessingTelemetry) -> list[str]:
        telemetry.result_ready({"artifact_type": "fused_overlay", "bytes": 42})
        return ["fused_overlay.mp4", "features.json"]

    def fail_report(**kwargs: Any) -> None:
        report = kwargs["payload"]
        reports.append(report)
        if report["status"] == "complete":
            events_path = run_root / kwargs["run_id"] / "processing_events.jsonl"
            terminal_seen_before_final_report.append(
                json.loads(events_path.read_text().splitlines()[-1])["event_type"]
                == "attempt_completed"
            )
        raise TimeoutError("transient report failure")

    monkeypatch.setattr(hosted_processor, "create_hosted_telemetry", make_telemetry)
    monkeypatch.setattr(hosted_processor, "_post_worker_report", fail_report)
    monkeypatch.setattr(hosted_processor, "_download_source", download)
    monkeypatch.setattr(hosted_processor, "_demo_upload_profile", lambda _: None)
    monkeypatch.setattr(
        hosted_processor,
        "inspect_video",
        lambda _: {"width": 1280, "height": 720, "fps": 30.0, "frame_count": 180},
    )
    monkeypatch.setattr(hosted_processor, "_write_hosted_manifest", write_manifest)
    monkeypatch.setattr(hosted_processor, "run_full_cv_pipeline", run_pipeline)
    monkeypatch.setattr(hosted_processor, "_apply_demo_reference_artifacts", lambda **_: [])
    monkeypatch.setattr(hosted_processor, "_upload_artifacts", upload)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="77777777-7777-4777-8777-777777777777",
        attempt_number=7,
        attempt_started_at="2026-07-09T11:59:30.000Z",
        processor_enqueued_at="2026-07-09T12:00:00.000Z",
        telemetry_sequence_start=120,
        runpod_job_id="runpod-job-99",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/source.mp4",
            "content_type": "video/mp4",
            "size_bytes": 12,
        },
    )

    result = hosted_processor.process_hosted_job(payload)

    assert result["status"] == "complete"
    assert result["attempt_id"] == "77777777-7777-4777-8777-777777777777"
    assert result["artifacts_uploaded"] == ["fused_overlay.mp4", "features.json"]
    assert all(
        report["attempt_id"] == "77777777-7777-4777-8777-777777777777"
        for report in reports
    )
    assert terminal_seen_before_final_report == [True]
    events_path = run_root / payload.run_id / "processing_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert events[0]["sequence"] == 120
    assert events[0]["runtime"]["attempt_number"] == 7
    assert events[0]["runtime"]["runpod_job_id"] == "runpod-job-99"
    assert events[0]["runtime"]["environment"] == "production"
    assert events[0]["runtime"]["processor_version"]
    assert next(
        event
        for event in events
        if event["event_type"] == "stage_completed" and event["stage"] == "run_preparation"
    )["input"]["duration_bucket"] == "5_10s"
    assert event_types[0] == "stage_started"
    assert events[0]["stage"] == "source_download"
    assert event_types.index("analysis_completed") < event_types.index("result_ready")
    assert event_types.index("result_ready") < event_types.index("attempt_completed")
    assert event_types[-1] == "attempt_completed"
    assert {
        "telemetry_delivery_pending": 0,
        "telemetry_delivery_failures": 0,
        "telemetry_delivery_dropped": 0,
        "telemetry_local_write_failures": 0,
    }.items() <= events[-1]["measurements"].items()
    assert [
        event["stage"] for event in events if event["event_type"] == "stage_started"
    ] == ["source_download", "run_preparation", "analysis_complete", "artifact_publish"]


def test_process_hosted_job_emits_failed_stage_and_attempt_on_processing_error(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from whodoirunlike import hosted_processor
    from whodoirunlike.processing_telemetry import ProcessingTelemetry

    run_root = tmp_path / "hosted"
    monkeypatch.setattr(hosted_processor, "DEFAULT_HOSTED_RUN_ROOT", run_root)
    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")

    def make_telemetry(**kwargs: Any) -> ProcessingTelemetry:
        return ProcessingTelemetry(
            run_id=kwargs["run_id"],
            attempt_id=kwargs["attempt_id"],
            local_path=kwargs["run_dir"] / "processing_events.jsonl",
            resource_sampler=lambda: {},
            asynchronous_delivery=False,
            sequence_start=kwargs["sequence_start"],
            attempt_elapsed_offset_seconds=kwargs["attempt_elapsed_offset_seconds"],
        )

    def fail_download(_: Any, __: Path) -> None:
        raise RuntimeError(
            "download failed from https://private.example/source token=private-token"
        )

    monkeypatch.setattr(hosted_processor, "create_hosted_telemetry", make_telemetry)
    monkeypatch.setattr(hosted_processor, "_post_worker_report", lambda **_: None)
    monkeypatch.setattr(hosted_processor, "_download_source", fail_download)
    payload = hosted_processor.WorkerJobRequest(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="99999999-9999-4999-8999-999999999999",
        callback_base_url="https://api.whodoirunlike.com",
        source={
            "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
            "key": "uploads/source.mp4",
            "content_type": "video/mp4",
            "size_bytes": 12,
        },
    )

    result = hosted_processor.process_hosted_job(payload)

    assert result["status"] == "failed"
    assert result["attempt_id"] == "99999999-9999-4999-8999-999999999999"
    events_path = run_root / payload.run_id / "processing_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "stage_started",
        "span_started",
        "span_failed",
        "stage_failed",
        "attempt_failed",
    ]
    assert events[0]["stage"] == "source_download"
    assert events[-1]["error"]["exception_type"] == "RuntimeError"
    assert "private.example" not in events[-1]["error"]["message"]
    assert "private-token" not in events[-1]["error"]["message"]


def test_processor_job_accepts_authorized_worker_job(monkeypatch: Any) -> None:
    from whodoirunlike import api as api_module
    from whodoirunlike import hosted_processor

    calls: list[str] = []
    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")
    monkeypatch.setattr(
        hosted_processor,
        "process_hosted_job",
        lambda payload: calls.append(payload.run_id),
    )
    client = TestClient(api_module.create_app())

    response = client.post(
        "/v1/processor/jobs",
        headers={"Authorization": "Bearer secret"},
        json={
            "run_id": "12345678-1234-4234-9234-123456789abc",
            "callback_base_url": "https://api.whodoirunlike.com",
            "source": {
                "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
                "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
                "filename": "clip.mp4",
                "content_type": "video/mp4",
                "size_bytes": 123,
            },
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert calls == ["12345678-1234-4234-9234-123456789abc"]
