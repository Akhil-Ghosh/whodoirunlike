from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
        json.dumps({"stages": {}}),
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
