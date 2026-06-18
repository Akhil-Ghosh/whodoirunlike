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
