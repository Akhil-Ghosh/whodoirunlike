from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48), True)
    assert writer.isOpened()
    for index in range(4):
        frame = np.full((48, 64, 3), 35 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_health_endpoint_uses_configured_artifact_root(tmp_path: Path) -> None:
    import whodoirunlike.api as api_module

    api_module.DEFAULT_ARTIFACT_ROOT = tmp_path / "api_runs"
    client = TestClient(api_module.create_app())

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["artifact_root"] == str((tmp_path / "api_runs").resolve())


def test_initial_manifest_preserves_partial_paths_and_source_extension(tmp_path: Path) -> None:
    import whodoirunlike.api as api_module

    run_dir = tmp_path / "run"
    source_path = run_dir / "source_segment.mov"
    upload = UploadFile(
        BytesIO(),
        filename="runner.mov",
        headers=Headers({"content-type": "video/quicktime"}),
    )

    manifest_path = api_module._write_initial_manifest(
        run_dir=run_dir,
        run_id="run-001",
        source_path=source_path,
        prompt_path=run_dir / "person_prompt.json",
        pose_landmarks_path=run_dir / "pose_landmarks.jsonl",
        skeleton_render_path=run_dir / "skeleton_render.mp4",
        qa_overlay_path=run_dir / "qa_overlay.mp4",
        features_path=run_dir / "features.json",
        form_features_path=run_dir / "form_features.json",
        form_feature_arrays_path=run_dir / "form_features.npz",
        upload=upload,
        size_bytes=123,
        video_meta={"fps": 10.0, "frame_count": 4},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["version"] == 1
    assert manifest["paths"] == {
        "source_segment": str(source_path),
        "person_prompt": str(run_dir / "person_prompt.json"),
        "pose_landmarks": str(run_dir / "pose_landmarks.jsonl"),
        "skeleton_render": str(run_dir / "skeleton_render.mp4"),
        "qa_overlay": str(run_dir / "qa_overlay.mp4"),
        "features": str(run_dir / "features.json"),
        "form_features": str(run_dir / "form_features.json"),
        "form_feature_arrays": str(run_dir / "form_features.npz"),
    }
    assert manifest["stages"]["upload"]["status"] == "complete"


def test_process_clip_rejects_unsupported_upload(tmp_path: Path) -> None:
    import whodoirunlike.api as api_module

    api_module.DEFAULT_ARTIFACT_ROOT = tmp_path / "api_runs"
    client = TestClient(api_module.create_app())

    response = client.post(
        "/v1/clips",
        files={"file": ("clip.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 415


def test_process_clip_returns_metrics_and_artifact_urls(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import whodoirunlike.api as api_module

    api_module.DEFAULT_ARTIFACT_ROOT = tmp_path / "api_runs"
    video_path = tmp_path / "upload.mp4"
    _write_tiny_video(video_path)

    def fake_process_clip(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        run_dir = kwargs["run_dir"]
        for name in [
            "skeleton_render.mp4",
            "qa_overlay.mp4",
            "pose_landmarks.jsonl",
            "features.json",
            "form_features.json",
        ]:
            (run_dir / name).write_text("{}", encoding="utf-8")
        return (
            {
                "quality": {"pose_hit_rate": 1.0, "usable_rate": 1.0},
                "explainability_metrics": {"torso_lean_mean_deg": 4.2},
            },
            {"summary_features": {"stride_rhythm_proxy": 1.5}},
        )

    monkeypatch.setattr(api_module, "_process_clip", fake_process_clip)
    client = TestClient(api_module.create_app())

    with video_path.open("rb") as f:
        response = client.post(
            "/v1/clips",
            data={"model_variant": "lite"},
            files={"file": ("upload.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["quality"]["pose_hit_rate"] == 1.0
    assert payload["summary_features"]["stride_rhythm_proxy"] == 1.5
    assert payload["artifacts"]["skeleton_render"].endswith("/skeleton_render.mp4")
