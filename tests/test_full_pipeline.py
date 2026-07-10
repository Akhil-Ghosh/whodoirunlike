from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from whodoirunlike import full_pipeline


def test_densepose_runtime_kwargs_resolve_env_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "densepose.yaml"
    weights_path = tmp_path / "densepose.pkl"
    monkeypatch.setenv("DENSEPOSE_CONFIG", str(config_path))
    monkeypatch.setenv("DENSEPOSE_WEIGHTS", str(weights_path))
    monkeypatch.setenv("DENSEPOSE_DEVICE", "cuda")

    kwargs = full_pipeline._densepose_runtime_kwargs()

    assert kwargs == {
        "config_path": config_path.resolve(),
        "weights_path": str(weights_path.resolve()),
        "device": "cuda",
    }


def test_full_pipeline_passes_densepose_runtime_and_stops_on_densepose_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "densepose.yaml"
    weights_path = tmp_path / "densepose.pkl"
    captured: dict[str, Any] = {}

    monkeypatch.setenv("DENSEPOSE_CONFIG", str(config_path))
    monkeypatch.setenv("DENSEPOSE_WEIGHTS", str(weights_path))
    monkeypatch.setenv("DENSEPOSE_DEVICE", "cuda")
    monkeypatch.setattr(
        full_pipeline,
        "run_identity_tracking",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        full_pipeline,
        "run_sam31_mlx_mask",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        "whodoirunlike.pose_runner.run_pose_landmarks",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        full_pipeline,
        "run_fused_form",
        lambda **_: pytest.fail("fusion should not run after DensePose fails"),
    )

    def fake_run_densepose(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "failed", "error": "densepose boom"}

    monkeypatch.setattr("whodoirunlike.densepose_runner.run_densepose", fake_run_densepose)

    with pytest.raises(RuntimeError, match="densepose boom"):
        full_pipeline.run_full_cv_pipeline(run_dir=tmp_path, pose_backend="mediapipe")

    assert captured["run_dir"] == tmp_path
    assert captured["config_path"] == config_path.resolve()
    assert captured["weights_path"] == str(weights_path.resolve())
    assert captured["device"] == "cuda"


def test_full_pipeline_skip_densepose_uses_configured_path_and_records_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "candidate-1"
    densepose_path = tmp_path / "custom-artifacts" / "body-map.jsonl"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "candidate-1",
                "paths": {"densepose": str(densepose_path)},
                "stages": {
                    "densepose": {"status": "pending_runner_mask", "custom_field": "keep"},
                    "future_stage": {"status": "future"},
                },
                "custom_manifest_field": {"keep": True},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_sam31_mlx_mask", lambda **_: {"status": "complete"})
    monkeypatch.setattr(
        "whodoirunlike.pose_runner.run_pose_landmarks",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(full_pipeline, "run_fused_form", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "compile_form_features", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "export_cv_tables", lambda *_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", lambda *_: {"status": "complete"})

    result = full_pipeline.run_full_cv_pipeline(
        run_dir=run_dir,
        pose_backend="mediapipe",
        skip_densepose=True,
    )

    assert densepose_path.read_text(encoding="utf-8") == ""
    assert not (run_dir / "densepose.jsonl").exists()
    assert next(step for step in result["steps"] if step["stage"] == "densepose")["result"] == {
        "status": "skipped"
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["stages"]["densepose"] == {
        "status": "skipped",
        "custom_field": "keep",
        "output": str(densepose_path),
    }
    assert manifest["stages"]["future_stage"] == {"status": "future"}
    assert manifest["custom_manifest_field"] == {"keep": True}
