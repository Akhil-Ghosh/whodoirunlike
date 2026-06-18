from __future__ import annotations

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
