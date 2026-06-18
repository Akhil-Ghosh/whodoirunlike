from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from whodoirunlike.densepose_runner import (
    DensePoseBackend,
    DensePoseSetupError,
    _summarize_chart_result,
    run_densepose,
)
from whodoirunlike.sam2_runner import write_json


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _make_run_dir(tmp_path: Path, *, frame_count: int = 3) -> Path:
    run_dir = tmp_path / "candidate-1"
    source_path = run_dir / "source_segment.mp4"
    mask_path = run_dir / "runner_mask.mp4"
    densepose_path = run_dir / "densepose.jsonl"
    qa_overlay_path = run_dir / "qa_overlay.mp4"

    frames = []
    masks = []
    for index in range(frame_count):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :, 1] = 30 + index
        mask = np.zeros((32, 48, 3), dtype=np.uint8)
        mask[8:24, 12:30] = 255
        frames.append(frame)
        masks.append(mask)
    _write_video(source_path, frames)
    _write_video(mask_path, masks)

    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": "candidate-1",
            "paths": {
                "source_segment": str(source_path),
                "runner_mask": str(mask_path),
                "densepose": str(densepose_path),
                "qa_overlay": str(qa_overlay_path),
            },
            "stages": {
                "densepose": {
                    "status": "pending_runner_mask",
                    "output": str(densepose_path),
                }
            },
        },
    )
    return run_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeChartResult:
    labels = _FakeTensor(np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int64))
    uv = _FakeTensor(
        np.array(
            [
                [[0.0, 0.2, 0.4], [0.6, 0.8, 0.0]],
                [[0.0, 0.1, 0.3], [0.5, 0.7, 0.0]],
            ],
            dtype=np.float32,
        )
    )


def test_summarize_chart_result_keeps_compact_part_and_uv_stats() -> None:
    summary = _summarize_chart_result(_FakeChartResult())

    assert summary["part_count"] == 2
    assert summary["part_ids"] == [1, 2]
    assert summary["part_pixels"] == {"1": 2, "2": 2}
    assert summary["part_centroids"]["1"] == {"bbox_x": 0.666667, "bbox_y": 0.25, "x": 0.666667, "y": 0.25}
    assert summary["densepose_shape"] == [3, 2]
    assert summary["densepose_coverage"] == 0.6667
    assert summary["uv_mean"] == [0.5, 0.4]


def test_run_densepose_writes_compact_rows_and_updates_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    run_dir = _make_run_dir(tmp_path)
    manifest_path = run_dir / "cv_run_manifest.json"
    densepose_path = run_dir / "densepose.jsonl"

    monkeypatch.setattr(
        "whodoirunlike.densepose_runner.load_densepose_backend",
        lambda **_: DensePoseBackend(predictor=object()),
    )

    def fake_apply(frame_bgr: np.ndarray, runner_mask: np.ndarray, backend: Any, *, frame_index: int) -> dict[str, Any]:
        assert frame_bgr.shape[:2] == runner_mask.shape
        return {
            "usable": True,
            "score": 0.91,
            "bbox": [12, 8, 18, 16],
            "mask_overlap": 0.88,
            "part_count": 7,
            "drop_reason": None,
        }

    monkeypatch.setattr("whodoirunlike.densepose_runner.apply_densepose_to_frame", fake_apply)

    result = run_densepose(run_dir=run_dir, config_path=Path("cfg.yaml"), weights_path="weights.pkl")

    assert result["status"] == "complete"
    assert result["frame_count"] == 3
    assert result["usable_frames"] == 3
    rows = _read_jsonl(densepose_path)
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["bbox"] == [12, 8, 18, 16]
    assert rows[0]["runner_bbox"] == [12, 8, 18, 16]
    assert rows[0]["part_count"] == 7

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stages"]["densepose"]["status"] == "complete"
    assert manifest["stages"]["densepose"]["output"] == str(densepose_path)
    assert manifest["stages"]["densepose"]["frame_count"] == 3
    assert manifest["stages"]["densepose"]["usable_frames"] == 3
    assert (run_dir / "qa_overlay.mp4").exists()


def test_run_densepose_marks_manifest_failed_when_optional_deps_are_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _make_run_dir(tmp_path)
    manifest_path = run_dir / "cv_run_manifest.json"

    monkeypatch.setattr(
        "whodoirunlike.densepose_runner.load_densepose_backend",
        lambda **_: (_ for _ in ()).throw(DensePoseSetupError("install densepose please")),
    )

    result = run_densepose(run_dir=run_dir)

    assert result["status"] == "failed"
    assert "install densepose please" in result["error"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    densepose_stage = manifest["stages"]["densepose"]
    assert densepose_stage["status"] == "failed"
    assert densepose_stage["frame_count"] == 0
    assert densepose_stage["usable_frames"] == 0
    assert "Detectron2" in densepose_stage["setup_instructions"]
