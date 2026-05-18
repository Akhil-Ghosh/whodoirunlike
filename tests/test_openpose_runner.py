from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whodoirunlike.openpose_runner import (
    BODY25_NAMES,
    compare_openpose_to_mediapipe,
    openpose_setup_status,
    run_openpose_comparison,
    select_openpose_person,
)
from whodoirunlike.sam2_runner import write_json


def _body25_points(*, x: float, y: float, score: float = 0.8) -> list[float]:
    values: list[float] = []
    for index in range(len(BODY25_NAMES)):
        values.extend([x + index * 0.5, y + index * 0.25, score])
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_openpose_setup_status_reports_missing_binary(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENPOSE_BIN", raising=False)
    monkeypatch.delenv("OPENPOSE_MODEL_FOLDER", raising=False)
    monkeypatch.setattr("whodoirunlike.openpose_runner.shutil.which", lambda _: None)

    status = openpose_setup_status()

    assert status["ready"] is False
    assert "OpenPose binary not found" in status["reasons"][0]
    assert status["env"]["binary"] == "OPENPOSE_BIN"


def test_select_openpose_person_prefers_runner_mask_overlap() -> None:
    people = [
        {"pose_keypoints_2d": _body25_points(x=5.0, y=12.0, score=0.95)},
        {"pose_keypoints_2d": _body25_points(x=62.0, y=14.0, score=0.55)},
    ]

    selected_index, _points, bbox, mask_iou = select_openpose_person(
        people,
        width=100,
        height=100,
        mask_bbox={"x": 0.6, "y": 0.12, "width": 0.22, "height": 0.16},
    )

    assert selected_index == 1
    assert bbox is not None
    assert mask_iou > 0


def test_compare_openpose_to_mediapipe_writes_summary(tmp_path: Path) -> None:
    openpose_path = tmp_path / "openpose_landmarks.jsonl"
    mediapipe_path = tmp_path / "pose_landmarks.jsonl"
    output_path = tmp_path / "pose_comparison.json"

    openpose_landmarks = [
        {"index": index, "name": name, "x": 0.2 + index * 0.001, "y": 0.3, "score": 0.8}
        for index, name in enumerate(BODY25_NAMES)
    ]
    media_landmarks = [
        {"index": index, "name": f"mp_{index}", "x": 0.2 + index * 0.001, "y": 0.3, "visibility": 0.9}
        for index in range(33)
    ]
    _write_jsonl(
        openpose_path,
        [
            {
                "frame_index": 0,
                "usable": True,
                "bbox": {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.4},
                "landmarks": openpose_landmarks,
            }
        ],
    )
    _write_jsonl(
        mediapipe_path,
        [
            {
                "frame_index": 0,
                "usable": True,
                "bbox": {"x": 0.22, "y": 0.2, "width": 0.2, "height": 0.4},
                "landmarks": media_landmarks,
            }
        ],
    )

    summary = compare_openpose_to_mediapipe(
        openpose_landmarks_path=openpose_path,
        mediapipe_landmarks_path=mediapipe_path,
        output_path=output_path,
    )

    assert summary["frame_count"] == 1
    assert summary["both_usable_frames"] == 1
    assert summary["bbox_iou_mean"] > 0
    assert summary["keypoint_pairs"] > 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary


def test_run_openpose_comparison_marks_manifest_unavailable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("OPENPOSE_BIN", raising=False)
    monkeypatch.setattr("whodoirunlike.openpose_runner.shutil.which", lambda _: None)
    run_dir = tmp_path / "candidate-1"
    manifest_path = run_dir / "cv_run_manifest.json"
    write_json(
        manifest_path,
        {
            "version": 1,
            "candidate_id": "candidate-1",
            "paths": {
                "source_segment": str(run_dir / "source_segment.mp4"),
                "runner_mask": str(run_dir / "runner_mask.mp4"),
                "pose_landmarks": str(run_dir / "pose_landmarks.jsonl"),
            },
            "stages": {},
        },
    )

    result = run_openpose_comparison(run_dir=run_dir)

    assert result["status"] == "unavailable"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stages"]["openpose"]["status"] == "unavailable"
    assert manifest["paths"]["pose_comparison"].endswith("pose_comparison.json")
