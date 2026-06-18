from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from whodoirunlike.fusion_runner import densepose_group_coverage, fuse_frame, run_fused_form
from whodoirunlike.sam2_runner import write_json


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _pose_row(frame_index: int) -> dict[str, Any]:
    landmarks = []
    for index in range(33):
        landmarks.append(
            {
                "index": index,
                "name": f"landmark_{index}",
                "x": 0.5,
                "y": 0.5,
                "visibility": 0.95,
                "presence": 0.95,
            }
        )
    landmarks[23]["name"] = "left_hip"
    landmarks[23]["x"] = 0.44
    landmarks[23]["y"] = 0.55
    landmarks[25]["name"] = "left_knee"
    landmarks[25]["x"] = 0.45
    landmarks[25]["y"] = 0.65
    landmarks[27]["name"] = "left_ankle"
    landmarks[27]["x"] = 0.46
    landmarks[27]["y"] = 0.75
    return {
        "frame_index": frame_index,
        "time_seconds": frame_index / 10,
        "frame_width": 64,
        "frame_height": 48,
        "usable": True,
        "visibility_mean": 0.95,
        "bbox": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.65},
        "landmarks": landmarks,
    }


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "clip-1"
    source_path = run_dir / "source_segment.mp4"
    mask_path = run_dir / "runner_mask.mp4"
    qa_path = run_dir / "qa_overlay.mp4"
    pose_path = run_dir / "pose_landmarks.jsonl"
    densepose_path = run_dir / "densepose.jsonl"
    fused_form_path = run_dir / "fused_form.jsonl"
    fused_overlay_path = run_dir / "fused_overlay.mp4"

    frames = []
    masks = []
    for index in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 40 + index
        mask = np.zeros((48, 64, 3), dtype=np.uint8)
        mask[12:42, 16:52] = 255
        frames.append(frame)
        masks.append(mask)
    _write_video(source_path, frames)
    _write_video(mask_path, masks)
    _write_video(qa_path, frames)
    _write_jsonl(pose_path, [_pose_row(0), _pose_row(1), _pose_row(2)])
    _write_jsonl(
        densepose_path,
        [
            {
                "frame_index": index,
                "usable": True,
                "score": 0.98,
                "mask_overlap": 0.45,
                "densepose_coverage": 0.28,
                "part_count": 20,
                "runner_bbox": [16, 12, 36, 30],
                "part_pixels": {"1": 100, "2": 120, "7": 80, "11": 60, "12": 70, "14": 60},
            }
            for index in range(3)
        ],
    )
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": "clip-1",
            "paths": {
                "source_segment": str(source_path),
                "pose_landmarks": str(pose_path),
                "runner_mask": str(mask_path),
                "densepose": str(densepose_path),
                "qa_overlay": str(qa_path),
                "fused_form": str(fused_form_path),
                "fused_overlay": str(fused_overlay_path),
            },
            "stages": {},
        },
    )
    return run_dir


def test_densepose_group_coverage_groups_fine_parts() -> None:
    coverage = densepose_group_coverage({"part_pixels": {"1": 10, "2": 10, "11": 20}})

    assert coverage["torso"] == 0.5
    assert coverage["lower_legs"] == 0.5
    assert coverage["head"] == 0.0


def test_fuse_frame_combines_pose_mask_and_densepose_confidence() -> None:
    mask = np.ones((48, 64), dtype=np.uint8) * 255
    row = fuse_frame(
        _pose_row(0),
        {
            "frame_index": 0,
            "usable": True,
            "score": 0.98,
            "mask_overlap": 0.45,
            "densepose_coverage": 0.28,
            "part_count": 20,
            "runner_bbox": [16, 12, 36, 30],
            "part_pixels": {"1": 100, "2": 120, "7": 80, "11": 60, "12": 70, "14": 60},
        },
        mask=mask,
        width=64,
        height=48,
    )

    assert row["usable"] is True
    assert row["frame_confidence"] > 0.8
    assert row["densepose_group_coverage"]["torso"] > 0
    assert any(joint["name"] == "left_knee" for joint in row["joint_weights"])


def test_fuse_frame_uses_pose_mask_fallback_when_densepose_is_missing() -> None:
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:44, 12:56] = 255

    row = fuse_frame(
        _pose_row(0),
        {"frame_index": 0, "usable": False, "drop_reason": "densepose_missing"},
        mask=mask,
        width=64,
        height=48,
    )

    assert row["frame_state"] == "pose_mask_fallback"
    assert row["usable"] is True
    assert row["frame_confidence"] > 0.6
    assert row["densepose_confidence"] == 0.0
    assert row["mask_area_ratio"] > 0


def test_fuse_frame_uses_target_mask_fallback_when_pose_is_unreliable() -> None:
    pose_row = _pose_row(0)
    pose_row["visibility_mean"] = 0.08
    pose_row["bbox"] = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    for landmark in pose_row["landmarks"]:
        landmark["visibility"] = 0.05
        landmark["presence"] = 0.05
        landmark["x"] = 0.0
        landmark["y"] = 0.0
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:44, 18:42] = 255

    row = fuse_frame(
        pose_row,
        {"frame_index": 0, "usable": False, "drop_reason": "densepose_missing"},
        mask=mask,
        width=64,
        height=48,
    )

    assert row["frame_state"] == "target_mask_fallback"
    assert row["usable"] is True
    assert row["pose_reliable"] is False
    assert row["frame_confidence"] >= 0.3


def test_run_fused_form_writes_rows_overlay_and_manifest(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)

    result = run_fused_form(run_dir=run_dir)

    assert result["status"] == "complete"
    assert result["frame_count"] == 3
    assert (run_dir / "fused_form.jsonl").exists()
    assert (run_dir / "fused_overlay.mp4").exists()
    rows = [json.loads(line) for line in (run_dir / "fused_form.jsonl").read_text().splitlines()]
    assert rows[0]["frame_state"] == "usable"
    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["fused_form"]["status"] == "complete"
    assert manifest["paths"]["fused_overlay"] == str(run_dir / "fused_overlay.mp4")
