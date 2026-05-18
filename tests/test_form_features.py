from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from whodoirunlike.form_features import (
    ANGLE_NAMES,
    compile_form_features,
    joint_angles,
    normalize_pose_xy,
)
from whodoirunlike.sam2_runner import write_json


def _landmarks(frame_index: int) -> list[dict[str, float | int | str]]:
    rows = []
    for index in range(33):
        rows.append(
            {
                "index": index,
                "name": str(index),
                "x": 0.35 + (index % 6) * 0.035 + frame_index * 0.004,
                "y": 0.25 + (index // 6) * 0.055,
                "z": 0.0,
                "visibility": 0.92,
                "presence": 0.9,
            }
        )
    rows[11] |= {"x": 0.42, "y": 0.3}
    rows[12] |= {"x": 0.58, "y": 0.3}
    rows[23] |= {"x": 0.44, "y": 0.56}
    rows[24] |= {"x": 0.56, "y": 0.56}
    rows[25] |= {"x": 0.42 + frame_index * 0.008, "y": 0.72 - frame_index * 0.01}
    rows[26] |= {"x": 0.58 - frame_index * 0.006, "y": 0.72 + frame_index * 0.006}
    rows[27] |= {"x": 0.38 + frame_index * 0.008, "y": 0.9}
    rows[28] |= {"x": 0.62 - frame_index * 0.006, "y": 0.9}
    rows[31] |= {"x": 0.34 + frame_index * 0.008, "y": 0.94}
    rows[32] |= {"x": 0.66 - frame_index * 0.006, "y": 0.94}
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_normalize_pose_xy_centers_hips_and_keeps_scale_signal() -> None:
    xy = np.full((1, 33, 2), np.nan, dtype=np.float32)
    xy[0, 11] = [0.4, 0.3]
    xy[0, 12] = [0.6, 0.3]
    xy[0, 23] = [0.45, 0.7]
    xy[0, 24] = [0.55, 0.7]

    normalized, hip_mid, scale = normalize_pose_xy(xy)

    assert np.allclose(hip_mid[0], [0.5, 0.7])
    assert scale[0] > 0
    assert np.allclose(np.nanmean(normalized[0, [23, 24]], axis=0), [0.0, 0.0])


def test_joint_angles_outputs_named_angle_columns() -> None:
    xy = np.zeros((2, 33, 2), dtype=np.float32)
    xy[:, 23] = [0.0, 0.0]
    xy[:, 25] = [0.0, 1.0]
    xy[:, 27] = [1.0, 1.0]

    angles = joint_angles(xy)

    assert angles.shape == (2, len(ANGLE_NAMES))
    assert np.isclose(angles[0, ANGLE_NAMES.index("left_knee")], 90.0)


def test_compile_form_features_writes_metadata_and_match_arrays(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip-001"
    run_dir.mkdir()
    pose_path = run_dir / "pose_landmarks.jsonl"
    fused_path = run_dir / "fused_form.jsonl"
    densepose_path = run_dir / "densepose.jsonl"
    metadata_path = run_dir / "form_features.json"
    arrays_path = run_dir / "form_features.npz"
    pose_rows = [
        {
            "frame_index": index,
            "time_seconds": index / 30,
            "usable": True,
            "visibility_mean": 0.9,
            "landmarks": _landmarks(index),
            "world_landmarks": [],
        }
        for index in range(6)
    ]
    fused_rows = [
        {
            "frame_index": index,
            "frame_state": "usable",
            "frame_confidence": 0.8,
            "pose_confidence": 0.9,
            "questionable_joints": ["left_ankle"] if index == 2 else [],
            "densepose_group_coverage": {"torso": 0.2, "upper_legs": 0.25, "lower_legs": 0.2},
            "joint_weights": [{"index": joint, "weight": 0.75} for joint in range(33)],
        }
        for index in range(6)
    ]
    densepose_rows = [
        {
            "frame_index": index,
            "usable": True,
            "part_pixels": {"1": 50, "7": 40, "11": 30},
            "part_centroids": {
                "1": {"x": 0.5, "y": 0.45},
                "7": {"x": 0.45, "y": 0.65},
                "11": {"x": 0.42, "y": 0.8},
            },
        }
        for index in range(6)
    ]
    _write_jsonl(pose_path, pose_rows)
    _write_jsonl(fused_path, fused_rows)
    _write_jsonl(densepose_path, densepose_rows)
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "candidate_id": "clip-001",
            "runner_name": "Test Runner",
            "runner_slug": "test-runner",
            "source": {"url": "https://example.com/watch"},
            "review": {
                "quality": "good",
                "camera_angle": "side",
                "primary_bucket": "800_1500",
                "duration_seconds": 0.2,
            },
            "paths": {
                "pose_landmarks": str(pose_path),
                "fused_form": str(fused_path),
                "densepose": str(densepose_path),
                "form_features": str(metadata_path),
                "form_feature_arrays": str(arrays_path),
            },
            "stages": {},
        },
    )

    result = compile_form_features(run_dir=run_dir)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as arrays:
        assert arrays["pose_xy_normalized"].shape == (6, 33, 2)
        assert arrays["bone_vectors"].shape[0] == 6
        assert arrays["joint_angles"].shape[1] == len(ANGLE_NAMES)
        assert arrays["joint_weights"][2, 27] == np.float32(0.75)
        assert arrays["densepose_groups"].shape[0] == 6
    assert result["status"] == "complete"
    assert metadata["candidate_id"] == "clip-001"
    assert metadata["quality"]["usable_rate"] == 1.0
    assert metadata["questionable_joints_by_frame"][2] == ["left_ankle"]
    assert "arm_swing_amplitude" in metadata["summary_features"]
