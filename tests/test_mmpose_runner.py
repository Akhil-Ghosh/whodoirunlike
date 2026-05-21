from __future__ import annotations

from typing import Any

import numpy as np

from whodoirunlike import mmpose_runner
from whodoirunlike.mmpose_runner import (
    mmpose_row_to_pose_row,
    mmpose_setup_status,
    rtmlib_arrays_to_predictions,
    select_mmpose_prediction,
)


def landmark(index: int, *, x: float, y: float, score: float = 0.9) -> dict[str, Any]:
    return {
        "index": index,
        "name": f"source_{index}",
        "x": x,
        "y": y,
        "score": score,
    }


def test_mmpose_row_to_pose_row_maps_wholebody_keypoints_to_canonical_pose() -> None:
    landmarks = [landmark(index, x=0.01, y=0.02, score=0.0) for index in range(133)]
    landmarks[5] = landmark(5, x=0.22, y=0.31)
    landmarks[6] = landmark(6, x=0.42, y=0.32)
    landmarks[13] = landmark(13, x=0.28, y=0.63)
    landmarks[14] = landmark(14, x=0.47, y=0.64)
    landmarks[17] = landmark(17, x=0.27, y=0.91)
    landmarks[19] = landmark(19, x=0.24, y=0.88)
    landmarks[20] = landmark(20, x=0.49, y=0.92)
    landmarks[22] = landmark(22, x=0.52, y=0.89)
    landmarks[95] = landmark(95, x=0.19, y=0.44)
    landmarks[120] = landmark(120, x=0.51, y=0.45)

    row = {
        "frame_index": 3,
        "time_seconds": 0.1,
        "frame_width": 1920,
        "frame_height": 1080,
        "detected": True,
        "usable": True,
        "landmarks": landmarks,
    }

    mapped = mmpose_row_to_pose_row(row, backend="mmpose_rtmw_l_384")

    assert mapped["source_pose_backend"] == "mmpose_rtmw_l_384"
    assert mapped["usable"] is True
    assert mapped["landmarks"][11]["x"] == 0.22
    assert mapped["landmarks"][12]["x"] == 0.42
    assert mapped["landmarks"][25]["y"] == 0.63
    assert mapped["landmarks"][29]["x"] == 0.24
    assert mapped["landmarks"][31]["y"] == 0.91
    assert mapped["landmarks"][21]["source_index"] == 95
    assert mapped["landmarks"][20]["source_index"] == 120
    assert mapped["visibility_mean"] > 0.0


def test_select_mmpose_prediction_prefers_mask_overlap() -> None:
    off_target = {
        "keypoints": np.asarray([[80, 80], [88, 88], [84, 90]], dtype=np.float32),
        "keypoint_scores": np.asarray([0.95, 0.9, 0.88], dtype=np.float32),
    }
    on_target = {
        "keypoints": np.asarray([[15, 15], [28, 20], [18, 31]], dtype=np.float32),
        "keypoint_scores": np.asarray([0.82, 0.84, 0.86], dtype=np.float32),
    }

    selected_index, selected, bbox, mask_iou = select_mmpose_prediction(
        [off_target, on_target],
        crop={"x": 0, "y": 0, "width": 100, "height": 100},
        frame_width=100,
        frame_height=100,
        mask_bbox={"x": 0.1, "y": 0.1, "width": 0.25, "height": 0.25},
    )

    assert selected_index == 1
    assert selected is on_target
    assert bbox is not None
    assert mask_iou > 0


def test_rtmlib_arrays_to_predictions_normalizes_single_person_arrays() -> None:
    predictions = rtmlib_arrays_to_predictions(
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        np.asarray([0.9, 0.8], dtype=np.float32),
    )

    assert len(predictions) == 1
    keypoints = predictions[0]["keypoints"]
    scores = predictions[0]["keypoint_scores"]
    assert keypoints.shape == (2, 2)
    np.testing.assert_allclose(scores, np.asarray([0.9, 0.8], dtype=np.float32))


def test_mmpose_setup_status_reports_missing_optional_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(mmpose_runner.importlib.util, "find_spec", lambda _name: None)

    status = mmpose_setup_status("mmpose_rtmw_l_384")

    assert status["ready"] is False
    assert "rtmlib" in status["reasons"][0]
    assert status["backend"] == "mmpose_rtmw_l_384"
