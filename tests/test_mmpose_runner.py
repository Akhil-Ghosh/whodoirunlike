from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import cv2

from whodoirunlike import mmpose_runner
from whodoirunlike.mmpose_runner import (
    mmpose_row_to_pose_row,
    mmpose_setup_status,
    rtmlib_arrays_to_predictions,
    select_mmpose_prediction,
)


def _write_test_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_process_mmpose_video_skips_inference_for_empty_identity_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    mask = tmp_path / "mask.mp4"
    frames = [np.full((32, 48, 3), 120, dtype=np.uint8) for _ in range(2)]
    empty_masks = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
    _write_test_video(source, frames)
    _write_test_video(mask, empty_masks)

    class Model:
        def __call__(self, _frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pytest.fail("empty identity-risk masks must not run RTMPose")

    monkeypatch.setattr(mmpose_runner, "build_rtmlib_model", lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(mmpose_runner, "make_browser_playable_mp4s", lambda _paths: None)
    pose_path = tmp_path / "pose.jsonl"

    result = mmpose_runner.process_mmpose_video(
        source_video=source,
        mask_video=mask,
        pose_landmarks_path=pose_path,
        raw_mmpose_landmarks_path=tmp_path / "raw.jsonl",
        skeleton_render_path=tmp_path / "skeleton.mp4",
        qa_overlay_path=tmp_path / "qa.mp4",
        features_path=tmp_path / "features.json",
        spec=mmpose_runner.mmpose_model_spec("mmpose_rtmpose_l_384"),
        device="cpu",
    )

    rows = [json.loads(line) for line in pose_path.read_text().splitlines()]
    assert result["quality"]["usable_frames"] == 0
    assert all(row["usable"] is False for row in rows)


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


def test_select_mmpose_prediction_rejects_all_off_mask_predictions() -> None:
    off_target = {
        "keypoints": np.asarray([[10, 10], [15, 18], [18, 22]], dtype=np.float32),
        "keypoint_scores": np.asarray([0.95, 0.9, 0.88], dtype=np.float32),
    }

    selected_index, selected, bbox, mask_iou = select_mmpose_prediction(
        [off_target],
        crop={"x": 0, "y": 0, "width": 100, "height": 100},
        frame_width=100,
        frame_height=100,
        mask_bbox={"x": 0.7, "y": 0.65, "width": 0.2, "height": 0.3},
    )

    assert selected_index is None
    assert selected is None
    assert bbox is None
    assert mask_iou == 0.0


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


@pytest.mark.parametrize(
    ("isolate_qa_overlay", "expected_name", "normalize_qa_overlay"),
    [(False, "qa_overlay.mp4", True), (True, "pose_qa_overlay.mp4", False)],
)
def test_run_mmpose_pose_can_isolate_qa_without_changing_standalone_default(
    tmp_path: Path,
    monkeypatch,
    isolate_qa_overlay: bool,
    expected_name: str,
    normalize_qa_overlay: bool,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = run_dir / "source_segment.mp4"
    mask = run_dir / "runner_mask.mp4"
    source.touch()
    mask.touch()
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "runner-1",
                "paths": {
                    "source_segment": str(source),
                    "runner_mask": str(mask),
                    "qa_overlay": str(run_dir / "qa_overlay.mp4"),
                },
                "stages": {"pose": {"status": "pending"}},
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        mmpose_runner,
        "mmpose_setup_status",
        lambda _model_id: {
            "ready": True,
            "device": "cpu",
            "runtime_backend": "onnxruntime",
            "reasons": [],
        },
    )

    def process(**kwargs: Any) -> dict[str, Any]:
        observed["qa_overlay_path"] = kwargs["qa_overlay_path"]
        observed["normalize_qa_overlay"] = kwargs["normalize_qa_overlay"]
        return {"frame_count": 3, "quality": {}}

    monkeypatch.setattr(mmpose_runner, "process_mmpose_video", process)

    result = mmpose_runner.run_mmpose_pose(
        run_dir=run_dir,
        isolate_qa_overlay=isolate_qa_overlay,
    )

    assert result["status"] == "complete"
    assert observed["qa_overlay_path"] == run_dir / expected_name
    assert observed["normalize_qa_overlay"] is normalize_qa_overlay
    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    expected_key = "pose_qa_overlay" if isolate_qa_overlay else "qa_overlay"
    assert manifest["paths"][expected_key] == str(run_dir / expected_name)
    assert manifest["paths"]["qa_overlay"] == str(run_dir / "qa_overlay.mp4")
    assert manifest["stages"]["renders"][expected_key] == str(run_dir / expected_name)
    if isolate_qa_overlay:
        assert "qa_overlay" not in manifest["stages"]["renders"]


def test_rtmlib_model_cache_keys_detector_runtime_and_device(
    monkeypatch,
) -> None:
    built: list[dict[str, Any]] = []
    fake_rtmlib = ModuleType("rtmlib")

    def custom(**kwargs: Any) -> object:
        built.append(kwargs)
        return object()

    fake_rtmlib.Custom = custom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    monkeypatch.setenv("MMPOSE_USE_DETECTOR", "false")
    mmpose_runner.clear_rtmlib_model_cache()
    spec = mmpose_runner.mmpose_model_spec("mmpose_rtmw_l_384")
    try:
        first = mmpose_runner.build_rtmlib_model(
            spec,
            device="cuda",
            runtime_backend="onnxruntime",
        )
        second = mmpose_runner.build_rtmlib_model(
            spec,
            device="cuda",
            runtime_backend="onnxruntime",
        )
        third = mmpose_runner.build_rtmlib_model(
            spec,
            device="cpu",
            runtime_backend="onnxruntime",
        )
    finally:
        mmpose_runner.clear_rtmlib_model_cache()

    assert first is second
    assert third is not first
    assert len(built) == 2
    assert built[0]["det_class"] is None
    assert built[0]["det"] is None


def test_cached_rtmlib_model_serializes_shared_inference(monkeypatch) -> None:
    state_lock = threading.Lock()
    active = 0
    peak = 0
    fake_rtmlib = ModuleType("rtmlib")

    class Custom:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return np.zeros((1, 1, 2)), np.ones((1, 1))

    fake_rtmlib.Custom = Custom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    mmpose_runner.clear_rtmlib_model_cache()
    try:
        model = mmpose_runner.build_rtmlib_model(
            mmpose_runner.mmpose_model_spec("mmpose_rtmpose_l_384"),
            device="cuda",
            runtime_backend="onnxruntime",
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: model(frame), range(2)))
    finally:
        mmpose_runner.clear_rtmlib_model_cache()

    assert peak == 1


def test_mmpose_setup_status_exposes_detector_switch_default_and_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mmpose_runner.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(mmpose_runner, "_MMPOSE_IMPORT_CHECKED", True)
    monkeypatch.setattr(mmpose_runner, "_MMPOSE_IMPORT_ERROR", None)
    monkeypatch.delenv("MMPOSE_USE_DETECTOR", raising=False)

    default_status = mmpose_runner.mmpose_setup_status("mmpose_rtmw_l_384")
    monkeypatch.setenv("MMPOSE_USE_DETECTOR", "0")
    detector_free_status = mmpose_runner.mmpose_setup_status("mmpose_rtmw_l_384")

    assert default_status["use_detector"] is True
    assert detector_free_status["use_detector"] is False
    assert detector_free_status["env"]["use_detector"] == "MMPOSE_USE_DETECTOR"


def test_rtmlib_model_keeps_yolox_detector_by_default(monkeypatch) -> None:
    built: list[dict[str, Any]] = []
    fake_rtmlib = ModuleType("rtmlib")
    fake_rtmlib.Custom = lambda **kwargs: built.append(kwargs) or object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtmlib", fake_rtmlib)
    monkeypatch.delenv("MMPOSE_USE_DETECTOR", raising=False)
    mmpose_runner.clear_rtmlib_model_cache()
    try:
        mmpose_runner.build_rtmlib_model(
            mmpose_runner.mmpose_model_spec("mmpose_rtmw_l_384"),
            device="cpu",
            runtime_backend="onnxruntime",
        )
    finally:
        mmpose_runner.clear_rtmlib_model_cache()

    assert built[0]["det_class"] == "YOLOX"
    assert built[0]["det"] == mmpose_runner.YOLOX_M_HUMANART_ONNX
