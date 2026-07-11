from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np

from whodoirunlike import densepose_runner
from whodoirunlike.densepose_runner import (
    DensePoseBackend,
    DensePoseFrameOutput,
    DensePoseSetupError,
    _summarize_chart_result,
    apply_densepose_to_frame,
    clear_densepose_backend_cache,
    load_densepose_backend,
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


def test_densepose_backend_serializes_shared_cached_predictor_inference() -> None:
    state_lock = threading.Lock()
    active = 0
    peak = 0

    def predict(_frame: np.ndarray) -> dict[str, Any]:
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1
        return {"instances": []}

    backend = DensePoseBackend(predictor=predict)
    frame = np.full((32, 48, 3), 120, dtype=np.uint8)
    mask = np.ones((32, 48), dtype=np.uint8) * 255
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(
            executor.map(
                lambda frame_index: apply_densepose_to_frame(
                    frame,
                    mask,
                    backend,
                    frame_index=frame_index,
                ),
                range(2),
            )
        )

    assert all(output.row["usable"] is False for output in outputs)
    assert peak == 1


def test_densepose_backend_cache_reuses_only_matching_effective_config(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    built: list[tuple[str, str, float, int, int]] = []

    class FakeConfig:
        def __init__(self) -> None:
            self.MODEL = SimpleNamespace(
                WEIGHTS="",
                DEVICE="",
                ROI_HEADS=SimpleNamespace(SCORE_THRESH_TEST=0.0),
            )
            self.INPUT = SimpleNamespace(MIN_SIZE_TEST=800, MAX_SIZE_TEST=1333)

        def merge_from_file(self, path: str) -> None:
            self.path = path

        def freeze(self) -> None:
            return None

    detectron2_module = ModuleType("detectron2")
    detectron2_module.__path__ = []  # type: ignore[attr-defined]
    config_module = ModuleType("detectron2.config")
    config_module.get_cfg = FakeConfig  # type: ignore[attr-defined]
    engine_module = ModuleType("detectron2.engine")

    def predictor(config: FakeConfig) -> object:
        built.append(
            (
                config.MODEL.WEIGHTS,
                config.MODEL.DEVICE,
                config.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                config.INPUT.MIN_SIZE_TEST,
                config.INPUT.MAX_SIZE_TEST,
            )
        )
        return object()

    engine_module.DefaultPredictor = predictor  # type: ignore[attr-defined]
    densepose_module = ModuleType("densepose")
    densepose_module.add_densepose_config = lambda _config: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "detectron2", detectron2_module)
    monkeypatch.setitem(sys.modules, "detectron2.config", config_module)
    monkeypatch.setitem(sys.modules, "detectron2.engine", engine_module)
    monkeypatch.setitem(sys.modules, "densepose", densepose_module)

    config_path = tmp_path / "densepose.yaml"
    clear_densepose_backend_cache()
    try:
        first = load_densepose_backend(
            config_path=config_path,
            weights_path="model.pkl",
            confidence_threshold=0.7,
            device="cuda",
            input_min_size_test=512,
            input_max_size_test=768,
        )
        second = load_densepose_backend(
            config_path=config_path,
            weights_path="model.pkl",
            confidence_threshold=0.7,
            device="cuda",
            input_min_size_test=512,
            input_max_size_test=768,
        )
        third = load_densepose_backend(
            config_path=config_path,
            weights_path="model.pkl",
            confidence_threshold=0.7,
            device="cuda",
            input_min_size_test=512,
            input_max_size_test=1024,
        )
        default_sized = load_densepose_backend(
            config_path=config_path,
            weights_path="model.pkl",
            confidence_threshold=0.7,
            device="cuda",
        )
    finally:
        clear_densepose_backend_cache()

    assert first is second
    assert third is not first
    assert default_sized.input_min_size_test == 800
    assert default_sized.input_max_size_test == 1333
    assert built == [
        ("model.pkl", "cuda", 0.7, 512, 768),
        ("model.pkl", "cuda", 0.7, 512, 1024),
        ("model.pkl", "cuda", 0.7, 800, 1333),
    ]


def test_target_crop_remaps_detection_centroids_and_label_overlay_to_full_frame(
    monkeypatch: Any,
) -> None:
    frame = np.full((100, 200, 3), 120, dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, 60:120] = 255
    predictor_inputs: list[np.ndarray] = []

    class FakeArray:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def numpy(self) -> np.ndarray:
            return self.value

    class FakeInstances:
        pred_boxes = SimpleNamespace(
            tensor=FakeArray(np.asarray([[12, 15, 52, 55]], dtype=np.float32))
        )
        scores = FakeArray(np.asarray([0.92], dtype=np.float32))
        pred_densepose = object()

        def __len__(self) -> int:
            return 1

    def predict(image: np.ndarray) -> dict[str, Any]:
        predictor_inputs.append(image.copy())
        return {"instances": FakeInstances()}

    chart_result = SimpleNamespace(
        labels=_FakeTensor(np.asarray([[1, 0], [0, 2]], dtype=np.uint8)),
        uv=_FakeTensor(np.zeros((2, 2, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(
        densepose_runner,
        "_chart_result_for_instance",
        lambda _instances, _index: chart_result,
    )

    output = densepose_runner.apply_densepose_to_frame(
        frame,
        mask,
        DensePoseBackend(predictor=predict),
        frame_index=0,
        target_crop_enabled=True,
        target_crop_padding_ratio=0.0,
        target_crop_padding_pixels=10,
    )

    assert predictor_inputs[0].shape == (60, 80, 3)
    assert np.all(predictor_inputs[0][:10] == 0)
    assert output.row["bbox"] == [62, 35, 40, 40]
    assert output.row["part_centroids"] == {
        "1": {"bbox_x": 0.25, "bbox_y": 0.25, "x": 0.36, "y": 0.45},
        "2": {"bbox_x": 0.75, "bbox_y": 0.75, "x": 0.46, "y": 0.65},
    }
    assert output.row["inference_input"] == {
        "target_crop_enabled": True,
        "crop_bbox": [50, 20, 80, 60],
        "width": 80,
        "height": 60,
    }
    overlay = densepose_runner._draw_densepose_overlay(
        np.zeros_like(frame),
        mask,
        output.row,
        labels=output.labels,
    )
    assert overlay[45, 72].any()
    assert not overlay[10, 10].any()


def test_apply_densepose_skips_predictor_when_runner_mask_is_empty() -> None:
    predictor_calls = 0

    def predict(_image: np.ndarray) -> dict[str, Any]:
        nonlocal predictor_calls
        predictor_calls += 1
        return {"instances": []}

    output = densepose_runner.apply_densepose_to_frame(
        np.zeros((40, 60, 3), dtype=np.uint8),
        np.zeros((40, 60), dtype=np.uint8),
        DensePoseBackend(predictor=predict),
        frame_index=0,
        target_crop_enabled=True,
    )

    assert predictor_calls == 0
    assert output.row == {
        "usable": False,
        "drop_reason": "runner_mask_empty",
        "inference_input": {
            "target_crop_enabled": True,
            "crop_bbox": None,
            "width": 0,
            "height": 0,
        },
    }


def test_apply_densepose_default_still_infers_on_full_masked_frame() -> None:
    observed_inputs: list[np.ndarray] = []

    class FakeArray:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def numpy(self) -> np.ndarray:
            return self.value

    class FakeInstances:
        pred_boxes = SimpleNamespace(
            tensor=FakeArray(np.asarray([[10, 8, 30, 28]], dtype=np.float32))
        )
        scores = FakeArray(np.asarray([0.9], dtype=np.float32))

        def __len__(self) -> int:
            return 1

    def predict(image: np.ndarray) -> dict[str, Any]:
        observed_inputs.append(image.copy())
        return {"instances": FakeInstances()}

    frame = np.full((40, 60, 3), 100, dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[5:35, 5:35] = 255

    output = densepose_runner.apply_densepose_to_frame(
        frame,
        mask,
        DensePoseBackend(predictor=predict),
        frame_index=0,
    )

    assert observed_inputs[0].shape == frame.shape
    assert np.all(observed_inputs[0][0, 0] == 0)
    assert np.all(observed_inputs[0][10, 10] == 100)
    assert output.row["bbox"] == [10, 8, 20, 20]
    assert output.row["inference_input"] == {
        "target_crop_enabled": False,
        "crop_bbox": [0, 0, 60, 40],
        "width": 60,
        "height": 40,
    }


def test_run_densepose_exposes_effective_resize_and_crop_measurements(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _make_run_dir(tmp_path, frame_count=1)
    loader_kwargs: list[dict[str, Any]] = []
    apply_kwargs: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []

    def load_backend(**kwargs: Any) -> DensePoseBackend:
        loader_kwargs.append(kwargs)
        return DensePoseBackend(
            predictor=object(),
            input_min_size_test=512,
            input_max_size_test=768,
        )

    def apply_frame(
        _frame: np.ndarray,
        _mask: np.ndarray,
        _backend: DensePoseBackend,
        **kwargs: Any,
    ) -> DensePoseFrameOutput:
        apply_kwargs.append(kwargs)
        return DensePoseFrameOutput(
            row={
                "usable": True,
                "bbox": [12, 8, 18, 16],
                "drop_reason": None,
                "inference_input": {
                    "target_crop_enabled": True,
                    "crop_bbox": [4, 2, 40, 28],
                    "width": 40,
                    "height": 28,
                },
            }
        )

    monkeypatch.setattr(densepose_runner, "load_densepose_backend", load_backend)
    monkeypatch.setattr(densepose_runner, "apply_densepose_to_frame", apply_frame)

    result = run_densepose(
        run_dir=run_dir,
        config_path=Path("cfg.yaml"),
        weights_path="weights.pkl",
        write_qa_overlay=False,
        input_min_size_test=512,
        input_max_size_test=768,
        target_crop_enabled=True,
        target_crop_padding_ratio=0.25,
        target_crop_padding_pixels=12,
        progress_callback=progress.append,
    )

    assert loader_kwargs[0]["input_min_size_test"] == 512
    assert loader_kwargs[0]["input_max_size_test"] == 768
    assert apply_kwargs[0] == {
        "frame_index": 0,
        "target_crop_enabled": True,
        "target_crop_padding_ratio": 0.25,
        "target_crop_padding_pixels": 12,
    }
    assert result["inference_settings"] == {
        "target_crop_enabled": True,
        "target_crop_padding_ratio": 0.25,
        "target_crop_padding_pixels": 12,
        "input_min_size_test": 512,
        "input_max_size_test": 768,
    }
    row = _read_jsonl(run_dir / "densepose.jsonl")[0]
    assert row["inference_input"]["width"] == 40
    assert row["inference_input"]["height"] == 28
    running_progress = [item for item in progress if item["phase"] == "running_densepose"]
    assert running_progress[-1]["inference_input"]["width"] == 40
    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["densepose"]["inference_settings"] == result[
        "inference_settings"
    ]


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
