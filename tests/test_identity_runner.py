from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from whodoirunlike import identity_runner, inline_segmentation
from whodoirunlike.identity_runner import (
    canonical_identity_backend,
    prompt_initial_box,
    run_identity_tracking,
)


def test_yolo_model_cache_reuses_and_serializes_mutable_predictor(monkeypatch) -> None:
    state_lock = threading.Lock()
    active = 0
    peak = 0
    builds = 0
    fake_ultralytics = ModuleType("ultralytics")

    class Model:
        task = "segment"

        def predict(self, *_args: object, **_kwargs: object) -> list[object]:
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return []

    def yolo(_model: str) -> Model:
        nonlocal builds
        builds += 1
        return Model()

    fake_ultralytics.YOLO = yolo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    identity_runner.clear_yolo_model_cache()
    try:
        first = identity_runner._load_yolo_model("yolo26n-seg.pt")
        second = identity_runner._load_yolo_model("yolo26n-seg.pt")
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: first.predict(np.zeros((8, 8, 3))), range(2)))
    finally:
        identity_runner.clear_yolo_model_cache()

    assert first is second
    assert builds == 1
    assert peak == 1
    assert first.task == "segment"


def test_inline_yolo_requests_masks_in_original_frame_coordinates() -> None:
    calls: list[dict[str, object]] = []

    class Result:
        boxes = None

    class Model:
        def predict(self, _frame: np.ndarray, **kwargs: object) -> list[Result]:
            calls.append(kwargs)
            return [Result()]

    frame = np.zeros((54, 96, 3), dtype=np.uint8)
    identity_runner._run_yolo_person_inference(
        Model(),
        frame,
        device="cpu",
        confidence=0.25,
        iou=0.7,
        imgsz=96,
        mask_threshold=0.5,
        include_masks=True,
    )
    identity_runner._run_yolo_person_inference(
        Model(),
        frame,
        device="cpu",
        confidence=0.25,
        iou=0.7,
        imgsz=96,
        mask_threshold=0.5,
        include_masks=False,
    )

    assert calls[0]["retina_masks"] is True
    assert "retina_masks" not in calls[1]


def write_identity_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    for index in range(18):
        frame = np.full((64, 96, 3), 28, dtype=np.uint8)
        x = 22 + index
        cv2.rectangle(frame, (x, 18), (x + 14, 48), (48, 180, 220), -1)
        cv2.circle(frame, (x + 7, 52), 3, (48, 180, 220), -1)
        writer.write(frame)
    writer.release()


def write_switch_identity_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    for index in range(18):
        frame = np.full((64, 96, 3), 28, dtype=np.uint8)
        target_x = 22 + index
        cv2.rectangle(frame, (target_x, 18), (target_x + 14, 48), (48, 180, 220), -1)
        cv2.circle(frame, (target_x + 7, 52), 3, (48, 180, 220), -1)
        cv2.rectangle(frame, (60, 18), (74, 48), (40, 220, 80), -1)
        cv2.circle(frame, (67, 52), 3, (40, 220, 80), -1)
        writer.write(frame)
    writer.release()


def write_lookalike_jump_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    for index in range(18):
        frame = np.full((64, 96, 3), 28, dtype=np.uint8)
        target_x = 22 + index
        target_color = (48, 180, 220)
        cv2.rectangle(frame, (target_x, 18), (target_x + 14, 48), target_color, -1)
        cv2.circle(frame, (target_x + 7, 52), 3, target_color, -1)
        cv2.rectangle(frame, (1, 18), (15, 48), (48, 180, 220), -1)
        cv2.circle(frame, (8, 52), 3, (48, 180, 220), -1)
        writer.write(frame)
    writer.release()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_identity_run(run_dir: Path) -> None:
    video_path = run_dir / "source_segment.mp4"
    write_identity_video(video_path)
    prompt_path = run_dir / "person_prompt.json"
    track_seed_path = run_dir / "track_seed.json"
    write_json(
        prompt_path,
        {
            "version": 1,
            "candidate_id": "identity-clip",
            "frame": {"frame_index": 9, "width": 96, "height": 64},
            "selection": {
                "type": "box",
                "positive_points": [{"x": 0.395833, "y": 0.515625}],
                "negative_points": [],
                "box": {"x": 0.322917, "y": 0.28125, "width": 0.145833, "height": 0.46875},
                "mask_path": None,
            },
        },
    )
    write_json(
        track_seed_path,
        {
            "version": 1,
            "candidate_id": "identity-clip",
            "status": "pending_detector_tracker",
            "reid": {"cosine_accept": 0.65, "cosine_recover": 0.58},
        },
    )
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": "identity-clip",
            "runner_name": "Identity Runner",
            "paths": {
                "source_segment": str(video_path),
                "person_prompt": str(prompt_path),
                "track_seed": str(track_seed_path),
                "tracklets": str(run_dir / "tracklets.parquet"),
                "tracklets_jsonl": str(run_dir / "tracklets.jsonl"),
                "reid": str(run_dir / "reid.parquet"),
                "reid_jsonl": str(run_dir / "reid.jsonl"),
                "qc_metrics": str(run_dir / "qc_metrics.json"),
            },
            "stages": {
                "detector_tracker": {"status": "pending_run"},
                "whole_runner_mask": {"status": "pending_tracker"},
            },
        },
    )


def test_prompt_initial_box_uses_positive_point_when_box_is_absent() -> None:
    prompt = {
        "selection": {
            "type": "point",
            "positive_points": [{"x": 0.5, "y": 0.5}],
            "negative_points": [],
            "box": None,
        }
    }

    x, y, width, height = prompt_initial_box(prompt, 100, 80)

    assert width == 16
    assert height == 34
    assert x == 42
    assert y == 25


def test_run_identity_tracking_writes_parquet_and_updates_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    result = run_identity_tracking(run_dir=run_dir, backend="prompt_template_tracker_v1")

    assert result["status"] == "complete"
    assert result["backend"] == "prompt_template_tracker_v1"
    assert result["frame_count"] == 18
    assert result["metrics"]["target_identity_stability_rate"] >= 0.75
    assert (run_dir / "tracklets.parquet").exists()
    assert (run_dir / "reid.parquet").exists()
    assert (run_dir / "tracklets.jsonl").exists()
    assert (run_dir / "qc_metrics.json").exists()

    track_table = pq.read_table(run_dir / "tracklets.parquet")
    reid_table = pq.read_table(run_dir / "reid.parquet")
    assert track_table.num_rows == 18
    assert reid_table.num_rows == 18

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    track_seed = json.loads((run_dir / "track_seed.json").read_text(encoding="utf-8"))
    qc_metrics = json.loads((run_dir / "qc_metrics.json").read_text(encoding="utf-8"))

    assert manifest["stages"]["detector_tracker"]["status"] == "complete"
    assert manifest["stages"]["whole_runner_mask"]["status"] == "pending_run"
    assert track_seed["target_track_id"] == 1
    assert track_seed["status"] == "complete"
    assert qc_metrics["identity"]["frame_count"] == 18


def test_template_identity_tracking_uses_configured_jsonl_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    configured_dir = tmp_path / "custom-layout" / "identity"
    configured_tracklets = configured_dir / "runner-tracklets.jsonl"
    configured_reid = configured_dir / "runner-reid.jsonl"
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paths"]["tracklets_jsonl"] = str(configured_tracklets)
    manifest["paths"]["reid_jsonl"] = str(configured_reid)
    write_json(manifest_path, manifest)

    default_tracklets = run_dir / "tracklets.jsonl"
    default_reid = run_dir / "reid.jsonl"
    default_tracklets.write_text("poisoned default tracklets\n", encoding="utf-8")
    default_reid.write_text("poisoned default reid\n", encoding="utf-8")

    result = run_identity_tracking(run_dir=run_dir, backend="prompt_template_tracker_v1")

    assert result["tracklets_jsonl_path"] == str(configured_tracklets)
    assert result["reid_jsonl_path"] == str(configured_reid)
    assert len(configured_tracklets.read_text(encoding="utf-8").splitlines()) == 18
    assert len(configured_reid.read_text(encoding="utf-8").splitlines()) == 18
    assert default_tracklets.read_text(encoding="utf-8") == "poisoned default tracklets\n"
    assert default_reid.read_text(encoding="utf-8") == "poisoned default reid\n"
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["paths"]["tracklets_jsonl"] == str(configured_tracklets)
    assert updated_manifest["paths"]["reid_jsonl"] == str(configured_reid)
    assert updated_manifest["stages"]["detector_tracker"]["tracklets_jsonl"] == str(
        configured_tracklets
    )
    assert updated_manifest["stages"]["detector_tracker"]["reid_jsonl"] == str(configured_reid)


def test_canonical_identity_backend_accepts_plan_aliases() -> None:
    assert canonical_identity_backend("botsort") == "boxmot_botsort"
    assert canonical_identity_backend("deep-oc-sort") == "boxmot_deepocsort"
    assert canonical_identity_backend("template") == "prompt_template_tracker_v1"


def test_inline_segmentation_requires_boxmot_identity_backend(tmp_path: Path) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    with pytest.raises(ValueError, match="requires a BoxMOT identity backend"):
        run_identity_tracking(
            run_dir=run_dir,
            backend="prompt_template_tracker_v1",
            inline_segmentation=True,
        )

    assert not (run_dir / "runner_mask.mp4").exists()


class FakeBoxes:
    def __init__(self, frame_index: int) -> None:
        x = 22 + frame_index
        self.xyxy = np.array([[x, 18, x + 14, 48]], dtype=np.float32)
        self.conf = np.array([0.91], dtype=np.float32)
        self.cls = np.array([0], dtype=np.float32)


class FakeResult:
    def __init__(self, frame_index: int) -> None:
        self.boxes = FakeBoxes(frame_index)


class FakeYolo:
    def __init__(self) -> None:
        self.frame_index = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[FakeResult]:
        result = FakeResult(self.frame_index)
        self.frame_index += 1
        return [result]


class FakeBoxmotTracker:
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        det = detections[0]
        return np.array([[det[0], det[1], det[2], det[3], 7, det[4], 0, 0]], dtype=np.float32)


class UnifiedSegmentationBoxes:
    def __init__(self, frame_index: int) -> None:
        target_x = 22 + frame_index
        self.xyxy = np.array(
            [[target_x, 18, target_x + 14, 48], [60, 18, 74, 48]],
            dtype=np.float32,
        )
        self.conf = np.array([0.94, 0.93], dtype=np.float32)
        self.cls = np.array([0, 0], dtype=np.float32)


class UnifiedSegmentationMasks:
    def __init__(self, frame_index: int) -> None:
        target_x = 22 + frame_index
        data = np.zeros((2, 64, 96), dtype=np.float32)
        data[0, 18:49, target_x : target_x + 15] = 1.0
        data[1, 18:49, 60:75] = 1.0
        self.data = data


class UnifiedSegmentationResult:
    def __init__(self, frame_index: int) -> None:
        self.boxes = UnifiedSegmentationBoxes(frame_index)
        self.masks = UnifiedSegmentationMasks(frame_index)


class UnifiedSegmentationYolo:
    def __init__(self) -> None:
        self.frame_index = 0
        self.predict_calls = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[UnifiedSegmentationResult]:
        result = UnifiedSegmentationResult(self.frame_index)
        self.frame_index += 1
        self.predict_calls += 1
        return [result]


class UnifiedSegmentationTracker:
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        rows = [
            [*detections[0, :4], 7, detections[0, 4], 0, 0],
            [*detections[1, :4], 8, detections[1, 4], 0, 1],
        ]
        return np.asarray(rows, dtype=np.float32)


class ExplosiveMaskTensor:
    @property
    def data(self) -> np.ndarray:
        raise AssertionError("detector-only mode must not transfer segmentation tensors")


class DetectorOnlyResult:
    def __init__(self, frame_index: int) -> None:
        self.boxes = FakeBoxes(frame_index)
        self.masks = ExplosiveMaskTensor()


class DetectorOnlyYolo:
    def __init__(self) -> None:
        self.frame_index = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[DetectorOnlyResult]:
        result = DetectorOnlyResult(self.frame_index)
        self.frame_index += 1
        return [result]


def test_boxmot_detector_only_mode_does_not_read_segmentation_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: DetectorOnlyYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: FakeBoxmotTracker(),
    )

    result = run_identity_tracking(run_dir=run_dir, backend="boxmot_botsort", device="cpu")

    assert result["status"] == "complete"
    assert result["inline_mask"] is None
    assert not (run_dir / "runner_mask.mp4").exists()


def test_inline_segmentation_fails_when_model_produces_no_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(identity_runner, "_load_yolo_model", lambda detector_model: FakeYolo())
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: FakeBoxmotTracker(),
    )

    with pytest.raises(RuntimeError, match="did not produce any person segmentation masks"):
        run_identity_tracking(
            run_dir=run_dir,
            backend="boxmot_botsort",
            detector_model="yolo11n.pt",
            device="cpu",
            inline_segmentation=True,
        )

    assert not (run_dir / "runner_mask.mp4").exists()


class OffTrackSegmentationYolo(UnifiedSegmentationYolo):
    def predict(self, frame: np.ndarray, **kwargs: object) -> list[UnifiedSegmentationResult]:
        results = super().predict(frame, **kwargs)
        if self.frame_index - 1 == 4:
            target_mask = results[0].masks.data[0]
            target_mask.fill(0)
            target_mask[0:10, 0:10] = 1.0
        return results


class SeveralOffTrackSegmentationYolo(UnifiedSegmentationYolo):
    def predict(self, frame: np.ndarray, **kwargs: object) -> list[UnifiedSegmentationResult]:
        results = super().predict(frame, **kwargs)
        if self.frame_index - 1 in {4, 5, 6}:
            target_mask = results[0].masks.data[0]
            target_mask.fill(0)
            target_mask[0:10, 0:10] = 1.0
        return results


class AreaJumpSegmentationYolo(UnifiedSegmentationYolo):
    def predict(self, frame: np.ndarray, **kwargs: object) -> list[UnifiedSegmentationResult]:
        results = super().predict(frame, **kwargs)
        if self.frame_index - 1 == 4:
            target_x = 22 + (self.frame_index - 1)
            target_mask = results[0].masks.data[0]
            target_mask.fill(0)
            target_mask[28:31, target_x + 5 : target_x + 8] = 1.0
        return results


def test_boxmot_identity_tracking_emits_selected_runner_mask_from_same_yolo_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_switch_identity_video(run_dir / "source_segment.mp4")
    detector = UnifiedSegmentationYolo()
    progress_phases: list[str] = []

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(identity_runner, "_load_yolo_model", lambda detector_model: detector)
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: UnifiedSegmentationTracker(),
    )

    result = run_identity_tracking(
        run_dir=run_dir,
        backend="boxmot_botsort",
        detector_model="yolo26n-seg.pt",
        device="cpu",
        inline_segmentation=True,
        inline_mask_dilation_pixels=2,
        inline_mask_temporal_reset_gap_frames=7,
        progress_callback=lambda payload: progress_phases.append(str(payload["phase"])),
    )

    assert detector.predict_calls == 18
    assert result["metrics"]["target_track_id"] == 7
    assert result["inline_mask"]["status"] == "complete"
    assert result["inline_mask"]["summary"]["associated_frames"] == 18
    assert result["inline_mask"]["summary"]["track_box_fallback_frames"] == 0
    assert result["inline_mask"]["summary"]["dilation_pixels"] == 2
    assert result["inline_mask"]["safety"]["temporal_reset_gap_frames"] == 7
    assert result["inline_mask"]["timing"]["render_seconds"] >= 0
    assert result["inline_mask"]["timing"]["encode_seconds"] >= 0
    metadata_first = json.loads(
        Path(result["inline_mask"]["metadata"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert metadata_first["mask_area"] > 465
    assert metadata_first["usable"] is True
    assert metadata_first["centroid"] is not None
    assert metadata_first["centroid_delta_px"] is None

    mask_summary = result["inline_mask"]["summary"]
    assert result["inline_mask"]["mask_summary"] == mask_summary
    assert mask_summary["fps"] == 10.0
    assert mask_summary["width"] == 96
    assert mask_summary["height"] == 64
    assert mask_summary["output_path"] == result["inline_mask"]["masks_jsonl"]
    assert mask_summary["mean_mask_churn"] == pytest.approx(
        1.0 - mask_summary["mean_temporal_iou"],
        abs=1e-6,
    )

    mask_capture = cv2.VideoCapture(result["inline_mask"]["runner_mask"])
    ok, mask_frame = mask_capture.read()
    mask_capture.release()
    assert ok
    mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    assert int(mask[30, 28]) > 20
    assert int(mask[30, 67]) < 20

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["whole_runner_mask"]["status"] == "complete"
    assert manifest["stages"]["whole_runner_mask"]["backend"] == "yolo26n_seg_inline"
    assert manifest["stages"]["whole_runner_mask"]["mask_summary"] == mask_summary
    assert manifest["paths"]["runner_mask"] == result["inline_mask"]["runner_mask"]
    assert manifest["paths"]["masks_jsonl"] == result["inline_mask"]["masks_jsonl"]
    phase_order = [
        "detect_track",
        "postprocessing",
        "rendering_inline_mask",
        "encoding_inline_mask",
        "writing_inline_mask_outputs",
        "completed",
    ]
    assert [
        next(index for index, phase in enumerate(progress_phases) if phase == expected)
        for expected in phase_order
    ] == sorted(
        next(index for index, phase in enumerate(progress_phases) if phase == expected)
        for expected in phase_order
    )


def test_inline_segmentation_can_defer_browser_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_switch_identity_video(run_dir / "source_segment.mp4")
    progress_phases: list[str] = []

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: UnifiedSegmentationYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: UnifiedSegmentationTracker(),
    )
    monkeypatch.setattr(
        inline_segmentation,
        "make_browser_playable_mp4s",
        lambda paths: pytest.fail(f"browser encoding unexpectedly ran for {list(paths)}"),
    )

    result = run_identity_tracking(
        run_dir=run_dir,
        backend="boxmot_botsort",
        detector_model="yolo26n-seg.pt",
        device="cpu",
        inline_segmentation=True,
        inline_mask_defer_browser_encoding=True,
        progress_callback=lambda payload: progress_phases.append(str(payload["phase"])),
    )

    inline_mask = result["inline_mask"]
    expected_paths = [
        inline_mask["runner_mask"],
        inline_mask["masked_runner"],
        inline_mask["qa_overlay"],
    ]
    assert inline_mask["deferred_browser_encoding"] == {
        "required": True,
        "paths": expected_paths,
    }
    assert all(Path(path).is_file() for path in expected_paths)
    assert "encoding_inline_mask" not in progress_phases

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["whole_runner_mask"]["deferred_browser_encoding"] == {
        "required": True,
        "paths": expected_paths,
    }


def test_inline_segmentation_blanks_severe_off_track_mask_and_recommends_sam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_switch_identity_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: OffTrackSegmentationYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: UnifiedSegmentationTracker(),
    )

    result = run_identity_tracking(
        run_dir=run_dir,
        backend="boxmot_botsort",
        detector_model="yolo26n-seg.pt",
        device="cpu",
        inline_segmentation=True,
        inline_mask_dilation_pixels=0,
    )

    inline_mask = result["inline_mask"]
    fallback = inline_mask["fallback"]
    assert fallback == {
        "used": False,
        "frame_indexes": [],
        "reasons": {"segmentation_mask_off_track": 1},
        "sam_fallback_recommended": True,
    }
    assert inline_mask["summary"]["track_box_fallback_frames"] == 0
    assert inline_mask["summary"]["rejected_segmentation_frames"] == 1
    assert inline_mask["summary"]["severe_rejection_frames"] == 1
    assert inline_mask["summary"]["sam_fallback_recommended"] is True
    metadata_rows = [
        json.loads(line)
        for line in Path(inline_mask["metadata"]).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata_rows[4]["source"] == "blank"
    assert metadata_rows[4]["mask_area"] == 0
    assert metadata_rows[4]["usable"] is False
    assert metadata_rows[4]["fallback_reason"] == "segmentation_mask_off_track"


def test_inline_segmentation_recommends_sam_when_rejected_masks_are_not_box_filled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_switch_identity_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: SeveralOffTrackSegmentationYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: UnifiedSegmentationTracker(),
    )

    result = run_identity_tracking(
        run_dir=run_dir,
        backend="boxmot_botsort",
        detector_model="yolo26n-seg.pt",
        device="cpu",
        inline_segmentation=True,
        inline_mask_dilation_pixels=0,
        inline_mask_fallback_to_track_box=False,
    )

    summary = result["inline_mask"]["summary"]
    assert summary["track_box_fallback_frames"] == 0
    assert summary["rejected_segmentation_frames"] == 3
    assert summary["degraded_mask_frames"] == 3
    assert summary["sam_fallback_recommended"] is True


def test_inline_segmentation_rejects_temporally_implausible_area_jump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_switch_identity_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: AreaJumpSegmentationYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: UnifiedSegmentationTracker(),
    )

    result = run_identity_tracking(
        run_dir=run_dir,
        backend="boxmot_botsort",
        detector_model="yolo26n-seg.pt",
        device="cpu",
        inline_segmentation=True,
        inline_mask_dilation_pixels=0,
    )

    inline_mask = result["inline_mask"]
    assert inline_mask["fallback"]["frame_indexes"] == [4]
    assert inline_mask["fallback"]["reasons"] == {"segmentation_mask_area_jump": 1}
    assert inline_mask["safety"]["temporal_rejection_frame_indexes"] == [4]


def test_inline_segmentation_resets_temporal_baseline_after_configured_gap(
    tmp_path: Path,
) -> None:
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(6)]

    def candidate(x: int) -> dict[str, object]:
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[18:49, x : x + 15] = 1
        ok, encoded = cv2.imencode(".png", mask * 255)
        assert ok
        return {
            "box": (x, 18, 15, 31),
            "_identity_state": "usable",
            "_inline_mask_png": encoded.tobytes(),
            "_inline_mask_association_method": "tracker_detection_index",
            "_inline_mask_detection_index": 0,
            "_inline_mask_track_box_iou": 1.0,
        }

    result = inline_segmentation.write_selected_runner_mask_artifacts(
        frames=frames,
        fps=30.0,
        target_candidates={0: candidate(4), 5: candidate(68)},
        runner_mask_path=tmp_path / "runner_mask.mp4",
        masked_runner_path=tmp_path / "masked_runner.mp4",
        qa_overlay_path=tmp_path / "qa_overlay.mp4",
        metadata_path=tmp_path / "runner_mask_metadata.jsonl",
        masks_jsonl_path=tmp_path / "masks.jsonl",
        model="yolo26n-seg.pt",
        config=inline_segmentation.InlineMaskConfig(
            dilation_pixels=0,
            temporal_reset_gap_frames=2,
        ),
        browser_playable=False,
    )

    metadata_rows = [
        json.loads(line)
        for line in Path(result["metadata"]).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata_rows[5]["source"] == "yolo_segmentation"
    assert metadata_rows[5]["fallback_reason"] is None
    assert metadata_rows[5]["temporal_frame_gap"] == 5
    assert result["summary"]["associated_frames"] == 2
    assert result["safety"]["temporal_baseline_reset_frame_indexes"] == [5]


def test_inline_segmentation_recommends_sam_for_material_missing_target_gap(
    tmp_path: Path,
) -> None:
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(3)]

    def candidate(x: int) -> dict[str, object]:
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[18:49, x : x + 15] = 1
        ok, encoded = cv2.imencode(".png", mask * 255)
        assert ok
        return {
            "box": (x, 18, 15, 31),
            "_identity_state": "usable",
            "_inline_mask_png": encoded.tobytes(),
            "_inline_mask_track_box_iou": 1.0,
        }

    result = inline_segmentation.write_selected_runner_mask_artifacts(
        frames=frames,
        fps=30.0,
        target_candidates={0: candidate(4), 2: candidate(6)},
        runner_mask_path=tmp_path / "runner_mask.mp4",
        masked_runner_path=tmp_path / "masked_runner.mp4",
        qa_overlay_path=tmp_path / "qa_overlay.mp4",
        metadata_path=tmp_path / "runner_mask_metadata.jsonl",
        masks_jsonl_path=tmp_path / "masks.jsonl",
        model="yolo26n-seg.pt",
        config=inline_segmentation.InlineMaskConfig(dilation_pixels=0),
        browser_playable=False,
    )

    summary = result["summary"]
    assert summary["missing_target_frames"] == 1
    assert summary["missing_target_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert summary["maximum_consecutive_missing_target_frames"] == 1
    assert summary["sam_fallback_recommended"] is True


def test_inline_segmentation_scales_temporal_motion_allowance_with_frame_gap(
    tmp_path: Path,
) -> None:
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(3)]

    def candidate(x: int) -> dict[str, object]:
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[18:49, x : x + 15] = 1
        ok, encoded = cv2.imencode(".png", mask * 255)
        assert ok
        return {
            "box": (x, 18, 15, 31),
            "_identity_state": "usable",
            "_inline_mask_png": encoded.tobytes(),
            "_inline_mask_track_box_iou": 1.0,
        }

    result = inline_segmentation.write_selected_runner_mask_artifacts(
        frames=frames,
        fps=30.0,
        target_candidates={0: candidate(4), 2: candidate(44)},
        runner_mask_path=tmp_path / "runner_mask.mp4",
        masked_runner_path=tmp_path / "masked_runner.mp4",
        qa_overlay_path=tmp_path / "qa_overlay.mp4",
        metadata_path=tmp_path / "runner_mask_metadata.jsonl",
        masks_jsonl_path=tmp_path / "masks.jsonl",
        model="yolo26n-seg.pt",
        config=inline_segmentation.InlineMaskConfig(
            dilation_pixels=0,
            temporal_reset_gap_frames=3,
        ),
        browser_playable=False,
    )

    metadata_rows = [
        json.loads(line)
        for line in Path(result["metadata"]).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata_rows[2]["source"] == "yolo_segmentation"
    assert metadata_rows[2]["fallback_reason"] is None
    assert metadata_rows[2]["temporal_frame_gap"] == 2
    assert result["safety"]["temporal_rejection_frame_indexes"] == []
    assert result["safety"]["temporal_baseline_reset_frame_indexes"] == []


def test_inline_segmentation_scales_area_change_allowance_with_frame_gap(
    tmp_path: Path,
) -> None:
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(3)]

    def candidate(size: int) -> dict[str, object]:
        x = 48 - size // 2
        y = 32 - size // 2
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[y : y + size, x : x + size] = 1
        ok, encoded = cv2.imencode(".png", mask * 255)
        assert ok
        return {
            "box": (x, y, size, size),
            "_identity_state": "usable",
            "_inline_mask_png": encoded.tobytes(),
            "_inline_mask_track_box_iou": 1.0,
        }

    result = inline_segmentation.write_selected_runner_mask_artifacts(
        frames=frames,
        fps=30.0,
        target_candidates={0: candidate(5), 2: candidate(10)},
        runner_mask_path=tmp_path / "runner_mask.mp4",
        masked_runner_path=tmp_path / "masked_runner.mp4",
        qa_overlay_path=tmp_path / "qa_overlay.mp4",
        metadata_path=tmp_path / "runner_mask_metadata.jsonl",
        masks_jsonl_path=tmp_path / "masks.jsonl",
        model="yolo26n-seg.pt",
        config=inline_segmentation.InlineMaskConfig(
            dilation_pixels=0,
            temporal_reset_gap_frames=3,
        ),
        browser_playable=False,
    )

    metadata_rows = [
        json.loads(line)
        for line in Path(result["metadata"]).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata_rows[2]["source"] == "yolo_segmentation"
    assert metadata_rows[2]["fallback_reason"] is None
    assert metadata_rows[2]["area_change_ratio"] == 4.0
    assert metadata_rows[2]["maximum_area_change_ratio"] == 6.0


def test_inline_segmentation_blanks_severe_adjacent_centroid_jump(
    tmp_path: Path,
) -> None:
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(2)]

    def candidate(x: int) -> dict[str, object]:
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[18:49, x : x + 15] = 1
        ok, encoded = cv2.imencode(".png", mask * 255)
        assert ok
        return {
            "box": (x, 18, 15, 31),
            "_identity_state": "usable",
            "_inline_mask_png": encoded.tobytes(),
            "_inline_mask_track_box_iou": 1.0,
        }

    result = inline_segmentation.write_selected_runner_mask_artifacts(
        frames=frames,
        fps=30.0,
        target_candidates={0: candidate(4), 1: candidate(68)},
        runner_mask_path=tmp_path / "runner_mask.mp4",
        masked_runner_path=tmp_path / "masked_runner.mp4",
        qa_overlay_path=tmp_path / "qa_overlay.mp4",
        metadata_path=tmp_path / "runner_mask_metadata.jsonl",
        masks_jsonl_path=tmp_path / "masks.jsonl",
        model="yolo26n-seg.pt",
        config=inline_segmentation.InlineMaskConfig(dilation_pixels=0),
        browser_playable=False,
    )

    metadata_rows = [
        json.loads(line)
        for line in Path(result["metadata"]).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata_rows[1]["source"] == "blank"
    assert metadata_rows[1]["fallback_reason"] == "segmentation_mask_centroid_jump"
    assert metadata_rows[1]["usable"] is False
    assert result["summary"]["severe_rejection_frames"] == 1
    assert result["summary"]["track_box_fallback_frames"] == 0
    assert result["summary"]["sam_fallback_recommended"] is True
    assert result["safety"]["temporal_rejection_frame_indexes"] == [1]


class SwitchFakeBoxes:
    def __init__(self, frame_index: int) -> None:
        target_x = 22 + frame_index
        self.xyxy = np.array(
            [
                [target_x, 18, target_x + 14, 48],
                [60, 18, 74, 48],
            ],
            dtype=np.float32,
        )
        self.conf = np.array([0.94, 0.93], dtype=np.float32)
        self.cls = np.array([0, 0], dtype=np.float32)


class SwitchFakeResult:
    def __init__(self, frame_index: int) -> None:
        self.boxes = SwitchFakeBoxes(frame_index)


class SwitchFakeYolo:
    def __init__(self) -> None:
        self.frame_index = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[SwitchFakeResult]:
        result = SwitchFakeResult(self.frame_index)
        self.frame_index += 1
        return [result]


class SwitchFakeBoxmotTracker:
    def __init__(self) -> None:
        self.frame_index = 0

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        frame_index = self.frame_index
        self.frame_index += 1
        target = detections[0]
        distractor = detections[1]
        if frame_index <= 9:
            rows = [
                [*target[:4], 3, target[4], 0, 0],
                [*distractor[:4], 8, distractor[4], 0, 0],
            ]
        else:
            rows = [
                [*target[:4], 7, target[4], 0, 0],
                [*distractor[:4], 3, distractor[4], 0, 0],
            ]
        return np.array(rows, dtype=np.float32)


class LookalikeFakeBoxes:
    def __init__(self, frame_index: int) -> None:
        target_x = 22 + frame_index
        self.xyxy = np.array(
            [
                [target_x, 18, target_x + 14, 48],
                [1, 18, 15, 48],
            ],
            dtype=np.float32,
        )
        self.conf = np.array([0.91, 0.96], dtype=np.float32)
        self.cls = np.array([0, 0], dtype=np.float32)


class LookalikeFakeResult:
    def __init__(self, frame_index: int) -> None:
        self.boxes = LookalikeFakeBoxes(frame_index)


class LookalikeFakeYolo:
    def __init__(self) -> None:
        self.frame_index = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[LookalikeFakeResult]:
        result = LookalikeFakeResult(self.frame_index)
        self.frame_index += 1
        return [result]


class LookalikeFakeBoxmotTracker:
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        target = detections[0]
        lookalike = detections[1]
        rows = [
            [*target[:4], 3, target[4], 0, 0],
            [*lookalike[:4], 6, lookalike[4], 0, 0],
        ]
        return np.array(rows, dtype=np.float32)


class NearLookalikeGapFakeBoxes:
    def __init__(self, frame_index: int) -> None:
        target_x = 22 + frame_index
        if frame_index <= 8:
            self.xyxy = np.array([[21, 18, 35, 48]], dtype=np.float32)
            self.conf = np.array([0.96], dtype=np.float32)
            self.cls = np.array([0], dtype=np.float32)
            return
        self.xyxy = np.array(
            [
                [target_x, 18, target_x + 14, 48],
                [21, 18, 35, 48],
            ],
            dtype=np.float32,
        )
        self.conf = np.array([0.91, 0.96], dtype=np.float32)
        self.cls = np.array([0, 0], dtype=np.float32)


class NearLookalikeGapFakeResult:
    def __init__(self, frame_index: int) -> None:
        self.boxes = NearLookalikeGapFakeBoxes(frame_index)


class NearLookalikeGapFakeYolo:
    def __init__(self) -> None:
        self.frame_index = 0

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[NearLookalikeGapFakeResult]:
        result = NearLookalikeGapFakeResult(self.frame_index)
        self.frame_index += 1
        return [result]


class NearLookalikeGapSegmentationResult(NearLookalikeGapFakeResult):
    def __init__(self, frame_index: int) -> None:
        super().__init__(frame_index)
        boxes = self.boxes.xyxy
        data = np.zeros((len(boxes), 64, 96), dtype=np.float32)
        for index, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            data[index, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        self.masks = type("FakeMasks", (), {"data": data})()


class NearLookalikeGapSegmentationYolo:
    def __init__(self) -> None:
        self.frame_index = 0

    def predict(
        self, frame: np.ndarray, **kwargs: object
    ) -> list[NearLookalikeGapSegmentationResult]:
        result = NearLookalikeGapSegmentationResult(self.frame_index)
        self.frame_index += 1
        return [result]


class NearLookalikeGapFakeBoxmotTracker:
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        rows = []
        if len(detections) > 1:
            target = detections[0]
            rows.append([*target[:4], 3, target[4], 0, 0])
            lookalike = detections[1]
        else:
            lookalike = detections[0]
        rows.append([*lookalike[:4], 6, lookalike[4], 0, 0])
        return np.array(rows, dtype=np.float32)


def test_boxmot_identity_tracking_uses_prompt_to_select_target_track(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(identity_runner, "_load_yolo_model", lambda detector_model: FakeYolo())
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: FakeBoxmotTracker(),
    )

    result = run_identity_tracking(run_dir=run_dir, backend="boxmot_botsort", device="cpu")

    assert result["status"] == "complete"
    assert result["backend"] == "boxmot_botsort"
    assert result["metrics"]["target_track_id"] == 7
    assert result["metrics"]["boxmot_tracker"] == "BotSort"

    track_table = pq.read_table(run_dir / "tracklets.parquet")
    track_rows = track_table.to_pylist()
    assert track_table.num_rows == 18
    assert {row["track_id"] for row in track_rows} == {7}
    assert all(row["is_target"] for row in track_rows)

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    track_seed = json.loads((run_dir / "track_seed.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["detector_tracker"]["backend"] == "boxmot_botsort"
    assert track_seed["backend"] == "boxmot_botsort"
    assert track_seed["tracker"]["name"] == "BotSort"
    assert track_seed["target_track_id"] == 7


def test_boxmot_identity_tracking_recovers_when_tracker_id_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_switch_identity_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner, "_load_yolo_model", lambda detector_model: SwitchFakeYolo()
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: SwitchFakeBoxmotTracker(),
    )

    result = run_identity_tracking(run_dir=run_dir, backend="boxmot_botsort", device="cpu")

    assert result["status"] == "complete"
    assert result["metrics"]["target_track_id"] == 3
    assert result["metrics"]["target_track_ids"] == [3, 7]
    assert result["metrics"]["target_track_switches"] == 1

    track_rows = pq.read_table(run_dir / "tracklets.parquet").to_pylist()
    target_rows = {
        int(row["frame_index"]): row
        for row in track_rows
        if row["is_target"] and row["identity_state"] == "usable"
    }
    assert target_rows[9]["track_id"] == 3
    assert target_rows[10]["track_id"] == 7

    track_seed = json.loads((run_dir / "track_seed.json").read_text(encoding="utf-8"))
    assert track_seed["target_track_id"] == 3
    assert track_seed["target_track_ids"] == [3, 7]
    assert track_seed["dynamic_target_selection"] is True


def test_dynamic_identity_state_trusts_strong_same_track_continuity() -> None:
    state, reasons, memory_updated = identity_runner._dynamic_candidate_state(
        candidate={},
        appearance_similarity=0.52,
        prompt_similarity=0.42,
        memory_similarity=0.80,
        continuity_iou=0.83,
        center_score=0.89,
        area_score=0.90,
        impossible_motion=False,
        same_track=True,
        reid_accept=0.65,
        reid_recover=0.58,
    )

    assert state == "usable"
    assert reasons == []
    assert memory_updated is False


def test_dynamic_identity_state_rejects_weak_stale_track() -> None:
    state, reasons, memory_updated = identity_runner._dynamic_candidate_state(
        candidate={},
        appearance_similarity=0.48,
        prompt_similarity=0.44,
        memory_similarity=0.56,
        continuity_iou=0.37,
        center_score=0.70,
        area_score=0.80,
        impossible_motion=False,
        same_track=True,
        reid_accept=0.65,
        reid_recover=0.58,
    )

    assert state == "identity_risk"
    assert "low_prompt_anchor_similarity" in reasons
    assert memory_updated is False


def test_boxmot_identity_tracking_rejects_impossible_lookalike_jump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_lookalike_jump_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner, "_load_yolo_model", lambda detector_model: LookalikeFakeYolo()
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: LookalikeFakeBoxmotTracker(),
    )

    result = run_identity_tracking(run_dir=run_dir, backend="boxmot_botsort", device="cpu")

    assert result["status"] == "complete"
    assert result["metrics"]["target_track_ids"] == [3]

    track_rows = pq.read_table(run_dir / "tracklets.parquet").to_pylist()
    target_rows = {
        int(row["frame_index"]): row
        for row in track_rows
        if row["is_target"] and row["identity_state"] == "usable"
    }
    assert target_rows[10]["track_id"] == 3
    assert target_rows[11]["track_id"] == 3


def test_boxmot_identity_tracking_marks_missing_instead_of_near_lookalike_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_lookalike_jump_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: NearLookalikeGapFakeYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: NearLookalikeGapFakeBoxmotTracker(),
    )

    result = run_identity_tracking(run_dir=run_dir, backend="boxmot_botsort", device="cpu")

    assert result["status"] == "complete"
    assert result["metrics"]["target_track_ids"] == [3]

    track_rows = pq.read_table(run_dir / "tracklets.parquet").to_pylist()
    lookalike_target_rows = [
        row
        for row in track_rows
        if row["is_target"] and row["track_id"] == 6 and row["identity_state"] == "usable"
    ]
    assert lookalike_target_rows == []

    target_rows = {int(row["frame_index"]): row for row in track_rows if row["is_target"]}
    assert target_rows[8]["track_id"] == 6
    assert target_rows[8]["identity_state"] == "identity_risk"


def test_inline_segmentation_blanks_identity_risk_without_box_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)
    write_lookalike_jump_video(run_dir / "source_segment.mp4")

    monkeypatch.setattr(
        identity_runner,
        "identity_setup_status",
        lambda backend=None: {"ready": True, "reasons": [], "install_command": None},
    )
    monkeypatch.setattr(
        identity_runner,
        "_load_yolo_model",
        lambda detector_model: NearLookalikeGapSegmentationYolo(),
    )
    monkeypatch.setattr(
        identity_runner,
        "_create_boxmot_tracker",
        lambda backend, reid_weights, device, half: NearLookalikeGapFakeBoxmotTracker(),
    )

    result = run_identity_tracking(
        run_dir=run_dir,
        backend="boxmot_botsort",
        detector_model="yolo26n-seg.pt",
        device="cpu",
        inline_segmentation=True,
        inline_mask_dilation_pixels=0,
    )

    safety = result["inline_mask"]["safety"]
    assert 8 in safety["identity_risk_blank_frame_indexes"]
    assert result["inline_mask"]["summary"]["sam_fallback_recommended"] is True
    assert result["inline_mask"]["fallback"]["sam_fallback_recommended"] is True
    metadata_rows = [
        json.loads(line)
        for line in Path(result["inline_mask"]["metadata"]).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata_rows[8]["identity_state"] == "identity_risk"
    assert metadata_rows[8]["source"] == "blank"
    assert metadata_rows[8]["fallback"] is False
    assert metadata_rows[8]["mask_area"] == 0
