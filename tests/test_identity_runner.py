from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from whodoirunlike import identity_runner
from whodoirunlike.identity_runner import (
    canonical_identity_backend,
    prompt_initial_box,
    run_identity_tracking,
)


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


def test_canonical_identity_backend_accepts_plan_aliases() -> None:
    assert canonical_identity_backend("botsort") == "boxmot_botsort"
    assert canonical_identity_backend("deep-oc-sort") == "boxmot_deepocsort"
    assert canonical_identity_backend("template") == "prompt_template_tracker_v1"


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
    monkeypatch.setattr(identity_runner, "_load_yolo_model", lambda detector_model: SwitchFakeYolo())
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
    monkeypatch.setattr(identity_runner, "_load_yolo_model", lambda detector_model: LookalikeFakeYolo())
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
