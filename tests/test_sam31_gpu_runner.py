from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np

from whodoirunlike.sam31_gpu_runner import (
    _build_track_box_fallback_masks,
    _collect_sam31_masks,
    _configure_interactive_tracker_for_user_prompt,
    _filter_masks_to_track_boxes,
    _first_nonempty_output_object_id,
    _has_prompt_points,
    _mask_from_outputs,
    _patch_multiplex_init_state_kwargs,
    _prompt_points_with_box_support,
    _seed_points_for_frame,
    _support_points_from_box,
    _track_prompt_anchors,
    update_manifest_after_sam31_gpu,
)


def test_mask_from_outputs_selects_requested_object() -> None:
    outputs = {
        "out_obj_ids": np.array([7, 1]),
        "out_binary_masks": np.array(
            [
                [[0, 1], [0, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    }

    mask = _mask_from_outputs(outputs, obj_id=1)

    assert mask is not None
    assert mask.tolist() == [[1, 1], [0, 0]]


def test_mask_from_outputs_falls_back_to_first_nonempty_mask() -> None:
    outputs = {
        "out_obj_ids": np.array([3]),
        "out_binary_masks": np.array([[[0, 0], [1, 0]]], dtype=np.uint8),
    }

    mask = _mask_from_outputs(outputs, obj_id=1)

    assert mask is not None
    assert mask.tolist() == [[0, 0], [1, 0]]


def test_first_nonempty_output_object_id_uses_mask_content() -> None:
    outputs = {
        "out_obj_ids": np.array([3, 7]),
        "out_binary_masks": np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    }

    assert _first_nonempty_output_object_id(outputs) == 7


def test_patch_multiplex_init_state_kwargs_filters_unknown_kwargs() -> None:
    class Model:
        def init_state(self, video_path: str) -> dict[str, str]:
            return {"video_path": video_path}

    class Predictor:
        model = Model()

    predictor = Predictor()

    _patch_multiplex_init_state_kwargs(predictor)

    assert predictor.model.init_state(
        video_path="clip.mp4",
        offload_state_to_cpu=True,
        offload_video_to_cpu=True,
    ) == {"video_path": "clip.mp4"}


def test_configure_interactive_tracker_disables_demo_suppression() -> None:
    class Model:
        masklet_confirmation_enable = True
        hotstart_delay = 15
        hotstart_unmatch_thresh = 8
        hotstart_dup_thresh = 8
        suppress_unmatched_only_within_hotstart = False
        suppress_overlapping_based_on_recent_occlusion_threshold = 0.7

    class Predictor:
        model = Model()

    result = _configure_interactive_tracker_for_user_prompt(Predictor())

    assert result["applied"] is True
    assert Predictor.model.masklet_confirmation_enable is False
    assert Predictor.model.hotstart_delay == 0
    assert Predictor.model.suppress_unmatched_only_within_hotstart is True
    assert Predictor.model.suppress_overlapping_based_on_recent_occlusion_threshold == 1.1


def test_prompt_points_with_box_support_keeps_user_point_and_adds_runner_body_points() -> None:
    prompt = {
        "points": np.array([[20, 30]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int32),
    }

    points, labels = _prompt_points_with_box_support(
        prompt,
        box=np.array([10, 10, 30, 50], dtype=np.float32),
    )

    assert points is not None
    assert labels is not None
    assert points.shape == (4, 2)
    assert labels.tolist() == [1, 1, 1, 1]
    assert points[0].tolist() == [20, 30]
    assert points[1:].tolist() == _support_points_from_box(
        np.array([10, 10, 30, 50], dtype=np.float32)
    ).tolist()


def test_seed_points_for_frame_uses_track_box_points_when_seed_frame_differs() -> None:
    prompt = {
        "points": np.array([[90, 90]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int32),
    }
    box = np.array([10, 10, 30, 50], dtype=np.float32)

    points, labels = _seed_points_for_frame(
        prompt=prompt,
        box=box,
        seed_frame=0,
        prompt_frame=99,
    )

    assert points is not None
    assert labels is not None
    assert points.tolist() == _support_points_from_box(box).tolist()
    assert labels.tolist() == [1, 1, 1]


def test_has_prompt_points_handles_numpy_arrays() -> None:
    assert _has_prompt_points({"points": np.array([[20, 30]], dtype=np.float32)}) is True
    assert _has_prompt_points({"points": np.zeros((0, 2), dtype=np.float32)}) is False
    assert _has_prompt_points({}) is False


def test_track_prompt_anchors_choose_prompt_start_middle_and_end_boxes() -> None:
    boxes = {
        0: np.array([0, 0, 10, 10], dtype=np.float32),
        50: np.array([1, 0, 11, 10], dtype=np.float32),
        99: np.array([2, 0, 12, 10], dtype=np.float32),
        150: np.array([3, 0, 13, 10], dtype=np.float32),
        199: np.array([4, 0, 14, 10], dtype=np.float32),
    }

    anchors = _track_prompt_anchors(
        prompt_frame=99,
        frame_count=200,
        track_boxes=boxes,
        max_anchors=4,
    )

    assert 99 in anchors
    assert anchors[0] == 0
    assert len(anchors) <= 4


def test_filter_masks_to_track_boxes_rejects_off_target_masks() -> None:
    on_target = np.zeros((30, 40), dtype=np.uint8)
    on_target[5:20, 10:25] = 1
    off_target = np.zeros((30, 40), dtype=np.uint8)
    off_target[5:20, 28:38] = 1

    filtered, summary = _filter_masks_to_track_boxes(
        {0: on_target, 1: off_target},
        {
            0: np.array([9, 4, 26, 21], dtype=np.float32),
            1: np.array([9, 4, 26, 21], dtype=np.float32),
        },
    )

    assert sorted(filtered) == [0]
    assert summary["enabled"] is True
    assert summary["accepted_frames"] == 1
    assert summary["rejected_frames"] == 1


def test_track_box_fallback_masks_follow_target_boxes_and_fill_short_gaps() -> None:
    masks, info = _build_track_box_fallback_masks(
        {
            0: np.array([10, 5, 20, 25], dtype=np.float32),
            2: np.array([14, 5, 24, 25], dtype=np.float32),
        },
        width=40,
        height=30,
        frame_count=3,
        max_interpolation_gap=4,
    )

    assert sorted(masks) == [0, 1, 2]
    assert info["backend"] == "identity_track_box"
    assert info["generated_frames"] == 3
    assert info["track_box_frames"] == 2
    assert info["interpolated_frames"] == 1
    assert masks[0][15, 15] == 1
    assert masks[1][15, 17] == 1
    assert masks[2][15, 19] == 1
    assert masks[1][1, 1] == 0


def test_update_manifest_after_sam31_gpu_records_fallback(tmp_path) -> None:
    manifest_path = tmp_path / "cv_run_manifest.json"
    metadata_path = tmp_path / "runner_mask_metadata.jsonl"
    masks_jsonl_path = tmp_path / "masks.jsonl"
    configured_masks_path = tmp_path / "custom-layout" / "runner-masks.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "clip-1",
                "custom_top_level": {"keep": True},
                "paths": {"masks_jsonl": str(configured_masks_path)},
                "stages": {
                    "whole_runner_mask": {"error": "old failure", "custom_stage_value": 7},
                    "future_stage": {"status": "vendor-specific"},
                },
            }
        ),
        encoding="utf-8",
    )

    update_manifest_after_sam31_gpu(
        manifest_path,
        metadata_path,
        masks_jsonl_path,
        checkpoint_path=None,
        elapsed_seconds=1.25,
        mask_summary={"nonempty_frames": 3},
        fallback={
            "backend": "identity_track_box",
            "reason": "sam31_gpu_empty_mask",
            "generated_frames": 3,
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = manifest["stages"]["whole_runner_mask"]
    assert stage["status"] == "complete"
    assert stage["fallback"]["reason"] == "sam31_gpu_empty_mask"
    assert stage["mask_summary"]["nonempty_frames"] == 3
    assert "error" not in stage
    assert stage["custom_stage_value"] == 7
    assert manifest["paths"]["masks_jsonl"] == str(configured_masks_path)
    assert manifest["custom_top_level"] == {"keep": True}
    assert manifest["stages"]["future_stage"] == {"status": "vendor-specific"}


def test_collect_sam31_masks_tracks_visual_box_object_id(monkeypatch, tmp_path: Path) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self) -> None:
            self.add_prompts: list[dict[str, object]] = []
            self.stream_requests: list[dict[str, object]] = []

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            request_type = request["type"]
            if request_type == "start_session":
                return {"session_id": "session-1"}
            if request_type == "close_session":
                return {}
            if request_type == "add_prompt":
                self.add_prompts.append(request)
                if "bounding_boxes" in request:
                    return {
                        "outputs": {
                            "out_obj_ids": np.array([7]),
                            "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                        }
                    }
                return {
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_binary_masks": np.array([[[0, 1], [0, 0]]], dtype=np.uint8),
                    }
                }
            raise AssertionError(f"unexpected request: {request_type}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            assert request["type"] == "propagate_in_video"
            self.stream_requests.append(request)
            return [
                {
                    "frame_index": 1,
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_binary_masks": np.array([[[1, 1], [0, 0]]], dtype=np.uint8),
                    },
                }
            ]

    predictor = FakePredictor()
    masks, diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 0,
            "box": np.array([0, 0, 2, 2], dtype=np.float32),
            "points": np.array([[1, 1]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=2,
        height=2,
        frame_count=2,
        obj_id=1,
    )

    point_prompt = predictor.add_prompts[1]
    assert point_prompt["obj_id"] == 7
    assert diagnostics["visual_box_prompt"] is True
    assert diagnostics["visual_box_obj_id"] == 7
    assert diagnostics["active_obj_id"] == 7
    assert [request["propagation_direction"] for request in predictor.stream_requests] == ["forward"]
    assert {request["start_frame_index"] for request in predictor.stream_requests} == {0}
    assert sorted(masks) == [0, 1]
    assert masks[1].tolist() == [[1, 1], [0, 0]]


def test_collect_sam31_masks_seeds_from_first_target_track_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self) -> None:
            self.add_prompts: list[dict[str, object]] = []
            self.stream_requests: list[dict[str, object]] = []

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            request_type = request["type"]
            if request_type == "start_session":
                return {"session_id": "session-1"}
            if request_type == "close_session":
                return {}
            if request_type == "add_prompt":
                self.add_prompts.append(request)
                return {
                    "outputs": {
                        "out_obj_ids": np.array([1]),
                        "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                    }
                }
            raise AssertionError(f"unexpected request: {request_type}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            self.stream_requests.append(request)
            return []

    predictor = FakePredictor()
    _masks, diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 99,
            "box": np.array([70, 70, 90, 100], dtype=np.float32),
            "points": np.array([[80, 90]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=100,
        height=100,
        frame_count=120,
        obj_id=1,
        track_boxes={
            0: np.array([10, 10, 30, 50], dtype=np.float32),
            99: np.array([70, 70, 90, 100], dtype=np.float32),
        },
    )

    assert diagnostics["seed_frame"] == 0
    assert diagnostics["seed_source"] == "target_track_first_visible_frame"
    assert predictor.add_prompts[0]["frame_index"] == 0
    assert predictor.add_prompts[1]["frame_index"] == 0
    point_prompt = predictor.add_prompts[1]
    assert np.asarray(point_prompt["points"]).shape == (3, 2)
    expected_points = _support_points_from_box(np.array([10, 10, 30, 50], dtype=np.float32))
    expected_points[:, 0] /= 100
    expected_points[:, 1] /= 100
    assert np.asarray(point_prompt["points"]).tolist() == expected_points.tolist()
    assert predictor.stream_requests[0]["start_frame_index"] == 0
    assert predictor.stream_requests[0]["propagation_direction"] == "forward"
