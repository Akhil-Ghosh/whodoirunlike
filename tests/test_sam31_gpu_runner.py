from __future__ import annotations

import json

import numpy as np

from whodoirunlike.sam31_gpu_runner import (
    _build_track_box_fallback_masks,
    _filter_masks_to_track_boxes,
    _mask_from_outputs,
    _patch_multiplex_init_state_kwargs,
    _prompt_points_with_box_support,
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
    manifest_path.write_text(
        json.dumps(
            {
                "candidate_id": "clip-1",
                "paths": {},
                "stages": {"whole_runner_mask": {"error": "old failure"}},
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
