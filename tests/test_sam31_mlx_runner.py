from __future__ import annotations

import numpy as np

from whodoirunlike.sam31_mlx_runner import (
    box_iou,
    choose_detection_index,
    mask_box,
    resolve_sam31_resolution,
)


def test_box_iou_handles_overlap_and_empty_union() -> None:
    assert round(box_iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])), 3) == 0.143
    assert box_iou(np.array([0, 0, 0, 0]), np.array([5, 5, 15, 15])) == 0.0


def test_mask_box_finds_binary_mask_bounds() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:10, 8:16] = 1

    assert mask_box(mask).tolist() == [8, 4, 15, 9]


def test_choose_detection_prefers_prompt_overlap_over_raw_score() -> None:
    masks = np.ones((2, 100, 100), dtype=np.uint8)
    selected = choose_detection_index(
        boxes=np.array([[0, 0, 10, 10], [50, 50, 80, 80]], dtype=float),
        masks=masks,
        scores=np.array([0.2, 0.9]),
        width=100,
        height=100,
        prompt_box=np.array([0, 0, 12, 12], dtype=float),
    )

    assert selected == 0


def test_resolve_sam31_resolution_modes() -> None:
    assert resolve_sam31_resolution(mode="fast") == 224
    assert resolve_sam31_resolution(mode="native") == 1008
    assert resolve_sam31_resolution(mode="max", video_meta={"width": 1920, "height": 1080}) == 1918
    assert resolve_sam31_resolution(mode="max", video_meta={"width": 3840, "height": 2160}) == 2016
    assert resolve_sam31_resolution(mode="native", resolution=336) == 336
