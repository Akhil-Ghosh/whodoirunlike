from __future__ import annotations

import numpy as np

from whodoirunlike.sam31_gpu_runner import _mask_from_outputs


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
