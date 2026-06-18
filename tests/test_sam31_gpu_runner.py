from __future__ import annotations

import numpy as np

from whodoirunlike.sam31_gpu_runner import _mask_from_outputs, _patch_multiplex_init_state_kwargs


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
