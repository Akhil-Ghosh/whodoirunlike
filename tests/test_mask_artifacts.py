from __future__ import annotations

import numpy as np

from whodoirunlike.mask_artifacts import encode_uncompressed_rle


def _scalar_uncompressed_rle(mask: np.ndarray) -> dict[str, object]:
    binary = (mask > 0).astype("uint8")
    height, width = binary.shape[:2]
    pixels = binary.ravel(order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for raw_value in pixels:
        value = int(raw_value)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            current = value
            run_length = 1
    counts.append(run_length)
    return {"size": [height, width], "counts": counts}


def test_vectorized_uncompressed_rle_matches_scalar_reference() -> None:
    rng = np.random.default_rng(20260710)
    masks = [
        np.zeros((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
        np.array([[0, 1], [1, 1]], dtype=np.uint8),
        *(rng.integers(0, 2, size=(height, width), dtype=np.uint8)
          for height, width in ((1, 1), (1, 17), (13, 1), (9, 11), (64, 96))),
    ]

    for mask in masks:
        assert encode_uncompressed_rle(mask) == _scalar_uncompressed_rle(mask)
