from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import whodoirunlike.sam2_runner as sam2_runner
from whodoirunlike.sam2_runner import write_mask_outputs


def test_write_mask_outputs_creates_each_output_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "00000.jpg"
    assert cv2.imwrite(str(frame_path), np.zeros((8, 10, 3), dtype=np.uint8))

    runner_mask_path = tmp_path / "runner-mask" / "runner_mask.mp4"
    masked_runner_path = tmp_path / "masked-runner" / "masked_runner.mp4"
    qa_overlay_path = tmp_path / "qa-overlay" / "qa_overlay.mp4"
    metadata_path = tmp_path / "metadata" / "runner_mask_metadata.jsonl"

    class ParentCheckingVideoWriter:
        def __init__(self, path: str, *_args: object) -> None:
            self.path = Path(path)
            assert self.path.parent.is_dir()
            self.path.touch()

        def isOpened(self) -> bool:
            return True

        def write(self, _frame: np.ndarray) -> None:
            return None

        def release(self) -> None:
            return None

    monkeypatch.setattr(sam2_runner.cv2, "VideoWriter", ParentCheckingVideoWriter)
    monkeypatch.setattr(sam2_runner, "make_browser_playable_mp4s", lambda _paths: None)

    write_mask_outputs(
        frame_paths=[frame_path],
        masks_by_frame={0: np.ones((8, 10), dtype=np.uint8)},
        fps=30.0,
        runner_mask_path=runner_mask_path,
        masked_runner_path=masked_runner_path,
        qa_overlay_path=qa_overlay_path,
        metadata_path=metadata_path,
    )

    assert runner_mask_path.is_file()
    assert masked_runner_path.is_file()
    assert qa_overlay_path.is_file()
    assert metadata_path.is_file()
