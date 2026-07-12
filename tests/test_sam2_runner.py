from __future__ import annotations

from pathlib import Path
import shutil

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


def test_split_mask_outputs_publish_data_before_optional_presentation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths: list[Path] = []
    for frame_index in range(2):
        frame_path = frame_dir / f"{frame_index:05d}.jpg"
        frame = np.full((8, 10, 3), frame_index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)

    runner_mask_path = tmp_path / "runner-mask" / "runner_mask.mp4"
    masked_runner_path = tmp_path / "masked-runner" / "masked_runner.mp4"
    qa_overlay_path = tmp_path / "qa-overlay" / "qa_overlay.mp4"
    metadata_path = tmp_path / "metadata" / "runner_mask_metadata.jsonl"
    writer_paths: list[Path] = []
    encoded_batches: list[list[Path]] = []

    class RecordingVideoWriter:
        def __init__(self, path: str, *_args: object) -> None:
            self.path = Path(path)
            writer_paths.append(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()

        def isOpened(self) -> bool:
            return True

        def write(self, _frame: np.ndarray) -> None:
            return None

        def release(self) -> None:
            return None

    monkeypatch.setattr(sam2_runner.cv2, "VideoWriter", RecordingVideoWriter)
    monkeypatch.setattr(
        sam2_runner,
        "make_browser_playable_mp4s",
        lambda paths: encoded_batches.append(list(paths)),
    )

    sam2_runner.write_runner_mask_data_outputs(
        frame_paths=frame_paths,
        masks_by_frame={
            0: np.ones((8, 10), dtype=np.uint8),
            1: np.ones((8, 10), dtype=np.uint8),
        },
        fps=30.0,
        runner_mask_path=runner_mask_path,
        metadata_path=metadata_path,
    )

    assert writer_paths == [runner_mask_path]
    assert encoded_batches == [[runner_mask_path]]
    assert metadata_path.read_text(encoding="utf-8").count("\n") == 2
    assert not masked_runner_path.exists()
    assert not qa_overlay_path.exists()

    sam2_runner.write_mask_presentation_outputs(
        frame_paths=frame_paths,
        masks_by_frame={
            0: np.ones((8, 10), dtype=np.uint8),
            1: np.ones((8, 10), dtype=np.uint8),
        },
        fps=30.0,
        masked_runner_path=masked_runner_path,
        qa_overlay_path=qa_overlay_path,
        render_qa_overlay=False,
    )

    assert writer_paths == [runner_mask_path, masked_runner_path]
    assert encoded_batches == [[runner_mask_path], [masked_runner_path]]
    assert masked_runner_path.is_file()
    assert not qa_overlay_path.exists()


def test_split_mask_outputs_are_decoded_frame_equivalent(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for browser-video parity")

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths: list[Path] = []
    masks_by_frame: dict[int, np.ndarray] = {}
    for frame_index in range(12):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = frame_index * 11
        frame[:, :, 1] = np.arange(64, dtype=np.uint8)
        frame[:, :, 2] = np.arange(48, dtype=np.uint8)[:, None]
        frame_path = frame_dir / f"{frame_index:05d}.png"
        assert cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        mask = np.zeros((48, 64), dtype=np.uint8)
        left = 4 + frame_index
        mask[8:40, left : left + 18] = 255
        masks_by_frame[frame_index] = mask

    combined = tmp_path / "combined"
    split = tmp_path / "split"
    combined.mkdir()
    split.mkdir()
    combined_paths = {
        "runner": combined / "runner_mask.mp4",
        "masked": combined / "masked_runner.mp4",
        "qa": combined / "qa_overlay.mp4",
        "metadata": combined / "metadata.jsonl",
    }
    split_paths = {
        "runner": split / "runner_mask.mp4",
        "masked": split / "masked_runner.mp4",
        "qa": split / "qa_overlay.mp4",
        "metadata": split / "metadata.jsonl",
    }

    write_mask_outputs(
        frame_paths=frame_paths,
        masks_by_frame=masks_by_frame,
        fps=30000 / 1001,
        runner_mask_path=combined_paths["runner"],
        masked_runner_path=combined_paths["masked"],
        qa_overlay_path=combined_paths["qa"],
        metadata_path=combined_paths["metadata"],
    )
    sam2_runner.write_runner_mask_data_outputs(
        frame_paths=frame_paths,
        masks_by_frame=masks_by_frame,
        fps=30000 / 1001,
        runner_mask_path=split_paths["runner"],
        metadata_path=split_paths["metadata"],
    )
    sam2_runner.write_mask_presentation_outputs(
        frame_paths=frame_paths,
        masks_by_frame=masks_by_frame,
        fps=30000 / 1001,
        masked_runner_path=split_paths["masked"],
        qa_overlay_path=split_paths["qa"],
        render_qa_overlay=True,
    )

    assert combined_paths["metadata"].read_bytes() == split_paths["metadata"].read_bytes()
    for artifact in ("runner", "masked", "qa"):
        combined_capture = cv2.VideoCapture(str(combined_paths[artifact]))
        split_capture = cv2.VideoCapture(str(split_paths[artifact]))
        compared_frames = 0
        while True:
            combined_ok, combined_frame = combined_capture.read()
            split_ok, split_frame = split_capture.read()
            assert combined_ok is split_ok
            if not combined_ok:
                break
            assert np.array_equal(combined_frame, split_frame)
            compared_frames += 1
        combined_capture.release()
        split_capture.release()
        assert compared_frames == len(frame_paths)
