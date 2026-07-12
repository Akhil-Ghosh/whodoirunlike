from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from whodoirunlike import qc as qc_module
from whodoirunlike.mask_artifacts import write_masks_jsonl_from_video


def _write_mask_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (20, 16),
    )
    assert writer.isOpened()
    for frame_index in range(3):
        frame = np.zeros((16, 20, 3), dtype=np.uint8)
        cv2.rectangle(frame, (2 + frame_index, 4), (8 + frame_index, 12), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _write_run_manifest(
    run_dir: Path,
    *,
    mask_summary: dict[str, object] | None,
) -> None:
    runner_mask = run_dir / "runner_mask.mp4"
    masks_jsonl = run_dir / "masks.jsonl"
    whole_runner_mask: dict[str, object] = {
        "status": "complete",
        "backend": "sam31_gpu",
        "masks_jsonl": str(masks_jsonl),
    }
    if mask_summary is not None:
        whole_runner_mask["mask_summary"] = mask_summary
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "candidate-qc",
                "paths": {
                    "runner_mask": str(runner_mask),
                    "masks_jsonl": str(masks_jsonl),
                    "qc_metrics": str(run_dir / "qc_metrics.json"),
                },
                "stages": {"whole_runner_mask": whole_runner_mask},
            }
        ),
        encoding="utf-8",
    )


def test_run_qc_reuses_current_sam_mask_summary_without_decoding_or_rewriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_mask = tmp_path / "runner_mask.mp4"
    masks_jsonl = tmp_path / "masks.jsonl"
    _write_mask_video(runner_mask)
    summary = write_masks_jsonl_from_video(runner_mask, masks_jsonl)
    expected = {"mask_available": True, **summary}
    _write_run_manifest(tmp_path, mask_summary=summary)

    original_bytes = masks_jsonl.read_bytes()
    original_mtime_ns = masks_jsonl.stat().st_mtime_ns

    def fail_if_decoded(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("current SAM mask artifacts should not be reconstructed")

    monkeypatch.setattr(qc_module, "write_masks_jsonl_from_video", fail_if_decoded)

    payload = qc_module.run_qc_metrics(tmp_path)

    assert payload["mask"] == expected
    assert masks_jsonl.read_bytes() == original_bytes
    assert masks_jsonl.stat().st_mtime_ns == original_mtime_ns


def test_run_qc_reconstructs_mask_metrics_when_sam_summary_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_mask = tmp_path / "runner_mask.mp4"
    masks_jsonl = tmp_path / "masks.jsonl"
    _write_mask_video(runner_mask)
    expected_summary = write_masks_jsonl_from_video(runner_mask, masks_jsonl)
    _write_run_manifest(tmp_path, mask_summary=None)

    reconstruction_calls: list[tuple[Path, Path]] = []
    original_reconstruct = qc_module.write_masks_jsonl_from_video

    def reconstruct(mask_path: Path, output_path: Path) -> dict[str, object]:
        reconstruction_calls.append((mask_path, output_path))
        return original_reconstruct(mask_path, output_path)

    monkeypatch.setattr(qc_module, "write_masks_jsonl_from_video", reconstruct)

    payload = qc_module.run_qc_metrics(tmp_path)

    assert reconstruction_calls == [(runner_mask, masks_jsonl)]
    assert payload["mask"] == {"mask_available": True, **expected_summary}


def test_run_qc_reconstructs_mask_metrics_when_runner_mask_is_newer_than_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_mask = tmp_path / "runner_mask.mp4"
    masks_jsonl = tmp_path / "masks.jsonl"
    _write_mask_video(runner_mask)
    expected_summary = write_masks_jsonl_from_video(runner_mask, masks_jsonl)
    _write_run_manifest(tmp_path, mask_summary=expected_summary)
    newer_ns = masks_jsonl.stat().st_mtime_ns + 1_000_000_000
    os.utime(runner_mask, ns=(newer_ns, newer_ns))

    reconstruction_calls: list[tuple[Path, Path]] = []
    original_reconstruct = qc_module.write_masks_jsonl_from_video

    def reconstruct(mask_path: Path, output_path: Path) -> dict[str, object]:
        reconstruction_calls.append((mask_path, output_path))
        return original_reconstruct(mask_path, output_path)

    monkeypatch.setattr(qc_module, "write_masks_jsonl_from_video", reconstruct)

    payload = qc_module.run_qc_metrics(tmp_path)

    assert reconstruction_calls == [(runner_mask, masks_jsonl)]
    assert payload["mask"] == {"mask_available": True, **expected_summary}


def test_run_qc_reconstructs_mask_metrics_when_masks_jsonl_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_mask = tmp_path / "runner_mask.mp4"
    masks_jsonl = tmp_path / "masks.jsonl"
    _write_mask_video(runner_mask)
    expected_summary = write_masks_jsonl_from_video(runner_mask, masks_jsonl)
    _write_run_manifest(tmp_path, mask_summary=expected_summary)
    masks_jsonl.write_text('{"frame_index":0}\nnot-json\n', encoding="utf-8")

    reconstruction_calls: list[tuple[Path, Path]] = []
    original_reconstruct = qc_module.write_masks_jsonl_from_video

    def reconstruct(mask_path: Path, output_path: Path) -> dict[str, object]:
        reconstruction_calls.append((mask_path, output_path))
        return original_reconstruct(mask_path, output_path)

    monkeypatch.setattr(qc_module, "write_masks_jsonl_from_video", reconstruct)

    payload = qc_module.run_qc_metrics(tmp_path)

    assert reconstruction_calls == [(runner_mask, masks_jsonl)]
    assert payload["mask"] == {"mask_available": True, **expected_summary}
    assert len(masks_jsonl.read_text(encoding="utf-8").splitlines()) == 3
