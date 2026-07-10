from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import whodoirunlike.sam31_mlx_runner as sam31_mlx_runner

from whodoirunlike.sam31_mlx_runner import (
    box_iou,
    build_sam31_progress,
    choose_detection_index,
    load_track_boxes,
    mask_box,
    resolve_sam31_resolution,
    run_sam31_mlx_mask,
    update_manifest_after_sam31_mlx,
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


def test_choose_detection_strict_prompt_box_rejects_wrong_identity() -> None:
    masks = np.ones((2, 100, 100), dtype=np.uint8)
    selected = choose_detection_index(
        boxes=np.array([[0, 0, 20, 20], [70, 70, 95, 95]], dtype=float),
        masks=masks,
        scores=np.array([0.3, 0.99]),
        width=100,
        height=100,
        prompt_box=np.array([0, 0, 22, 22], dtype=float),
        strict_prompt_box=True,
        min_prompt_iou=0.3,
    )

    assert selected == 0


def test_choose_detection_strict_prompt_box_returns_none_when_track_is_unmatched() -> None:
    masks = np.ones((1, 100, 100), dtype=np.uint8)
    selected = choose_detection_index(
        boxes=np.array([[70, 70, 95, 95]], dtype=float),
        masks=masks,
        scores=np.array([0.99]),
        width=100,
        height=100,
        prompt_box=np.array([0, 0, 22, 22], dtype=float),
        strict_prompt_box=True,
        min_prompt_iou=0.3,
    )

    assert selected is None


def test_resolve_sam31_resolution_modes() -> None:
    assert resolve_sam31_resolution(mode="fast") == 224
    assert resolve_sam31_resolution(mode="native") == 1008
    assert resolve_sam31_resolution(mode="max", video_meta={"width": 1920, "height": 1080}) == 1918
    assert resolve_sam31_resolution(mode="max", video_meta={"width": 3840, "height": 2160}) == 2016
    assert resolve_sam31_resolution(mode="native", resolution=336) == 336


def test_build_sam31_progress_estimates_eta() -> None:
    progress = build_sam31_progress(
        phase="detecting",
        processed_frames=25,
        total_frames=100,
        elapsed_seconds=50,
        frame_index=24,
        direction="forward",
        detection_count=3,
        selected=True,
        resolution=1008,
    )

    assert progress["percent"] == 0.25
    assert progress["eta_seconds"] == 150.0
    assert progress["frame_index"] == 24
    assert progress["direction"] == "forward"
    assert progress["detection_count"] == 3
    assert progress["selected"] is True
    assert progress["resolution"] == 1008


def test_load_track_boxes_ignores_missing_manifest_paths() -> None:
    assert load_track_boxes({}, width=100, height=80) == {}


def test_load_track_boxes_reads_target_rows(tmp_path: Path) -> None:
    path = tmp_path / "tracklets.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "frame_index": 0,
                        "is_target": False,
                        "identity_state": "usable",
                        "bbox_x": 0.1,
                        "bbox_y": 0.2,
                        "bbox_width": 0.3,
                        "bbox_height": 0.4,
                    }
                ),
                json.dumps(
                    {
                        "frame_index": 1,
                        "is_target": True,
                        "identity_state": "usable",
                        "bbox_x": 0.1,
                        "bbox_y": 0.2,
                        "bbox_width": 0.3,
                        "bbox_height": 0.4,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    boxes = load_track_boxes({"tracklets_jsonl": str(path)}, width=100, height=80)

    assert sorted(boxes) == [1]
    np.testing.assert_allclose(boxes[1], [9.9, 15.8, 39.6, 47.4])


def test_run_sam31_mlx_prefers_manifest_track_paths_to_stale_stage_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custom_jsonl_path = tmp_path / "custom" / "identity-tracklets.jsonl"
    custom_parquet_path = tmp_path / "custom" / "identity-tracklets.parquet"
    manifest_path = tmp_path / "cv_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "clip-1",
                "paths": {
                    "tracklets_jsonl": str(custom_jsonl_path),
                    "tracklets": str(custom_parquet_path),
                },
                "stages": {
                    "detector_tracker": {
                        "tracklets_jsonl": str(tmp_path / "stale" / "tracklets.jsonl"),
                        "tracklets": str(tmp_path / "stale" / "tracklets.parquet"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    fake_modules = {
        "mlx_vlm.generate": {"wired_limit": object()},
        "mlx_vlm.models.sam3.generate": {"Sam3Predictor": object()},
        "mlx_vlm.models.sam3_1.processing_sam3_1": {"Sam31Processor": object()},
        "mlx_vlm.utils": {"get_model_path": object(), "load_model": object()},
    }
    for module_name, attributes in fake_modules.items():
        module = types.ModuleType(module_name)
        for name, value in attributes.items():
            setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    monkeypatch.setattr(
        sam31_mlx_runner,
        "inspect_video",
        lambda _path: {"width": 100, "height": 80, "fps": 30.0},
    )
    monkeypatch.setattr(
        sam31_mlx_runner,
        "extract_video_frames",
        lambda *_args, **_kwargs: [tmp_path / "frame.jpg"],
    )
    monkeypatch.setattr(
        sam31_mlx_runner,
        "load_prompt",
        lambda *_args, **_kwargs: {"frame_index": 0, "box": None, "raw": {}},
    )

    captured_paths: dict[str, object] = {}

    class TrackPathsCaptured(Exception):
        pass

    def capture_track_paths(paths: dict[str, object], **_kwargs: object) -> dict[int, np.ndarray]:
        captured_paths.update(paths)
        raise TrackPathsCaptured

    monkeypatch.setattr(sam31_mlx_runner, "load_track_boxes", capture_track_paths)

    with pytest.raises(TrackPathsCaptured):
        run_sam31_mlx_mask(run_dir=tmp_path)

    assert captured_paths == {
        "tracklets_jsonl": str(custom_jsonl_path),
        "tracklets": str(custom_parquet_path),
    }


def test_update_manifest_after_sam31_mlx_preserves_unknown_manifest_data(tmp_path: Path) -> None:
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

    update_manifest_after_sam31_mlx(
        manifest_path,
        metadata_path,
        masks_jsonl_path,
        model_path="mlx-community/sam3.1-bf16",
        prompts=("a runner", "a person"),
        quality_mode="native",
        resolution=1008,
        elapsed_seconds=1.25,
        mask_summary={"nonempty_frames": 3},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = manifest["stages"]["whole_runner_mask"]
    assert stage["status"] == "complete"
    assert stage["backend"] == "sam31_mlx"
    assert stage["custom_stage_value"] == 7
    assert "error" not in stage
    assert manifest["paths"]["masks_jsonl"] == str(configured_masks_path)
    assert manifest["custom_top_level"] == {"keep": True}
    assert manifest["stages"]["future_stage"] == {"status": "vendor-specific"}
