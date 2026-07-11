from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from whodoirunlike import artifact_tables
from whodoirunlike.active_learning import build_uncertainty_queue
from whodoirunlike.artifact_tables import export_cv_tables
from whodoirunlike.mask_artifacts import encode_uncompressed_rle, write_masks_jsonl_from_video
from whodoirunlike.multiview import cross_view_cost
from whodoirunlike.qc import run_qc_metrics


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_mask_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (20, 16))
    assert writer.isOpened()
    for index in range(3):
        frame = np.zeros((16, 20, 3), dtype=np.uint8)
        cv2.rectangle(frame, (2 + index, 4), (8 + index, 12), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def write_run(run_dir: Path, *, embedding: list[float] | None = None) -> None:
    run_dir.mkdir(parents=True)
    mask_path = run_dir / "runner_mask.mp4"
    write_mask_video(mask_path)
    write_jsonl(
        run_dir / "tracklets.jsonl",
        [
            {
                "frame_index": 0,
                "identity_state": "usable",
                "reid_similarity": 1.0,
                "is_target": True,
                "bbox_x": 0.1,
                "bbox_y": 0.2,
                "bbox_width": 0.2,
                "bbox_height": 0.5,
            },
            {
                "frame_index": 1,
                "identity_state": "identity_risk",
                "reid_similarity": 0.4,
                "is_target": True,
                "bbox_x": 0.2,
                "bbox_y": 0.2,
                "bbox_width": 0.2,
                "bbox_height": 0.5,
            },
            {
                "frame_index": 2,
                "identity_state": "usable",
                "reid_similarity": 0.95,
                "is_target": True,
                "bbox_x": 0.3,
                "bbox_y": 0.2,
                "bbox_width": 0.2,
                "bbox_height": 0.5,
            },
        ],
    )
    write_jsonl(
        run_dir / "reid.jsonl",
        [
            {
                "frame_index": 0,
                "embedding": embedding or [1.0, 0.0],
            }
        ],
    )
    write_jsonl(
        run_dir / "pose_landmarks.jsonl",
        [
            {"frame_index": 0, "usable": True, "visibility_mean": 0.8},
            {"frame_index": 1, "usable": False, "visibility_mean": 0.2, "drop_reason": "occluded"},
        ],
    )
    write_jsonl(run_dir / "densepose.jsonl", [{"frame_index": 0, "usable": True, "score": 0.7}])
    write_jsonl(
        run_dir / "fused_form.jsonl",
        [{"frame_index": 0, "frame_confidence": 0.75, "frame_state": "usable"}],
    )
    write_json(
        run_dir / "view_bucket.json",
        {"version": 1, "view_bucket": "side"},
    )
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": run_dir.name,
            "runner_name": "Runner",
            "paths": {
                "runner_mask": str(mask_path),
                "masks_jsonl": str(run_dir / "masks.jsonl"),
                "tracklets": str(run_dir / "tracklets.parquet"),
                "tracklets_jsonl": str(run_dir / "tracklets.jsonl"),
                "reid": str(run_dir / "reid.parquet"),
                "reid_jsonl": str(run_dir / "reid.jsonl"),
                "pose_landmarks": str(run_dir / "pose_landmarks.jsonl"),
                "poses": str(run_dir / "poses.parquet"),
                "densepose": str(run_dir / "densepose.jsonl"),
                "densepose_parquet": str(run_dir / "densepose.parquet"),
                "fused_form": str(run_dir / "fused_form.jsonl"),
                "fused_form_parquet": str(run_dir / "fused_form.parquet"),
                "qc_metrics": str(run_dir / "qc_metrics.json"),
                "view_bucket": str(run_dir / "view_bucket.json"),
            },
            "stages": {},
        },
    )


def test_mask_video_exports_uncompressed_rle(tmp_path: Path) -> None:
    mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    assert encode_uncompressed_rle(mask) == {"size": [2, 2], "counts": [1, 3]}
    mask_video = tmp_path / "mask.mp4"
    write_mask_video(mask_video)

    summary = write_masks_jsonl_from_video(mask_video, tmp_path / "masks.jsonl")

    rows = [json.loads(line) for line in (tmp_path / "masks.jsonl").read_text().splitlines()]
    assert summary["frame_count"] == 3
    assert rows[0]["rle"]["size"] == [16, 20]
    assert rows[0]["area"] > 0


def test_qc_tables_active_learning_and_multiview(tmp_path: Path) -> None:
    run_a = tmp_path / "cv_runs" / "run-a"
    run_b = tmp_path / "cv_runs" / "run-b"
    write_run(run_a, embedding=[1.0, 0.0])
    write_run(run_b, embedding=[0.9, 0.1])

    qc = run_qc_metrics(run_a)
    tables = export_cv_tables(run_a)
    queue = build_uncertainty_queue(tmp_path / "cv_runs", tmp_path / "queue.json")
    match = cross_view_cost(run_a, run_b)

    assert qc["identity"]["identity_risk_rate"] > 0
    assert qc["mask"]["mask_available"] is True
    assert tables["exports"]["poses"]["status"] == "complete"
    assert queue["entry_count"] == 2
    assert queue["entries"][0]["reason_tags"]
    assert match["appearance_cost"] < 0.01


def test_table_fallbacks_keep_canonical_basenames_with_custom_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "candidate-custom"
    inputs_dir = tmp_path / "custom-layout" / "signals"
    pose_path = inputs_dir / "pose-sequence.jsonl"
    densepose_path = inputs_dir / "body-map.jsonl"
    fused_path = inputs_dir / "form-signal.jsonl"
    write_jsonl(pose_path, [{"frame_index": 0, "usable": True}])
    write_jsonl(densepose_path, [{"frame_index": 0, "usable": False}])
    write_jsonl(fused_path, [{"frame_index": 0, "frame_confidence": 0.5}])
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": "candidate-custom",
            "paths": {
                "pose_landmarks": str(pose_path),
                "densepose": str(densepose_path),
                "fused_form": str(fused_path),
            },
            "stages": {
                "artifact_tables": {"status": "pending", "custom_field": "keep"},
                "future_stage": {"status": "future"},
            },
        },
    )
    written_paths: list[Path] = []

    def fake_write_parquet(path: Path, rows: list[dict[str, object]]) -> int:
        written_paths.append(path)
        return len(rows)

    monkeypatch.setattr(artifact_tables, "write_parquet", fake_write_parquet)

    result = export_cv_tables(run_dir)

    assert written_paths == [
        run_dir / "poses.parquet",
        run_dir / "densepose.parquet",
        run_dir / "fused_form.parquet",
    ]
    assert result["exports"]["densepose_parquet"]["output"] == str(
        run_dir / "densepose.parquet"
    )
    assert result["exports"]["fused_form_parquet"]["output"] == str(
        run_dir / "fused_form.parquet"
    )
    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["artifact_tables"]["custom_field"] == "keep"
    assert manifest["stages"]["future_stage"] == {"status": "future"}
