from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from whodoirunlike.identity_runner import prompt_initial_box, run_identity_tracking


def write_identity_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    for index in range(18):
        frame = np.full((64, 96, 3), 28, dtype=np.uint8)
        x = 22 + index
        cv2.rectangle(frame, (x, 18), (x + 14, 48), (48, 180, 220), -1)
        cv2.circle(frame, (x + 7, 52), 3, (48, 180, 220), -1)
        writer.write(frame)
    writer.release()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_identity_run(run_dir: Path) -> None:
    video_path = run_dir / "source_segment.mp4"
    write_identity_video(video_path)
    prompt_path = run_dir / "person_prompt.json"
    track_seed_path = run_dir / "track_seed.json"
    write_json(
        prompt_path,
        {
            "version": 1,
            "candidate_id": "identity-clip",
            "frame": {"frame_index": 9, "width": 96, "height": 64},
            "selection": {
                "type": "box",
                "positive_points": [{"x": 0.395833, "y": 0.515625}],
                "negative_points": [],
                "box": {"x": 0.322917, "y": 0.28125, "width": 0.145833, "height": 0.46875},
                "mask_path": None,
            },
        },
    )
    write_json(
        track_seed_path,
        {
            "version": 1,
            "candidate_id": "identity-clip",
            "status": "pending_detector_tracker",
            "reid": {"cosine_accept": 0.65, "cosine_recover": 0.58},
        },
    )
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": "identity-clip",
            "runner_name": "Identity Runner",
            "paths": {
                "source_segment": str(video_path),
                "person_prompt": str(prompt_path),
                "track_seed": str(track_seed_path),
                "tracklets": str(run_dir / "tracklets.parquet"),
                "tracklets_jsonl": str(run_dir / "tracklets.jsonl"),
                "reid": str(run_dir / "reid.parquet"),
                "reid_jsonl": str(run_dir / "reid.jsonl"),
                "qc_metrics": str(run_dir / "qc_metrics.json"),
            },
            "stages": {
                "detector_tracker": {"status": "pending_run"},
                "whole_runner_mask": {"status": "pending_tracker"},
            },
        },
    )


def test_prompt_initial_box_uses_positive_point_when_box_is_absent() -> None:
    prompt = {
        "selection": {
            "type": "point",
            "positive_points": [{"x": 0.5, "y": 0.5}],
            "negative_points": [],
            "box": None,
        }
    }

    x, y, width, height = prompt_initial_box(prompt, 100, 80)

    assert width == 16
    assert height == 34
    assert x == 42
    assert y == 25


def test_run_identity_tracking_writes_parquet_and_updates_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "cv_runs" / "identity-clip"
    run_dir.mkdir(parents=True)
    write_identity_run(run_dir)

    result = run_identity_tracking(run_dir=run_dir)

    assert result["status"] == "complete"
    assert result["backend"] == "prompt_template_tracker_v1"
    assert result["frame_count"] == 18
    assert result["metrics"]["target_identity_stability_rate"] >= 0.75
    assert (run_dir / "tracklets.parquet").exists()
    assert (run_dir / "reid.parquet").exists()
    assert (run_dir / "tracklets.jsonl").exists()
    assert (run_dir / "qc_metrics.json").exists()

    track_table = pq.read_table(run_dir / "tracklets.parquet")
    reid_table = pq.read_table(run_dir / "reid.parquet")
    assert track_table.num_rows == 18
    assert reid_table.num_rows == 18

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    track_seed = json.loads((run_dir / "track_seed.json").read_text(encoding="utf-8"))
    qc_metrics = json.loads((run_dir / "qc_metrics.json").read_text(encoding="utf-8"))

    assert manifest["stages"]["detector_tracker"]["status"] == "complete"
    assert manifest["stages"]["whole_runner_mask"]["status"] == "pending_run"
    assert track_seed["target_track_id"] == 1
    assert track_seed["status"] == "complete"
    assert qc_metrics["identity"]["frame_count"] == 18
