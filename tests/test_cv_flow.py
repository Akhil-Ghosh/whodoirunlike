from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from whodoirunlike.cv_flow import prepare_single_clip_cv_run


def write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(30):
        frame = np.full((48, 64, 3), 24 + index, dtype=np.uint8)
        cv2.circle(frame, (20 + index % 20, 24), 8, (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def test_prepare_single_clip_cv_run_creates_artifact_scaffold(tmp_path: Path) -> None:
    source_video = tmp_path / "source.mp4"
    write_test_video(source_video)
    manifest_path = tmp_path / "manifest.json"
    annotations_path = tmp_path / "annotations.json"

    manifest_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "clip-001",
                    "runner_name": "Test Runner",
                    "title": "Test race clip",
                    "channel": "Test Channel",
                    "url": "https://www.youtube.com/watch?v=test",
                    "video_path": str(source_video),
                    "primary_bucket": "5k_10k",
                }
            ]
        ),
        encoding="utf-8",
    )
    annotations_path.write_text(
        json.dumps(
            {
                "version": 1,
                "annotations": {
                    "clip-001": {
                        "candidate_id": "clip-001",
                        "quality": "good",
                        "camera_angle": "side",
                        "start_seconds": 0.2,
                        "end_seconds": 1.8,
                        "notes": "clean side view",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_single_clip_cv_run(
        candidate_id="clip-001",
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        output_root=tmp_path / "cv_runs",
    )

    run_dir = tmp_path / "cv_runs" / "clip-001"
    prompt_payload = json.loads((run_dir / "person_prompt.json").read_text(encoding="utf-8"))
    track_seed = json.loads((run_dir / "track_seed.json").read_text(encoding="utf-8"))
    view_bucket = json.loads((run_dir / "view_bucket.json").read_text(encoding="utf-8"))

    assert (run_dir / "source_segment.mp4").exists()
    assert (run_dir / "prompt_frame.jpg").exists()
    assert (run_dir / "cv_run_manifest.json").exists()
    assert (run_dir / "track_seed.json").exists()
    assert (run_dir / "view_bucket.json").exists()
    assert prompt_payload["selection"]["type"] == "unset"
    assert track_seed["tracker"]["preferred"] == "BoT-SORT"
    assert view_bucket["view_bucket"] == "side"
    assert prompt_payload["frame"]["width"] == 64
    assert manifest["review"]["camera_angle"] == "side"
    assert manifest["paths"]["target_prompt"] == manifest["paths"]["person_prompt"]
    assert manifest["paths"]["tracklets_jsonl"].endswith("tracklets.jsonl")
    assert manifest["paths"]["reid_jsonl"].endswith("reid.jsonl")
    assert manifest["stages"]["detector_tracker"]["status"] == "pending_prompt"
    assert manifest["stages"]["whole_runner_mask"]["status"] == "pending_prompt"
    assert manifest["stages"]["densepose"]["status"] == "pending_runner_mask"
