from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from whodoirunlike.curation import (
    ShotInterval,
    WindowMetrics,
    partition_shots,
    propose_clip_windows,
    score_window,
    side_view_prior_for_bucket,
    write_curation_manifest,
)


def write_motion_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 60))
    assert writer.isOpened()
    for index in range(50):
        frame = np.full((60, 80, 3), 20, dtype=np.uint8)
        x = 8 + index
        cv2.rectangle(frame, (x, 18), (x + 10, 46), (210, 210, 210), -1)
        cv2.circle(frame, (x + 5, 50), 4, (230, 230, 230), -1)
        writer.write(frame)
    writer.release()


def test_partition_shots_uses_overlapping_windows_and_final_window() -> None:
    windows = partition_shots(
        [ShotInterval(0.0, 9.0, source="test")],
        window_seconds=4.0,
        step_seconds=2.0,
    )

    assert [(item.start_seconds, item.end_seconds) for item in windows] == [
        (0.0, 4.0),
        (2.0, 6.0),
        (4.0, 8.0),
        (5.0, 9.0),
    ]


def test_score_window_blends_visibility_and_penalizes_crowding() -> None:
    clean = WindowMetrics(
        visible_person_fraction=0.9,
        pose_visibility=0.8,
        track_continuity=0.7,
        runningness=0.75,
        side_view_prior=side_view_prior_for_bucket("side"),
        bbox_scale=0.55,
        crowd_penalty=0.0,
    )
    crowded = WindowMetrics(
        visible_person_fraction=0.9,
        pose_visibility=0.8,
        track_continuity=0.7,
        runningness=0.75,
        side_view_prior=side_view_prior_for_bucket("side"),
        bbox_scale=0.55,
        crowd_penalty=1.0,
    )

    assert score_window(clean) == 0.732
    assert score_window(crowded) == 0.682


def test_propose_clip_windows_writes_ranked_manifest_and_thumbnail(tmp_path: Path) -> None:
    source = tmp_path / "runner.mp4"
    write_motion_video(source)

    manifest = propose_clip_windows(
        source,
        top_k=2,
        view_bucket="diagonal",
        use_scenedetect=False,
        window_seconds=2.0,
        step_seconds=1.0,
        sample_fps=2.0,
        output_dir=tmp_path / "curation",
        write_thumbnails=True,
    )

    assert manifest["pipeline_goal"] == "identity_stable_runner_clip_proposal"
    assert manifest["windows_considered"] >= 1
    assert len(manifest["windows"]) == 2
    assert manifest["windows"][0]["rank"] == 1
    assert manifest["windows"][0]["score"] >= manifest["windows"][1]["score"]
    assert manifest["windows"][0]["score_components"]["sampled_frames"] > 0
    assert Path(manifest["windows"][0]["thumbnail_path"]).exists()


def test_write_curation_manifest_flattens_video_windows(tmp_path: Path) -> None:
    manifest = {
        "source_video": "/tmp/source.mp4",
        "windows": [
            {
                "window_id": "abc",
                "video_path": "/tmp/source.mp4",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "duration_seconds": 4.0,
                "shot_index": 0,
                "rank": 1,
                "score": 0.8,
                "score_components": {},
            }
        ],
    }

    output = tmp_path / "clip_windows.json"
    payload = write_curation_manifest(output, [manifest])
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert payload["windows"][0]["source_video"] == "/tmp/source.mp4"
    assert saved["pipeline_goal"] == "identity_stable_runner_clip_proposal"
