from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from whodoirunlike.pose_runner import (
    PoseCandidate,
    choose_pose_candidate,
    hard_mask_frame,
    pose_row,
    summarize_pose,
    update_manifest_after_pose,
)
from whodoirunlike.sam2_runner import write_json


def _landmarks(visibility: float = 0.8) -> list[SimpleNamespace]:
    points = []
    for index in range(33):
        points.append(
            SimpleNamespace(
                x=0.35 + (index % 4) * 0.04,
                y=0.2 + (index % 8) * 0.07,
                z=0.0,
                visibility=visibility,
                presence=visibility,
            )
        )
    points[11].x = 0.42
    points[11].y = 0.28
    points[12].x = 0.5
    points[12].y = 0.29
    points[23].x = 0.43
    points[23].y = 0.55
    points[24].x = 0.51
    points[24].y = 0.56
    points[27].x = 0.37
    points[27].y = 0.84
    points[28].x = 0.56
    points[28].y = 0.82
    return points


def test_choose_pose_candidate_prefers_prompt_overlap_before_continuity() -> None:
    far_candidate = PoseCandidate(
        index=0,
        landmarks=_landmarks(),
        world_landmarks=None,
        visibility_mean=0.96,
        bbox={"x": 0.65, "y": 0.2, "width": 0.2, "height": 0.55},
        score=0.96,
    )
    prompt_candidate = PoseCandidate(
        index=1,
        landmarks=_landmarks(),
        world_landmarks=None,
        visibility_mean=0.78,
        bbox={"x": 0.2, "y": 0.2, "width": 0.25, "height": 0.58},
        score=0.78,
    )

    selected = choose_pose_candidate(
        [far_candidate, prompt_candidate],
        prompt_box={"x": 0.18, "y": 0.18, "width": 0.3, "height": 0.65},
    )

    assert selected is prompt_candidate


def test_choose_pose_candidate_rejects_candidates_outside_runner_mask() -> None:
    wrong_runner = PoseCandidate(
        index=0,
        landmarks=_landmarks(),
        world_landmarks=None,
        visibility_mean=0.98,
        bbox={"x": 0.18, "y": 0.2, "width": 0.18, "height": 0.55},
        score=0.98,
    )
    masked_runner = PoseCandidate(
        index=1,
        landmarks=_landmarks(),
        world_landmarks=None,
        visibility_mean=0.72,
        bbox={"x": 0.65, "y": 0.24, "width": 0.16, "height": 0.58},
        score=0.72,
    )

    selected = choose_pose_candidate(
        [wrong_runner, masked_runner],
        previous_bbox={"x": 0.18, "y": 0.2, "width": 0.18, "height": 0.55},
        mask_box={"x": 0.64, "y": 0.22, "width": 0.19, "height": 0.62},
    )

    assert selected is masked_runner


def test_choose_pose_candidate_returns_none_when_all_candidates_miss_mask() -> None:
    wrong_runner = PoseCandidate(
        index=0,
        landmarks=_landmarks(),
        world_landmarks=None,
        visibility_mean=0.98,
        bbox={"x": 0.18, "y": 0.2, "width": 0.18, "height": 0.55},
        score=0.98,
    )

    assert (
        choose_pose_candidate(
            [wrong_runner],
            mask_box={"x": 0.64, "y": 0.22, "width": 0.19, "height": 0.62},
        )
        is None
    )


def test_hard_mask_frame_blacks_out_background() -> None:
    frame = np.full((4, 5, 3), 100, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255

    masked = hard_mask_frame(frame, mask)

    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[1, 2].tolist() == [100, 100, 100]


def test_pose_row_records_usable_landmarks_and_world_landmarks() -> None:
    candidate = PoseCandidate(
        index=2,
        landmarks=_landmarks(),
        world_landmarks=_landmarks(),
        visibility_mean=0.82,
        bbox={"x": 0.32, "y": 0.18, "width": 0.3, "height": 0.7},
        score=0.82,
    )

    row = pose_row(
        frame_index=12,
        fps=30.0,
        frame_width=1920,
        frame_height=1080,
        candidate=candidate,
        candidate_count=3,
    )

    assert row["usable"] is True
    assert row["selected_pose_index"] == 2
    assert row["landmarks"][11]["name"] == "left_shoulder"
    assert row["world_landmarks"][27]["name"] == "left_ankle"
    assert row["time_seconds"] == 0.4


def test_summarize_pose_writes_quality_and_metrics() -> None:
    rows = [
        pose_row(
            frame_index=index,
            fps=10.0,
            frame_width=64,
            frame_height=48,
            candidate=PoseCandidate(
                index=0,
                landmarks=_landmarks(),
                world_landmarks=None,
                visibility_mean=0.75,
                bbox={"x": 0.3, "y": 0.2, "width": 0.3, "height": 0.68},
                score=0.75,
            ),
            candidate_count=1,
        )
        for index in range(3)
    ]

    summary = summarize_pose(rows, input_video=Path("masked_runner.mp4"), model_variant="heavy", fps=10.0)

    assert summary["quality"]["pose_hit_rate"] == 1.0
    assert summary["quality"]["usable_rate"] == 1.0
    assert summary["model"]["variant"] == "heavy"
    assert "torso_lean_mean_deg" in summary["explainability_metrics"]


def test_update_manifest_after_pose_marks_outputs_complete(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cv_run_manifest.json"
    write_json(
        manifest_path,
        {
            "version": 1,
            "paths": {},
            "stages": {
                "pose": {"status": "pending"},
                "renders": {"status": "pending"},
                "features": {"status": "pending"},
            },
        },
    )

    update_manifest_after_pose(
        manifest_path,
        pose_landmarks_path=tmp_path / "pose_landmarks.jsonl",
        skeleton_render_path=tmp_path / "skeleton_render.mp4",
        features_path=tmp_path / "features.json",
        result={"quality": {"pose_hit_rate": 0.7, "usable_rate": 0.6, "visibility_mean": 0.8}},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stages"]["pose"]["status"] == "complete"
    assert manifest["stages"]["pose"]["summary"]["pose_hit_rate"] == 0.7
    assert manifest["stages"]["renders"]["status"] == "partial_complete"
    assert manifest["stages"]["features"]["status"] == "complete"
