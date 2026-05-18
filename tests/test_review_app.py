from __future__ import annotations

import json
from pathlib import Path

import pytest

from whodoirunlike.review_app import (
    ReviewAppConfig,
    list_cv_runs,
    load_cv_run_payload,
    load_review_clips,
    mask_job_status,
    save_annotation,
    save_cv_prompt,
    sam2_job_status,
    start_mask_job,
    start_sam2_job,
)


def write_source(path: Path, video_a: Path, video_b: Path) -> None:
    rows = [
        {
            "candidate_id": "lower",
            "runner_name": "Runner Low",
            "title": "Lower score clip",
            "video_path": str(video_a),
            "duration_seconds_local": 100,
            "cv_score": 61,
            "score": 100,
            "view_count": 2_000,
        },
        {
            "candidate_id": "higher",
            "runner_name": "Runner High",
            "title": "Higher score clip",
            "video_path": str(video_b),
            "duration_seconds_local": 80,
            "cv_score": 88,
            "score": 95,
            "view_count": 1_000,
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_load_review_clips_sorts_by_cv_score_and_limits(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    source = tmp_path / "source.json"
    write_source(source, video_a, video_b)

    config = ReviewAppConfig(
        source_path=source,
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=1,
    )

    clips = load_review_clips(config)

    assert [clip["candidate_id"] for clip in clips] == ["higher"]


def test_save_annotation_clamps_and_swaps_segment_times(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    source = tmp_path / "source.json"
    annotations = tmp_path / "annotations.json"
    write_source(source, video_a, video_b)
    config = ReviewAppConfig(
        source_path=source,
        annotations_path=annotations,
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    annotation = save_annotation(
        config,
        {
            "candidate_id": "higher",
            "quality": "good",
            "camera_angle": "diagonal",
            "start_seconds": 90,
            "end_seconds": 12,
            "notes": "usable side view",
        },
    )

    assert annotation["quality"] == "good"
    assert annotation["camera_angle"] == "diagonal"
    assert annotation["start_seconds"] == 12
    assert annotation["end_seconds"] == 80
    assert json.loads(annotations.read_text(encoding="utf-8"))["annotations"]["higher"]["notes"] == (
        "usable side view"
    )


def test_save_annotation_rejects_unknown_camera_angle_value(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    source = tmp_path / "source.json"
    write_source(source, video_a, video_b)
    config = ReviewAppConfig(
        source_path=source,
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    with pytest.raises(ValueError, match="camera_angle"):
        save_annotation(
            config,
            {
                "candidate_id": "higher",
                "quality": "good",
                "camera_angle": "overhead",
            },
        )


def write_cv_run(root: Path, candidate_id: str = "clip-001") -> Path:
    run_dir = root / "artifacts" / "cv_runs" / candidate_id
    run_dir.mkdir(parents=True)
    (run_dir / "source_segment.mp4").write_bytes(b"video")
    (run_dir / "prompt_frame.jpg").write_bytes(b"image")
    (run_dir / "person_prompt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": candidate_id,
                "frame": {"width": 1920, "height": 1080},
                "selection": {
                    "type": "unset",
                    "positive_points": [],
                    "negative_points": [],
                    "box": None,
                    "mask_path": None,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": candidate_id,
                "runner_name": "Runner High",
                "source": {"title": "Fast race", "channel": "Test Channel"},
                "review": {
                    "camera_angle": "side",
                    "duration_seconds": 4.5,
                    "primary_bucket": "5k_10k",
                },
                "paths": {
                    "source_segment": str(run_dir / "source_segment.mp4"),
                    "prompt_frame": str(run_dir / "prompt_frame.jpg"),
                    "person_prompt": str(run_dir / "person_prompt.json"),
                    "runner_mask": str(run_dir / "runner_mask.mp4"),
                    "pose_landmarks": str(run_dir / "pose_landmarks.jsonl"),
                    "densepose": str(run_dir / "densepose.jsonl"),
                    "skeleton_render": str(run_dir / "skeleton_render.mp4"),
                    "masked_runner": str(run_dir / "masked_runner.mp4"),
                    "qa_overlay": str(run_dir / "qa_overlay.mp4"),
                    "features": str(run_dir / "features.json"),
                },
                "stages": {
                    "person_prompt": {"status": "needs_selection"},
                    "whole_runner_mask": {"status": "pending_prompt"},
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_cv_run_payload_serves_artifact_urls(tmp_path: Path) -> None:
    write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    payload = load_cv_run_payload(config, "clip-001")

    assert payload["candidate_id"] == "clip-001"
    assert payload["artifacts"]["prompt_frame"]["exists"] is True
    assert payload["artifacts"]["prompt_frame"]["url"] == "/cv-artifacts/clip-001/prompt_frame.jpg"
    assert payload["artifacts"]["runner_mask"]["exists"] is False
    assert payload["stages"]["person_prompt"]["status"] == "needs_selection"


def test_save_cv_prompt_updates_selection_and_stage_state(tmp_path: Path) -> None:
    run_dir = write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    payload = save_cv_prompt(
        config,
        "clip-001",
        {
            "selection": {
                "positive_points": [{"x": 1.4, "y": 0.5}],
                "negative_points": [{"x": 0.1, "y": -0.2}],
                "box": None,
            }
        },
    )
    saved_prompt = json.loads((run_dir / "person_prompt.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))

    assert saved_prompt["selection"]["type"] == "point"
    assert saved_prompt["selection"]["positive_points"] == [{"x": 1.0, "y": 0.5}]
    assert saved_prompt["selection"]["negative_points"] == [{"x": 0.1, "y": 0.0}]
    assert saved_manifest["stages"]["person_prompt"]["status"] == "ready"
    assert saved_manifest["stages"]["whole_runner_mask"]["status"] == "pending_run"
    assert payload["stages"]["person_prompt"]["status"] == "ready"
    assert list_cv_runs(config)[0]["prompt_ready"] is True


def test_save_cv_prompt_preserves_many_clicks(tmp_path: Path) -> None:
    run_dir = write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )
    positive_points = [{"x": index / 100, "y": 0.5} for index in range(25)]
    negative_points = [{"x": 0.5, "y": index / 100} for index in range(35)]

    save_cv_prompt(
        config,
        "clip-001",
        {
            "selection": {
                "positive_points": positive_points,
                "negative_points": negative_points,
            }
        },
    )
    saved_prompt = json.loads((run_dir / "person_prompt.json").read_text(encoding="utf-8"))

    assert len(saved_prompt["selection"]["positive_points"]) == 25
    assert len(saved_prompt["selection"]["negative_points"]) == 35


def test_sam2_job_status_defaults_to_idle() -> None:
    status = sam2_job_status("clip-001")

    assert status["status"] == "idle"
    assert status["candidate_id"] == "clip-001"


def test_mask_job_status_defaults_to_idle() -> None:
    status = mask_job_status("clip-002")

    assert status["status"] == "idle"
    assert status["backend"] is None
    assert status["options"] == {}
    assert status["progress"] is None
    assert status["candidate_id"] == "clip-002"


def test_start_sam2_job_requires_checkpoint(tmp_path: Path) -> None:
    write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        start_sam2_job(config, "clip-001")


def test_start_mask_job_rejects_unknown_backend(tmp_path: Path) -> None:
    write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    with pytest.raises(ValueError, match="backend"):
        start_mask_job(config, "clip-001", backend="sam9")


def test_start_mask_job_rejects_unknown_quality_mode(tmp_path: Path) -> None:
    write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    with pytest.raises(ValueError, match="quality_mode"):
        start_mask_job(config, "clip-001", backend="sam31_mlx", options={"quality_mode": "ultra"})
