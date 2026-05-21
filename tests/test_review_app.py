from __future__ import annotations

import json
from pathlib import Path

import pytest

from whodoirunlike.review_app import (
    ReviewAppConfig,
    build_cv_run_candidates_payload,
    densepose_job_status,
    densepose_setup_status,
    features_job_status,
    list_cv_runs,
    load_cv_run_payload,
    load_review_clips,
    load_subject_candidates,
    mask_job_status,
    pose_job_status,
    prepare_cv_run_from_review,
    save_annotation,
    save_cv_prompt,
    sam2_job_status,
    start_densepose_job,
    start_features_job,
    start_mask_job,
    start_pose_job,
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
                    "fused_form": str(run_dir / "fused_form.jsonl"),
                    "skeleton_render": str(run_dir / "skeleton_render.mp4"),
                    "masked_runner": str(run_dir / "masked_runner.mp4"),
                    "qa_overlay": str(run_dir / "qa_overlay.mp4"),
                    "fused_overlay": str(run_dir / "fused_overlay.mp4"),
                    "features": str(run_dir / "features.json"),
                    "form_features": str(run_dir / "form_features.json"),
                    "form_feature_arrays": str(run_dir / "form_features.npz"),
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
    assert saved_manifest["stages"]["detector_tracker"]["status"] == "pending_run"
    assert saved_manifest["stages"]["whole_runner_mask"]["status"] == "pending_tracker"
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


def test_save_cv_prompt_preserves_subject_candidate(tmp_path: Path) -> None:
    run_dir = write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    save_cv_prompt(
        config,
        "clip-001",
        {
            "selection": {
                "box": {"x": 0.2, "y": 0.3, "width": 0.1, "height": 0.4},
                "subject_candidate": {
                    "id": "sam31-4",
                    "index": 4,
                    "score": 0.87654,
                    "mask_area_ratio": 0.045678,
                    "center": {"x": 0.25, "y": 0.5},
                    "box": {"x": 0.2, "y": 0.3, "width": 0.1, "height": 0.4},
                },
            }
        },
    )
    saved_prompt = json.loads((run_dir / "person_prompt.json").read_text(encoding="utf-8"))

    assert saved_prompt["selection"]["type"] == "box"
    assert saved_prompt["selection"]["subject_candidate"]["id"] == "sam31-4"
    assert saved_prompt["selection"]["subject_candidate"]["score"] == 0.8765
    assert saved_prompt["selection"]["subject_candidate"]["center"] == {"x": 0.25, "y": 0.5}


def test_load_subject_candidates_defaults_to_empty(tmp_path: Path) -> None:
    write_cv_run(tmp_path)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    payload = load_subject_candidates(config, "clip-001")

    assert payload["cached"] is False
    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []


def test_build_cv_run_candidates_marks_prepared_and_preparable(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    source = tmp_path / "source.json"
    annotations = tmp_path / "annotations.json"
    write_source(source, video_a, video_b)
    annotations.write_text(
        json.dumps(
            {
                "version": 1,
                "annotations": {
                    "higher": {
                        "candidate_id": "higher",
                        "quality": "good",
                        "camera_angle": "side",
                        "start_seconds": 1.0,
                        "end_seconds": 3.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    write_cv_run(tmp_path, candidate_id="higher")
    config = ReviewAppConfig(
        source_path=source,
        annotations_path=annotations,
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    payload = build_cv_run_candidates_payload(config)
    higher = next(clip for clip in payload["clips"] if clip["candidate_id"] == "higher")
    lower = next(clip for clip in payload["clips"] if clip["candidate_id"] == "lower")

    assert payload["prepared"] == 1
    assert higher["cv_run_prepared"] is True
    assert higher["can_prepare"] is True
    assert higher["duration_seconds"] == 2.0
    assert lower["can_prepare"] is False


def test_prepare_cv_run_from_review_returns_existing_run_without_reset(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    annotations = tmp_path / "annotations.json"
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")
    write_source(source, video_a, video_b)
    annotations.write_text(json.dumps({"version": 1, "annotations": {}}), encoding="utf-8")
    run_dir = write_cv_run(tmp_path, candidate_id="higher")
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["pose"] = {"status": "complete"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = ReviewAppConfig(
        source_path=source,
        annotations_path=annotations,
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    payload = prepare_cv_run_from_review(config, "higher")

    assert payload["candidate_id"] == "higher"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["stages"]["pose"]["status"] == "complete"


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


def test_pose_job_status_defaults_to_idle() -> None:
    status = pose_job_status("pose-clip")

    assert status["status"] == "idle"
    assert status["stage"] == "pose"
    assert status["candidate_id"] == "pose-clip"


def test_start_pose_job_marks_stage_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = write_cv_run(tmp_path, candidate_id="pose-clip")
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    class DummyThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

    import whodoirunlike.review_app as review_app

    monkeypatch.setattr(review_app.threading, "Thread", DummyThread)

    status = start_pose_job(config, "pose-clip", pose_backend="mediapipe")

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["backend"] == "mediapipe_pose"
    assert manifest["stages"]["pose"]["status"] == "running"
    assert pose_job_status("pose-clip")["status"] == "running"


def test_start_densepose_job_requires_runner_mask(tmp_path: Path) -> None:
    write_cv_run(tmp_path, candidate_id="densepose-clip")
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    with pytest.raises(FileNotFoundError, match="Runner mask"):
        start_densepose_job(config, "densepose-clip")


def test_densepose_job_status_defaults_to_idle() -> None:
    status = densepose_job_status("densepose-clip")

    assert status["status"] == "idle"
    assert status["stage"] == "densepose"
    assert status["candidate_id"] == "densepose-clip"


def test_features_job_status_defaults_to_idle() -> None:
    status = features_job_status("feature-clip")

    assert status["status"] == "idle"
    assert status["stage"] == "features"
    assert status["candidate_id"] == "feature-clip"


def test_start_features_job_requires_fused_form(tmp_path: Path) -> None:
    run_dir = write_cv_run(tmp_path, candidate_id="feature-clip")
    (run_dir / "pose_landmarks.jsonl").write_text("", encoding="utf-8")
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    with pytest.raises(FileNotFoundError, match="Fused form"):
        start_features_job(config, "feature-clip")


def test_start_features_job_marks_form_feature_stage_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = write_cv_run(tmp_path, candidate_id="feature-clip")
    (run_dir / "pose_landmarks.jsonl").write_text("", encoding="utf-8")
    (run_dir / "fused_form.jsonl").write_text("", encoding="utf-8")
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    class DummyThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

    import whodoirunlike.review_app as review_app

    monkeypatch.setattr(review_app.threading, "Thread", DummyThread)

    status = start_features_job(config, "feature-clip")

    manifest = json.loads((run_dir / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["backend"] == "form_feature_compiler"
    assert manifest["stages"]["form_features"]["status"] == "running"


def test_densepose_setup_status_uses_default_model_files_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
    weights_path = tmp_path / "models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl"
    config_path.parent.mkdir(parents=True)
    weights_path.parent.mkdir(parents=True)
    config_path.write_text("_BASE_: Base-DensePose-RCNN-FPN.yaml\n", encoding="utf-8")
    weights_path.write_bytes(b"weights")
    monkeypatch.setattr("whodoirunlike.review_app.DENSEPOSE_DEFAULT_CONFIG", config_path)
    monkeypatch.setattr("whodoirunlike.review_app.DENSEPOSE_DEFAULT_WEIGHTS", weights_path)
    monkeypatch.delenv("DENSEPOSE_CONFIG", raising=False)
    monkeypatch.delenv("DENSEPOSE_WEIGHTS", raising=False)
    config = ReviewAppConfig(
        source_path=tmp_path / "source.json",
        annotations_path=tmp_path / "annotations.json",
        static_dir=tmp_path,
        repo_root=tmp_path,
        limit=2,
    )

    status = densepose_setup_status(config)

    assert status["config_path"] == str(config_path)
    assert status["weights"] == str(weights_path)
    assert status["using_defaults"] == {"config": True, "weights": True}
