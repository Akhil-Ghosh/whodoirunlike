from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from whodoirunlike import full_pipeline
from whodoirunlike.processing_telemetry import ProcessingTelemetry


def test_parallel_pose_densepose_requires_isolated_mmpose_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parallel_pose_densepose requires an mmpose backend"):
        full_pipeline.run_full_cv_pipeline(
            run_dir=tmp_path,
            pose_backend="mediapipe",
            parallel_pose_densepose=True,
        )


def test_finalize_render_artifact_pointers_keeps_pose_and_canonical_qa_distinct(
    tmp_path: Path,
) -> None:
    canonical_qa = tmp_path / "qa_overlay.mp4"
    pose_qa = tmp_path / "pose_qa_overlay.mp4"
    canonical_qa.write_bytes(b"densepose-qa")
    pose_qa.write_bytes(b"pose-qa")
    (tmp_path / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "paths": {
                    "qa_overlay": str(canonical_qa),
                    "pose_qa_overlay": str(pose_qa),
                },
                "stages": {
                    "renders": {
                        "status": "partial_complete",
                        "qa_overlay": str(pose_qa),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    full_pipeline._finalize_render_artifact_pointers(tmp_path)

    manifest = json.loads((tmp_path / "cv_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["renders"]["qa_overlay"] == str(canonical_qa)
    assert manifest["stages"]["renders"]["pose_qa_overlay"] == str(pose_qa)


def test_full_pipeline_can_fork_pose_and_densepose_then_join_before_fusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pose_started = threading.Event()
    densepose_started = threading.Event()
    pose_finished = threading.Event()
    densepose_finished = threading.Event()

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_sam31_mlx_mask", lambda **_: {"status": "complete"})

    def run_pose(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["isolate_qa_overlay"] is True
        pose_started.set()
        assert densepose_started.wait(timeout=1.0), "DensePose did not overlap pose"
        pose_finished.set()
        return {"status": "complete", "runner": "pose"}

    def run_densepose(**_: Any) -> dict[str, Any]:
        densepose_started.set()
        assert pose_started.wait(timeout=1.0), "Pose did not overlap DensePose"
        densepose_finished.set()
        return {"status": "complete", "runner": "densepose"}

    def run_fusion(**_: Any) -> dict[str, Any]:
        assert pose_finished.is_set()
        assert densepose_finished.is_set()
        return {"status": "complete"}

    monkeypatch.setattr("whodoirunlike.mmpose_runner.run_mmpose_pose", run_pose)
    monkeypatch.setattr("whodoirunlike.densepose_runner.run_densepose", run_densepose)
    monkeypatch.setattr(full_pipeline, "run_fused_form", run_fusion)
    monkeypatch.setattr(full_pipeline, "compile_form_features", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "export_cv_tables", lambda *_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", lambda *_: {"status": "complete"})

    result = full_pipeline.run_full_cv_pipeline(
        run_dir=tmp_path,
        pose_backend="mmpose_rtmpose_l_384",
        parallel_pose_densepose=True,
    )

    assert [step["stage"] for step in result["steps"]] == [
        "identity",
        "mask",
        "pose",
        "densepose",
        "fusion",
        "features",
        "artifact_tables",
        "qc",
    ]


def test_full_pipeline_overlaps_sam_presentation_with_pose_and_densepose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pose_started = threading.Event()
    densepose_started = threading.Event()
    mask_presentation_finished = threading.Event()

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})

    def run_mask(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["render_qa_overlay"] is False
        ready = kwargs["runner_mask_ready_callback"]
        assert callable(ready)
        ready()
        assert pose_started.wait(timeout=1.0), "Pose waited for mask presentation"
        assert densepose_started.wait(timeout=1.0), "DensePose waited for mask presentation"
        mask_presentation_finished.set()
        return {"status": "complete", "backend": "sam31_gpu"}

    def run_pose(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["isolate_qa_overlay"] is True
        assert not mask_presentation_finished.is_set()
        pose_started.set()
        return {"status": "complete"}

    def run_densepose(**_: Any) -> dict[str, Any]:
        assert not mask_presentation_finished.is_set()
        densepose_started.set()
        return {"status": "complete"}

    def run_fusion(**_: Any) -> dict[str, Any]:
        assert mask_presentation_finished.is_set()
        return {"status": "complete"}

    monkeypatch.setattr(full_pipeline, "_run_mask_stage", run_mask)
    monkeypatch.setattr("whodoirunlike.mmpose_runner.run_mmpose_pose", run_pose)
    monkeypatch.setattr("whodoirunlike.densepose_runner.run_densepose", run_densepose)
    monkeypatch.setattr(full_pipeline, "run_fused_form", run_fusion)
    monkeypatch.setattr(full_pipeline, "compile_form_features", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "export_cv_tables", lambda *_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", lambda *_: {"status": "complete"})

    result = full_pipeline.run_full_cv_pipeline(
        run_dir=tmp_path,
        mask_backend="sam31_gpu",
        pose_backend="mmpose_rtmpose_l_384",
        parallel_mask_presentation=True,
        parallel_pose_densepose=True,
    )

    assert [step["stage"] for step in result["steps"]] == [
        "identity",
        "mask",
        "pose",
        "densepose",
        "fusion",
        "features",
        "artifact_tables",
        "qc",
    ]


def test_parallel_mask_presentation_requires_sam_gpu_mmpose_and_densepose(
    tmp_path: Path,
) -> None:
    for kwargs in (
        {"mask_backend": "sam31_mlx", "pose_backend": "mmpose_rtmpose_l_384"},
        {"mask_backend": "sam31_gpu", "pose_backend": "mediapipe"},
        {
            "mask_backend": "sam31_gpu",
            "pose_backend": "mmpose_rtmpose_l_384",
            "skip_densepose": True,
        },
    ):
        with pytest.raises(ValueError, match="parallel_mask_presentation requires"):
            full_pipeline.run_full_cv_pipeline(
                run_dir=tmp_path,
                parallel_mask_presentation=True,
                **kwargs,
            )


def test_parallel_mask_presentation_propagates_failure_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})
    monkeypatch.setattr(
        full_pipeline,
        "_run_mask_stage",
        lambda **_: (_ for _ in ()).throw(RuntimeError("mask failed before readiness")),
    )

    with pytest.raises(RuntimeError, match="mask failed before readiness"):
        full_pipeline.run_full_cv_pipeline(
            run_dir=tmp_path,
            mask_backend="sam31_gpu",
            pose_backend="mmpose_rtmpose_l_384",
            parallel_mask_presentation=True,
            parallel_pose_densepose=True,
        )


def test_parallel_mask_presentation_failure_after_readiness_blocks_fusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pose_started = threading.Event()
    densepose_started = threading.Event()
    fusion_called = False
    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})

    def run_mask(**kwargs: Any) -> dict[str, Any]:
        kwargs["runner_mask_ready_callback"]()
        assert pose_started.wait(timeout=1.0)
        assert densepose_started.wait(timeout=1.0)
        raise RuntimeError("mask presentation failed")

    def run_pose(**_: Any) -> dict[str, Any]:
        pose_started.set()
        return {"status": "complete"}

    def run_densepose(**_: Any) -> dict[str, Any]:
        densepose_started.set()
        return {"status": "complete"}

    def run_fusion(**_: Any) -> dict[str, Any]:
        nonlocal fusion_called
        fusion_called = True
        return {"status": "complete"}

    monkeypatch.setattr(full_pipeline, "_run_mask_stage", run_mask)
    monkeypatch.setattr("whodoirunlike.mmpose_runner.run_mmpose_pose", run_pose)
    monkeypatch.setattr("whodoirunlike.densepose_runner.run_densepose", run_densepose)
    monkeypatch.setattr(full_pipeline, "run_fused_form", run_fusion)

    with pytest.raises(RuntimeError, match="mask presentation failed"):
        full_pipeline.run_full_cv_pipeline(
            run_dir=tmp_path,
            mask_backend="sam31_gpu",
            pose_backend="mmpose_rtmpose_l_384",
            parallel_mask_presentation=True,
            parallel_pose_densepose=True,
        )

    assert fusion_called is False


def test_parallel_mask_and_analysis_failures_are_both_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pose_started = threading.Event()
    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})

    def run_mask(**kwargs: Any) -> dict[str, Any]:
        kwargs["runner_mask_ready_callback"]()
        assert pose_started.wait(timeout=1.0)
        raise RuntimeError("presentation boom")

    def run_pose(**_: Any) -> dict[str, Any]:
        pose_started.set()
        raise ValueError("pose boom")

    monkeypatch.setattr(full_pipeline, "_run_mask_stage", run_mask)
    monkeypatch.setattr("whodoirunlike.mmpose_runner.run_mmpose_pose", run_pose)
    monkeypatch.setattr(
        "whodoirunlike.densepose_runner.run_densepose",
        lambda **_: (_ for _ in ()).throw(RuntimeError("densepose boom")),
    )

    with pytest.raises(ExceptionGroup) as caught:
        full_pipeline.run_full_cv_pipeline(
            run_dir=tmp_path,
            mask_backend="sam31_gpu",
            pose_backend="mmpose_rtmpose_l_384",
            parallel_mask_presentation=True,
            parallel_pose_densepose=True,
        )

    assert [str(error) for error in caught.value.exceptions] == [
        "pose boom",
        "densepose boom",
        "presentation boom",
    ]


def test_full_pipeline_can_fan_out_independent_post_fusion_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = {name: threading.Event() for name in ("features", "tables", "qc")}

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_sam31_mlx_mask", lambda **_: {"status": "complete"})
    monkeypatch.setattr(
        "whodoirunlike.pose_runner.run_pose_landmarks",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        "whodoirunlike.densepose_runner.run_densepose",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(full_pipeline, "run_fused_form", lambda **_: {"status": "complete"})

    def overlapping(name: str) -> dict[str, Any]:
        started[name].set()
        assert all(event.wait(timeout=1.0) for event in started.values())
        return {"status": "complete", "runner": name}

    monkeypatch.setattr(full_pipeline, "compile_form_features", lambda **_: overlapping("features"))
    monkeypatch.setattr(full_pipeline, "export_cv_tables", lambda *_: overlapping("tables"))
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", lambda *_: overlapping("qc"))

    result = full_pipeline.run_full_cv_pipeline(
        run_dir=tmp_path,
        pose_backend="mediapipe",
        parallel_post_fusion=True,
    )

    results = {step["stage"]: step["result"] for step in result["steps"]}
    assert results["features"]["runner"] == "features"
    assert results["artifact_tables"]["runner"] == "tables"
    assert results["qc"]["runner"] == "qc"


def test_full_pipeline_promotes_private_pose_qa_when_densepose_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pose_qa = tmp_path / "pose_qa_overlay.mp4"
    canonical_qa = tmp_path / "qa_overlay.mp4"
    densepose = tmp_path / "densepose.jsonl"
    (tmp_path / "cv_run_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "paths": {
                    "pose_qa_overlay": str(pose_qa),
                    "qa_overlay": str(canonical_qa),
                    "densepose": str(densepose),
                },
                "stages": {"densepose": {"status": "pending"}},
            }
        ),
        encoding="utf-8",
    )
    canonical_qa.write_bytes(b"mask-stage-overlay")
    normalized: list[Path] = []
    monkeypatch.setattr(full_pipeline, "make_browser_playable_mp4", normalized.append)
    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_sam31_mlx_mask", lambda **_: {"status": "complete"})

    def run_pose(**_: Any) -> dict[str, Any]:
        pose_qa.write_bytes(b"pose-stage-overlay")
        return {"status": "complete"}

    def run_fusion(**_: Any) -> dict[str, Any]:
        assert canonical_qa.read_bytes() == b"pose-stage-overlay"
        return {"status": "complete"}

    monkeypatch.setattr("whodoirunlike.mmpose_runner.run_mmpose_pose", run_pose)
    monkeypatch.setattr(full_pipeline, "run_fused_form", run_fusion)
    monkeypatch.setattr(full_pipeline, "compile_form_features", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "export_cv_tables", lambda *_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", lambda *_: {"status": "complete"})

    full_pipeline.run_full_cv_pipeline(
        run_dir=tmp_path,
        pose_backend="mmpose_rtmpose_l_384",
        skip_densepose=True,
    )
    assert normalized == [canonical_qa]


def test_densepose_runtime_kwargs_resolve_env_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "densepose.yaml"
    weights_path = tmp_path / "densepose.pkl"
    monkeypatch.setenv("DENSEPOSE_CONFIG", str(config_path))
    monkeypatch.setenv("DENSEPOSE_WEIGHTS", str(weights_path))
    monkeypatch.setenv("DENSEPOSE_DEVICE", "cuda")

    kwargs = full_pipeline._densepose_runtime_kwargs()

    assert kwargs == {
        "config_path": config_path.resolve(),
        "weights_path": str(weights_path.resolve()),
        "device": "cuda",
    }


def test_full_pipeline_passes_densepose_runtime_and_stops_on_densepose_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "densepose.yaml"
    weights_path = tmp_path / "densepose.pkl"
    captured: dict[str, Any] = {}

    monkeypatch.setenv("DENSEPOSE_CONFIG", str(config_path))
    monkeypatch.setenv("DENSEPOSE_WEIGHTS", str(weights_path))
    monkeypatch.setenv("DENSEPOSE_DEVICE", "cuda")
    monkeypatch.setattr(
        full_pipeline,
        "run_identity_tracking",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        full_pipeline,
        "run_sam31_mlx_mask",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        "whodoirunlike.pose_runner.run_pose_landmarks",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(
        full_pipeline,
        "run_fused_form",
        lambda **_: pytest.fail("fusion should not run after DensePose fails"),
    )

    def fake_run_densepose(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "failed", "error": "densepose boom"}

    monkeypatch.setattr("whodoirunlike.densepose_runner.run_densepose", fake_run_densepose)

    with pytest.raises(RuntimeError, match="densepose boom"):
        full_pipeline.run_full_cv_pipeline(run_dir=tmp_path, pose_backend="mediapipe")

    assert captured["run_dir"] == tmp_path
    assert captured["config_path"] == config_path.resolve()
    assert captured["weights_path"] == str(weights_path.resolve())
    assert captured["device"] == "cuda"


def test_full_pipeline_skip_densepose_uses_configured_path_and_records_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "candidate-1"
    densepose_path = tmp_path / "custom-artifacts" / "body-map.jsonl"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "candidate-1",
                "paths": {"densepose": str(densepose_path)},
                "stages": {
                    "densepose": {"status": "pending_runner_mask", "custom_field": "keep"},
                    "future_stage": {"status": "future"},
                },
                "custom_manifest_field": {"keep": True},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_sam31_mlx_mask", lambda **_: {"status": "complete"})
    monkeypatch.setattr(
        "whodoirunlike.pose_runner.run_pose_landmarks",
        lambda **_: {"status": "complete"},
    )
    monkeypatch.setattr(full_pipeline, "run_fused_form", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "compile_form_features", lambda **_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "export_cv_tables", lambda *_: {"status": "complete"})
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", lambda *_: {"status": "complete"})

    result = full_pipeline.run_full_cv_pipeline(
        run_dir=run_dir,
        pose_backend="mediapipe",
        skip_densepose=True,
    )

    assert densepose_path.read_text(encoding="utf-8") == ""
    assert not (run_dir / "densepose.jsonl").exists()
    assert next(step for step in result["steps"] if step["stage"] == "densepose")["result"] == {
        "status": "skipped"
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["stages"]["densepose"] == {
        "status": "skipped",
        "custom_field": "keep",
        "output": str(densepose_path),
    }
    assert manifest["stages"]["future_stage"] == {"status": "future"}
    assert manifest["custom_manifest_field"] == {"keep": True}


def test_full_pipeline_emits_canonical_stage_and_subspan_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MMPOSE_DEVICE", "cuda")
    monkeypatch.setenv("RTMW_RUNTIME_BACKEND", "onnxruntime")
    current = [10.0]

    def clock() -> float:
        return current[0]

    telemetry = ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="11111111-1111-4111-8111-111111111111",
        local_path=tmp_path / "events.jsonl",
        monotonic_clock=clock,
        wall_clock=lambda: datetime(2026, 7, 9, tzinfo=timezone.utc),
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
    )

    def runner(phase: str, *, frame_count: int = 12) -> Any:
        def run(**kwargs: Any) -> dict[str, Any]:
            callback = kwargs.get("progress_callback")
            assert callback is not None
            callback(
                {
                    "phase": phase,
                    "processed_frames": 0,
                    "total_frames": frame_count,
                }
            )
            current[0] += 1.0
            callback(
                {
                    "phase": "completed",
                    "processed_frames": frame_count,
                    "total_frames": frame_count,
                }
            )
            return {
                "status": "complete",
                "frame_count": frame_count,
                "elapsed_seconds": 1.0,
            }

        return run

    def unreported_result(*_: Any, **__: Any) -> dict[str, Any]:
        current[0] += 0.5
        return {"status": "complete"}

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", runner("loading_model"))
    monkeypatch.setattr(full_pipeline, "run_sam31_mlx_mask", runner("running_sam31"))
    monkeypatch.setattr("whodoirunlike.mmpose_runner.run_mmpose_pose", runner("detecting_pose"))
    monkeypatch.setattr("whodoirunlike.densepose_runner.run_densepose", runner("running_densepose"))
    monkeypatch.setattr(full_pipeline, "run_fused_form", runner("rendering"))
    monkeypatch.setattr(full_pipeline, "compile_form_features", runner("compiling_features"))
    monkeypatch.setattr(full_pipeline, "export_cv_tables", unreported_result)
    monkeypatch.setattr(full_pipeline, "run_qc_metrics", unreported_result)

    result = full_pipeline.run_full_cv_pipeline(
        run_dir=tmp_path,
        pose_backend="mmpose_rtmpose_l_384",
        telemetry=telemetry,
    )

    assert [step["stage"] for step in result["steps"]] == [
        "identity",
        "mask",
        "pose",
        "densepose",
        "fusion",
        "features",
        "artifact_tables",
        "qc",
    ]
    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    canonical_stages = [
        "target_tracking",
        "runner_mask",
        "pose_sequence",
        "densepose_body_map",
        "fused_form_signal",
        "form_feature_compilation",
        "artifact_table_export",
        "quality_control",
    ]
    assert [
        event["stage"] for event in events if event["event_type"] == "stage_started"
    ] == canonical_stages
    assert [
        event["stage"] for event in events if event["event_type"] == "stage_completed"
    ] == canonical_stages
    assert {
        event["span"] for event in events if event["event_type"] == "span_started"
    } >= {"model_load", "inference", "render", "postprocess", "write"}
    target_span = next(
        event
        for event in events
        if event["event_type"] == "span_started" and event["stage"] == "target_tracking"
    )
    target_progress = next(
        event
        for event in events
        if event["event_type"] == "progress_sampled"
        and event["stage"] == "target_tracking"
    )
    assert target_span["runtime"]["backend"] == full_pipeline.DEFAULT_IDENTITY_BACKEND
    assert target_progress["runtime"]["backend"] == full_pipeline.DEFAULT_IDENTITY_BACKEND
    pose_span = next(
        event
        for event in events
        if event["event_type"] == "span_started" and event["stage"] == "pose_sequence"
    )
    assert pose_span["runtime"]["device"] == "cuda"
    assert pose_span["runtime"]["runtime_backend"] == "onnxruntime"
    target_complete = next(
        event
        for event in events
        if event["event_type"] == "stage_completed" and event["stage"] == "target_tracking"
    )
    assert target_complete["elapsed_seconds"] == 1.0
    assert target_complete["measurements"]["milliseconds_per_frame"] == pytest.approx(
        1000.0 / 12
    )


def test_full_pipeline_emits_failed_stage_and_span_without_running_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current = [10.0]
    telemetry = ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="11111111-1111-4111-8111-111111111111",
        local_path=tmp_path / "events.jsonl",
        monotonic_clock=lambda: current[0],
        wall_clock=lambda: datetime(2026, 7, 9, tzinfo=timezone.utc),
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
    )

    def fail_identity(**kwargs: Any) -> dict[str, Any]:
        kwargs["progress_callback"](
            {"phase": "detect_track", "processed_frames": 1, "total_frames": 20}
        )
        current[0] += 2.25
        raise RuntimeError("identity exploded")

    monkeypatch.setattr(full_pipeline, "run_identity_tracking", fail_identity)
    monkeypatch.setattr(
        full_pipeline,
        "run_sam31_mlx_mask",
        lambda **_: pytest.fail("mask must not run after identity failure"),
    )

    with pytest.raises(RuntimeError, match="identity exploded"):
        full_pipeline.run_full_cv_pipeline(
            run_dir=tmp_path,
            pose_backend="mediapipe",
            telemetry=telemetry,
        )

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "stage_started",
        "span_started",
        "progress_sampled",
        "span_failed",
        "stage_failed",
    ]
    assert events[-2]["elapsed_seconds"] == 2.25
    assert events[-1]["elapsed_seconds"] == 2.25
    assert events[-1]["error"]["exception_type"] == "RuntimeError"


def test_full_pipeline_raises_when_stage_returns_unavailable_and_stops_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry = ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="11111111-1111-4111-8111-111111111111",
        local_path=tmp_path / "events.jsonl",
        wall_clock=lambda: datetime(2026, 7, 9, tzinfo=timezone.utc),
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
    )
    monkeypatch.setattr(
        full_pipeline,
        "run_identity_tracking",
        lambda **_: {"status": "unavailable", "error": "identity backend missing"},
    )
    monkeypatch.setattr(
        full_pipeline,
        "run_sam31_mlx_mask",
        lambda **_: pytest.fail("mask must not run after unavailable identity stage"),
    )

    with pytest.raises(RuntimeError, match="identity backend missing"):
        full_pipeline.run_full_cv_pipeline(
            run_dir=tmp_path,
            pose_backend="mediapipe",
            telemetry=telemetry,
        )

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["stage_started", "stage_failed"]
    assert events[-1]["stage"] == "target_tracking"
    assert events[-1]["measurements"]["outcome"] == "unavailable"
