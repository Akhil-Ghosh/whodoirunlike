from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest

from whodoirunlike.running_clip_run import RunningClipRun


LEGACY_CANONICAL_FILENAMES = {
    "source_segment": "source_segment.mp4",
    "prompt_frame": "prompt_frame.jpg",
    "person_prompt": "person_prompt.json",
    "target_prompt": "person_prompt.json",
    "track_seed": "track_seed.json",
    "view_bucket": "view_bucket.json",
    "tracklets": "tracklets.parquet",
    "tracklets_jsonl": "tracklets.jsonl",
    "reid": "reid.parquet",
    "reid_jsonl": "reid.jsonl",
    "masks_jsonl": "masks.jsonl",
    "mask_logits": "mask_logits.zarr",
    "poses": "poses.parquet",
    "pose_landmarks": "pose_landmarks.jsonl",
    "runner_mask": "runner_mask.mp4",
    "densepose": "densepose.jsonl",
    "densepose_parquet": "densepose.parquet",
    "fused_form": "fused_form.jsonl",
    "fused_form_parquet": "fused_form.parquet",
    "skeleton_render": "skeleton_render.mp4",
    "masked_runner": "masked_runner.mp4",
    "pose_qa_overlay": "pose_qa_overlay.mp4",
    "qa_overlay": "qa_overlay.mp4",
    "fused_overlay": "fused_overlay.mp4",
    "qc_metrics": "qc_metrics.json",
    "features": "features.json",
    "form_features": "form_features.json",
    "form_feature_arrays": "form_features.npz",
    "mmpose_landmarks": "mmpose_landmarks.jsonl",
    "openpose_landmarks": "openpose_landmarks.jsonl",
    "openpose_skeleton_render": "openpose_skeleton_render.mp4",
    "openpose_qa_overlay": "openpose_qa_overlay.mp4",
    "pose_comparison": "pose_comparison.json",
}


def test_canonical_paths_preserve_legacy_catalog_and_alias(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "run")

    paths = run.canonical_paths()

    assert {key: Path(path).name for key, path in paths.items()} == LEGACY_CANONICAL_FILENAMES
    assert paths["target_prompt"] == paths["person_prompt"]
    auxiliary_paths = run.canonical_paths(("runner_mask_metadata", "hosted_pipeline_result"))
    assert Path(auxiliary_paths["runner_mask_metadata"]).name == "runner_mask_metadata.jsonl"
    assert Path(auxiliary_paths["hosted_pipeline_result"]).name == "hosted_pipeline_result.json"
    assert run.canonical_paths(("source_segment", "qc_metrics")) == {
        "source_segment": str(run.run_dir / "source_segment.mp4"),
        "qc_metrics": str(run.run_dir / "qc_metrics.json"),
    }


def test_ensure_paths_fills_partial_legacy_manifest_without_mutating_it(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "run")
    manifest = {
        "version": 1,
        "future_field": {"keep": True},
        "paths": {"source_segment": "/legacy/source.mp4", "future_path": "future.bin"},
        "stages": {"future_stage": {"status": "future"}},
    }

    updated = run.ensure_paths(manifest, ("source_segment", "pose_landmarks"))

    assert updated is not manifest
    assert updated["paths"] is not manifest["paths"]
    assert manifest["paths"] == {
        "source_segment": "/legacy/source.mp4",
        "future_path": "future.bin",
    }
    assert updated["version"] == 1
    assert updated["future_field"] == {"keep": True}
    assert updated["stages"] == manifest["stages"]
    assert updated["paths"]["source_segment"] == "/legacy/source.mp4"
    assert updated["paths"]["future_path"] == "future.bin"
    assert updated["paths"]["pose_landmarks"] == str(run.run_dir / "pose_landmarks.jsonl")


@pytest.mark.parametrize("configured_key", ["person_prompt", "target_prompt"])
def test_ensure_paths_carries_prompt_alias_configuration(
    configured_key: str,
    tmp_path: Path,
) -> None:
    run = RunningClipRun(tmp_path / "run")
    manifest = {"paths": {configured_key: "prompts/custom.json"}}

    updated = run.ensure_paths(manifest, ("person_prompt", "target_prompt"))

    assert updated["paths"]["person_prompt"] == "prompts/custom.json"
    assert updated["paths"]["target_prompt"] == "prompts/custom.json"


def test_artifact_path_prefers_absolute_and_relative_manifest_paths(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "run")
    absolute_path = tmp_path / "outside" / "pose.jsonl"
    manifest = {
        "paths": {
            "pose_landmarks": str(absolute_path),
            "qc_metrics": "relative/custom-qc.json",
            "person_prompt": "relative/prompt.json",
        }
    }

    assert run.artifact_path("pose_landmarks", manifest) == absolute_path
    assert run.artifact_path("qc_metrics", manifest) == Path("relative/custom-qc.json")
    assert not run.artifact_path("qc_metrics", manifest).is_absolute()
    assert run.artifact_path("target_prompt", manifest) == Path("relative/prompt.json")
    assert run.artifact_path("fused_form", manifest) == run.run_dir / "fused_form.jsonl"


def test_write_and_read_manifest_preserve_version_and_unknown_data(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "nested" / "run")
    manifest = {
        "version": 1,
        "future_field": ["untouched"],
        "paths": {"future_artifact": "future.data"},
        "stages": {"future_stage": {"status": "waiting", "future": 7}},
    }

    written_path = run.write_manifest(manifest)

    assert written_path == run.manifest_path
    assert run.read_manifest() == manifest


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ([], "manifest must be an object"),
        ({"paths": []}, "manifest 'paths' must be an object"),
        ({"stages": []}, "manifest 'stages' must be an object"),
    ],
)
def test_manifest_payload_requires_object_shapes(
    manifest: Any,
    message: str,
    tmp_path: Path,
) -> None:
    run = RunningClipRun(tmp_path / "run")

    with pytest.raises(ValueError, match=message):
        run.write_manifest(manifest)


def test_update_stage_merges_only_selected_stage_and_writes_manifest(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "run")
    manifest = {
        "version": 1,
        "future_field": {"keep": "yes"},
        "paths": {"future_path": "future.bin"},
        "stages": {
            "pose": {"status": "pending", "recommended_tool": "keep-me"},
            "future_stage": {"status": "future", "payload": [1, 2, 3]},
        },
    }

    updated = run.update_stage("pose", {"status": "complete", "output": "pose.jsonl"}, manifest)

    assert manifest["stages"]["pose"] == {
        "status": "pending",
        "recommended_tool": "keep-me",
    }
    assert updated["version"] == 1
    assert updated["future_field"] == manifest["future_field"]
    assert updated["paths"] == manifest["paths"]
    assert updated["stages"]["future_stage"] == manifest["stages"]["future_stage"]
    assert updated["stages"]["pose"] == {
        "status": "complete",
        "recommended_tool": "keep-me",
        "output": "pose.jsonl",
    }
    assert run.read_manifest() == updated


def test_concurrent_stage_updates_from_stale_snapshots_preserve_both_results(
    tmp_path: Path,
) -> None:
    run = RunningClipRun(tmp_path / "run")
    run.write_manifest(
        {
            "version": 1,
            "paths": {},
            "stages": {
                "pose": {"status": "pending"},
                "densepose": {"status": "pending"},
            },
        }
    )
    pose_snapshot = run.read_manifest()
    densepose_snapshot = run.read_manifest()
    pose_snapshot["paths"]["pose_qa_overlay"] = "pose_qa_overlay.mp4"
    densepose_snapshot["paths"]["qa_overlay"] = "qa_overlay.mp4"
    start = threading.Barrier(3)

    def update_pose() -> None:
        start.wait()
        RunningClipRun(run.run_dir).update_stage(
            "pose",
            {"status": "complete", "output": "pose.jsonl"},
            pose_snapshot,
        )

    def update_densepose() -> None:
        start.wait()
        RunningClipRun(run.run_dir).update_stage(
            "densepose",
            {"status": "complete", "output": "densepose.jsonl"},
            densepose_snapshot,
        )

    pose_thread = threading.Thread(target=update_pose)
    densepose_thread = threading.Thread(target=update_densepose)
    pose_thread.start()
    densepose_thread.start()
    start.wait()
    pose_thread.join(timeout=2.0)
    densepose_thread.join(timeout=2.0)

    assert not pose_thread.is_alive()
    assert not densepose_thread.is_alive()
    manifest = run.read_manifest()
    assert manifest["stages"]["pose"] == {
        "status": "complete",
        "output": "pose.jsonl",
    }
    assert manifest["stages"]["densepose"] == {
        "status": "complete",
        "output": "densepose.jsonl",
    }
    assert manifest["paths"] == {
        "pose_qa_overlay": "pose_qa_overlay.mp4",
        "qa_overlay": "qa_overlay.mp4",
    }


def test_stale_stage_snapshot_cannot_move_updated_at_backward(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "run")
    run.write_manifest(
        {
            "version": 1,
            "updated_at": "2026-07-11T18:05:00+00:00",
            "paths": {},
            "stages": {"pose": {"status": "pending"}},
        }
    )
    stale_snapshot = run.read_manifest()
    stale_snapshot["updated_at"] = "2026-07-11T18:04:00+00:00"

    run.update_stage("pose", {"status": "complete"}, stale_snapshot)

    assert run.read_manifest()["updated_at"] == "2026-07-11T18:05:00+00:00"


def test_stage_owner_can_clear_stale_error_without_replacing_other_stages(
    tmp_path: Path,
) -> None:
    run = RunningClipRun(tmp_path / "run")
    run.write_manifest(
        {
            "version": 1,
            "paths": {},
            "stages": {
                "pose": {"status": "failed", "error": "old failure"},
                "densepose": {"status": "complete", "usable_frames": 8},
            },
        }
    )
    pose_snapshot = run.read_manifest()
    pose_snapshot["stages"]["pose"].pop("error")

    run.update_stage("pose", {"status": "complete"}, pose_snapshot)

    manifest = run.read_manifest()
    assert manifest["stages"]["pose"] == {"status": "complete"}
    assert manifest["stages"]["densepose"] == {
        "status": "complete",
        "usable_frames": 8,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ([], "stage updates must be an object"),
        ({"pose": []}, "stage values must be an object"),
    ],
)
def test_update_stages_requires_object_shapes(
    updates: Any,
    message: str,
    tmp_path: Path,
) -> None:
    run = RunningClipRun(tmp_path / "run")

    with pytest.raises(ValueError, match=message):
        run.update_stages(updates, {"version": 1, "paths": {}, "stages": {}})

    assert not run.manifest_path.exists()


def test_update_stages_replaces_once_without_exposing_intermediate_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from whodoirunlike import running_clip_run as module

    run = RunningClipRun(tmp_path / "run")
    old_manifest = {
        "version": 1,
        "future_field": {"keep": True},
        "paths": {"future_path": "future.bin"},
        "stages": {
            "pose": {"status": "pending", "recommended_tool": "keep-me"},
            "renders": {"status": "pending"},
            "features": {"status": "pending"},
            "future_stage": {"status": "future", "payload": [1, 2, 3]},
        },
    }
    run.write_manifest(old_manifest)
    original_replace = os.replace
    observed_before_replace: list[dict[str, Any]] = []

    def observe_replace(source: str | Path, destination: str | Path) -> None:
        observed_before_replace.append(run.read_manifest())
        original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", observe_replace)

    updated = run.update_stages(
        {
            "pose": {"status": "complete", "output": "pose.jsonl"},
            "renders": {"status": "partial_complete", "output": "skeleton.mp4"},
            "features": {"status": "complete", "output": "features.json"},
        }
    )

    assert observed_before_replace == [old_manifest]
    assert {updated["stages"][stage]["status"] for stage in ("pose", "renders", "features")} == {
        "complete",
        "partial_complete",
    }
    assert updated["stages"]["pose"]["recommended_tool"] == "keep-me"
    assert updated["stages"]["future_stage"] == old_manifest["stages"]["future_stage"]
    assert updated["future_field"] == old_manifest["future_field"]
    assert run.read_manifest() == updated


def test_manifest_replacement_keeps_old_json_readable_until_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from whodoirunlike import running_clip_run as module

    run = RunningClipRun(tmp_path / "run")
    old_manifest = {"version": 1, "paths": {}, "stages": {}, "marker": "old"}
    new_manifest = {"version": 1, "paths": {}, "stages": {}, "marker": "new"}
    run.write_manifest(old_manifest)
    original_replace = os.replace
    observed_before_replace: list[dict[str, Any]] = []

    def observe_replace(source: str | Path, destination: str | Path) -> None:
        observed_before_replace.append(run.read_manifest())
        original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", observe_replace)

    run.write_manifest(new_manifest)

    assert observed_before_replace == [old_manifest]
    assert run.read_manifest() == new_manifest
    assert list(run.run_dir.glob(f".{run.manifest_path.name}.*.tmp")) == []


def test_failed_manifest_serialization_leaves_previous_manifest_intact(tmp_path: Path) -> None:
    run = RunningClipRun(tmp_path / "run")
    manifest = {"version": 1, "paths": {}, "stages": {}, "marker": "stable"}
    run.write_manifest(manifest)

    with pytest.raises(TypeError):
        run.write_manifest({**manifest, "not_json": object()})

    assert run.read_manifest() == manifest
    assert list(run.run_dir.glob(f".{run.manifest_path.name}.*.tmp")) == []


def test_existing_artifacts_use_configured_paths_deduplicate_aliases_and_add_names(
    tmp_path: Path,
) -> None:
    run = RunningClipRun(tmp_path / "run")
    run.run_dir.mkdir(parents=True)
    custom_prompt = tmp_path / "custom" / "prompt.json"
    custom_prompt.parent.mkdir()
    custom_prompt.write_text("{}", encoding="utf-8")
    (run.run_dir / "qc_metrics.json").write_text("{}", encoding="utf-8")
    (run.run_dir / "extra-report.json").write_text("{}", encoding="utf-8")
    run.write_manifest(
        {
            "version": 1,
            "paths": {
                "person_prompt": str(custom_prompt),
                "target_prompt": str(custom_prompt),
            },
            "stages": {},
        }
    )

    artifacts = run.existing_artifacts(
        ("person_prompt", "target_prompt", "qc_metrics", "fused_form"),
        ("extra-report.json", "qc_metrics.json"),
    )

    assert artifacts == [
        custom_prompt,
        run.run_dir / "qc_metrics.json",
        run.run_dir / "extra-report.json",
    ]
