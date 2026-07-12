from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import gzip
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import statistics
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np

from whodoirunlike.mask_artifacts import iter_mask_video, write_masks_jsonl_from_video
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    EXACT_CANDIDATE_COMMIT,
    EXACT_CANDIDATE_IMAGE_DIGEST,
    EXACT_CONTROL_COMMIT,
    EXACT_CONTROL_IMAGE_DIGEST,
    STRICT_MASK_GATE_THRESHOLDS,
    get_parity_fixture,
    validate_prompt_for_fixture,
    verify_non_overlay_production_files,
)
from whodoirunlike.sam2_runner import inspect_video


@dataclass(frozen=True)
class PoseParityMeasurements:
    control_frame_count: int
    candidate_frame_count: int
    schema_match: bool
    control_schema_preserved: bool
    required_fields_present: bool
    aligned_frame_count: int
    usable_agreement_rate: float | None
    new_unusable_frame_count: int
    common_visible_point_count: int
    pck_at_001_diagonal: float | None
    joint_error_median_normalized: float | None
    joint_error_p95_normalized: float | None
    visibility_mae: float | None


@dataclass(frozen=True)
class DensePoseParityMeasurements:
    control_frame_count: int
    candidate_frame_count: int
    schema_match: bool
    control_schema_preserved: bool
    required_fields_present: bool
    aligned_frame_count: int
    usable_agreement_rate: float | None
    new_unusable_frame_count: int
    common_usable_frame_count: int
    part_jaccard_mean: float | None
    part_jaccard_p05: float | None
    centroid_error_normalized_mean: float | None
    centroid_error_normalized_p95: float | None
    bbox_iou_p05: float | None
    coverage_mae: float | None
    mask_overlap_mae: float | None


@dataclass(frozen=True)
class FusionParityMeasurements:
    control_frame_count: int
    candidate_frame_count: int
    schema_match: bool
    required_fields_present: bool
    aligned_frame_count: int
    frame_state_agreement_rate: float | None
    risk_state_increase_count: int
    usable_agreement_rate: float | None
    confidence_mae: float | None
    confidence_mean_drop: float | None
    common_joint_weight_count: int
    joint_weight_mae: float | None
    joint_weight_p95_error: float | None


@dataclass(frozen=True)
class FeatureParityMeasurements:
    control_frame_count: int
    candidate_frame_count: int
    npz_keys_match: bool
    npz_shapes_match: bool
    npz_dtypes_match: bool
    array_schema_match: bool
    comparable_array_count: int
    array_max_abs_delta: float | None
    valid_frame_loss_count: int
    joint_angle_common_value_count: int
    joint_angle_median_abs_error: float | None
    joint_angle_p95_abs_error: float | None
    runner_metric_keys_match: bool
    comparable_runner_metric_count: int
    runner_metric_max_abs_delta: float | None
    runner_metrics_within_tolerance: bool


@dataclass(frozen=True)
class QcParityMeasurements:
    schema_match: bool
    required_components_present: bool
    categorical_match: bool
    numeric_field_count: int
    numeric_max_abs_delta: float | None
    identity_exact: bool
    mask_churn_abs_delta: float | None
    uncertainty_increase: float | None


@dataclass(frozen=True)
class ArtifactParityMeasurements:
    control_required_artifacts_present: bool
    candidate_required_artifacts_present: bool
    inventory_match: bool
    control_inventory_preserved: bool
    schema_artifact_count: int
    json_schema_match: bool
    json_control_schema_preserved: bool
    parquet_schema_match: bool
    parquet_control_schema_preserved: bool
    parquet_row_counts_match: bool


@dataclass(frozen=True)
class VideoParityMeasurements:
    control_required_videos_present: bool
    candidate_required_videos_present: bool
    decoded_video_count: int
    all_videos_playable: bool
    no_blank_frames: bool
    dimensions_exact: bool
    frame_counts_exact: bool
    profile_metadata_match: bool
    fps_expected_match: bool
    fps_max_abs_delta: float | None


@dataclass(frozen=True)
class RunnerMaskParityMeasurements:
    control_width: int
    control_height: int
    candidate_width: int
    candidate_height: int
    control_frame_count: int
    candidate_frame_count: int
    control_nonempty_frame_count: int
    candidate_nonempty_frame_count: int
    iou_mean: float | None
    iou_p05: float | None
    boundary_f1_mean: float | None
    centroid_error_normalized_mean: float | None
    coverage_mae: float | None
    mask_churn_abs_delta: float | None


@dataclass(frozen=True)
class PipelineBenchmarkProfile:
    profile_id: str
    execution_mode: str
    mask_source: str
    parallel_mask_presentation: bool
    parallel_pose_densepose: bool
    parallel_post_fusion: bool
    environment_overrides: tuple[tuple[str, str | None], ...] = ()


PipelineStageCallable = Callable[[Path], dict[str, Any]]


_CONTROL_DEPLOY_ENV = (
    ("WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION", "false"),
    ("WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE", "false"),
    ("WHODOIRUNLIKE_PARALLEL_POST_FUSION", "false"),
    ("WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH", "false"),
    ("MMPOSE_DEVICE", "cpu"),
    ("RTMW_RUNTIME_BACKEND", "onnxruntime"),
    ("MMPOSE_USE_DETECTOR", "true"),
    ("DENSEPOSE_INPUT_MIN_SIZE_TEST", None),
    ("DENSEPOSE_INPUT_MAX_SIZE_TEST", None),
    ("DENSEPOSE_TARGET_CROP_ENABLED", None),
)
_SCHEDULE_ONLY_DEPLOY_ENV = (
    ("WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION", "true"),
    ("WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE", "true"),
    ("WHODOIRUNLIKE_PARALLEL_POST_FUSION", "true"),
    ("WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH", "true"),
    ("MMPOSE_DEVICE", "cpu"),
    ("RTMW_RUNTIME_BACKEND", "onnxruntime"),
    ("MMPOSE_USE_DETECTOR", "true"),
    ("DENSEPOSE_INPUT_MIN_SIZE_TEST", None),
    ("DENSEPOSE_INPUT_MAX_SIZE_TEST", None),
    ("DENSEPOSE_TARGET_CROP_ENABLED", None),
)
_OPTIMIZED_DEPLOY_ENV = (
    ("WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION", "true"),
    ("WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE", "true"),
    ("WHODOIRUNLIKE_PARALLEL_POST_FUSION", "true"),
    ("WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH", "true"),
    ("MMPOSE_DEVICE", "cuda"),
    ("RTMW_RUNTIME_BACKEND", "onnxruntime"),
    ("MMPOSE_USE_DETECTOR", "false"),
    ("DENSEPOSE_INPUT_MIN_SIZE_TEST", "512"),
    ("DENSEPOSE_INPUT_MAX_SIZE_TEST", "960"),
    ("DENSEPOSE_TARGET_CROP_ENABLED", "true"),
)


@dataclass(frozen=True)
class PipelineStageFunctions:
    pose: PipelineStageCallable
    densepose: PipelineStageCallable
    fusion: PipelineStageCallable
    features: PipelineStageCallable
    tables: PipelineStageCallable
    qc: PipelineStageCallable
    full_pipeline: PipelineStageCallable


PIPELINE_BENCHMARK_PROFILES = {
    "downstream_baseline_control": PipelineBenchmarkProfile(
        profile_id="downstream_baseline_control",
        execution_mode="fixed_mask_downstream",
        mask_source="baseline",
        parallel_mask_presentation=False,
        parallel_pose_densepose=False,
        parallel_post_fusion=False,
        environment_overrides=_CONTROL_DEPLOY_ENV,
    ),
    "downstream_candidate_control": PipelineBenchmarkProfile(
        profile_id="downstream_candidate_control",
        execution_mode="fixed_mask_downstream",
        mask_source="candidate",
        parallel_mask_presentation=False,
        parallel_pose_densepose=False,
        parallel_post_fusion=False,
        environment_overrides=_CONTROL_DEPLOY_ENV,
    ),
    "downstream_candidate_optimized": PipelineBenchmarkProfile(
        profile_id="downstream_candidate_optimized",
        execution_mode="fixed_mask_downstream",
        mask_source="candidate",
        parallel_mask_presentation=False,
        parallel_pose_densepose=True,
        parallel_post_fusion=True,
        environment_overrides=_OPTIMIZED_DEPLOY_ENV,
    ),
    "production_control": PipelineBenchmarkProfile(
        profile_id="production_control",
        execution_mode="production_full_pipeline",
        mask_source="production",
        parallel_mask_presentation=False,
        parallel_pose_densepose=False,
        parallel_post_fusion=False,
        environment_overrides=_CONTROL_DEPLOY_ENV,
    ),
    "production_candidate": PipelineBenchmarkProfile(
        profile_id="production_candidate",
        execution_mode="production_full_pipeline",
        mask_source="production",
        parallel_mask_presentation=True,
        parallel_pose_densepose=True,
        parallel_post_fusion=True,
        environment_overrides=_OPTIMIZED_DEPLOY_ENV,
    ),
    "production_candidate_schedule_only": PipelineBenchmarkProfile(
        profile_id="production_candidate_schedule_only",
        execution_mode="production_full_pipeline",
        mask_source="production",
        parallel_mask_presentation=True,
        parallel_pose_densepose=True,
        parallel_post_fusion=True,
        environment_overrides=_SCHEDULE_ONLY_DEPLOY_ENV,
    ),
}
DEFAULT_PIPELINE_PROFILE_MATRIX = (
    "downstream_baseline_control",
    "downstream_candidate_control",
    "downstream_candidate_optimized",
)

_FULL_ASSET_ENCODINGS = {
    "person_prompt_json": "base64",
    "tracklets_jsonl": "gzip+base64",
    "baseline_runner_mask_mp4": "base64",
}
_FULL_ASSET_MAX_BYTES = {
    "person_prompt_json": 16 * 1024,
    "tracklets_jsonl": 2 * 1024 * 1024,
    "baseline_runner_mask_mp4": 2 * 1024 * 1024,
}

_BENCHMARK_REPO_ROOT = Path(__file__).resolve().parents[2]
_DENSEPOSE_DEFAULT_CONFIG = (
    _BENCHMARK_REPO_ROOT / "models/densepose/detectron2/projects/DensePose/configs/"
    "densepose_rcnn_R_50_FPN_s1x.yaml"
)
_DENSEPOSE_DEFAULT_WEIGHTS = (
    _BENCHMARK_REPO_ROOT
    / "models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl"
)


def _environment_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _resolve_benchmark_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_BENCHMARK_REPO_ROOT / path).resolve()


def _optional_positive_int(name: str) -> int | None:
    value = _environment_value(name)
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(name: str, default: int) -> int:
    value = _environment_value(name)
    parsed = int(value) if value else default
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _nonnegative_float(name: str, default: float) -> float:
    value = _environment_value(name)
    parsed = float(value) if value else default
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _densepose_benchmark_kwargs() -> dict[str, Any]:
    """Resolve the production DensePose configuration without private imports."""

    config_value = _environment_value("DENSEPOSE_CONFIG")
    weights_value = _environment_value("DENSEPOSE_WEIGHTS")
    config_path = (
        _resolve_benchmark_path(config_value) if config_value else _DENSEPOSE_DEFAULT_CONFIG
    )
    if weights_value.startswith(("http://", "https://")):
        weights_path = weights_value
    elif weights_value:
        weights_path = str(_resolve_benchmark_path(weights_value))
    else:
        weights_path = str(_DENSEPOSE_DEFAULT_WEIGHTS)
    return {
        "config_path": config_path,
        "weights_path": weights_path,
        "device": _environment_value("DENSEPOSE_DEVICE") or "cpu",
        "input_min_size_test": _optional_positive_int("DENSEPOSE_INPUT_MIN_SIZE_TEST"),
        "input_max_size_test": _optional_positive_int("DENSEPOSE_INPUT_MAX_SIZE_TEST"),
        "target_crop_enabled": _environment_value("DENSEPOSE_TARGET_CROP_ENABLED").lower()
        in {"1", "true", "yes", "on"},
        "target_crop_padding_ratio": _nonnegative_float(
            "DENSEPOSE_TARGET_CROP_PADDING_RATIO",
            0.2,
        ),
        "target_crop_padding_pixels": _nonnegative_int(
            "DENSEPOSE_TARGET_CROP_PADDING_PIXELS",
            16,
        ),
    }


def resolve_pipeline_profiles(
    profile_ids: list[str] | tuple[str, ...] | None,
) -> list[PipelineBenchmarkProfile]:
    selected_ids = list(profile_ids or DEFAULT_PIPELINE_PROFILE_MATRIX)
    if not selected_ids:
        raise ValueError("Full pipeline benchmark requires at least one profile.")
    if len(selected_ids) > 3:
        raise ValueError("Full pipeline benchmark accepts at most three profiles per job.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Full pipeline benchmark profile IDs must be unique.")
    unsupported = [
        profile_id for profile_id in selected_ids if profile_id not in PIPELINE_BENCHMARK_PROFILES
    ]
    if unsupported:
        raise ValueError(f"Unsupported full pipeline benchmark profile: {unsupported[0]}")
    profiles = [PIPELINE_BENCHMARK_PROFILES[profile_id] for profile_id in selected_ids]
    if len({profile.execution_mode for profile in profiles}) > 1:
        raise ValueError("Full pipeline comparison profiles must use the same execution mode.")
    if profiles[0].execution_mode == "production_full_pipeline" and len(profiles) > 2:
        raise ValueError("Production full-pipeline comparison accepts at most two profiles.")
    return profiles


def _decode_full_asset(
    name: str,
    payload: Any,
    *,
    expected_sha256: str,
) -> bytes:
    if not isinstance(payload, dict):
        raise ValueError(f"Full benchmark asset {name} must be an object.")
    encoding = _FULL_ASSET_ENCODINGS[name]
    if payload.get("encoding") != encoding:
        raise ValueError(f"Full benchmark asset {name} has an unsupported encoding.")
    if payload.get("sha256") != expected_sha256:
        raise ValueError(f"Full benchmark asset {name} does not match the fixture hash.")
    encoded = payload.get("data")
    if not isinstance(encoded, str):
        raise ValueError(f"Full benchmark asset {name} must contain base64 data.")
    max_bytes = _FULL_ASSET_MAX_BYTES[name]
    if len(encoded) > ((max_bytes + 2) // 3) * 4 + 4096:
        raise ValueError(f"Full benchmark asset {name} exceeds its encoded size limit.")
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Full benchmark asset {name} is not valid base64.") from exc
    if encoding == "gzip+base64":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(packed), mode="rb") as stream:
                raw = stream.read(max_bytes + 1)
        except (EOFError, OSError) as exc:
            raise ValueError(f"Full benchmark asset {name} is not valid gzip data.") from exc
    else:
        raw = packed
    if len(raw) > max_bytes:
        raise ValueError(f"Full benchmark asset {name} exceeds its decoded size limit.")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Full benchmark asset {name} failed SHA-256 verification.")
    return raw


def validate_full_benchmark_request(
    payload: Any,
) -> tuple[list[PipelineBenchmarkProfile], dict[str, bytes]]:
    if not isinstance(payload, dict):
        raise ValueError("Full pipeline benchmark input must be an object.")
    if payload.get("type") != "sam31_benchmark":
        raise ValueError("Unsupported full pipeline benchmark request type.")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported full pipeline benchmark schema version.")
    if payload.get("scope") != "full":
        raise ValueError("Full pipeline benchmark requires scope=full.")
    if payload.get("fixture_id") != CANONICAL_FRAME130_FIXTURE_ID:
        raise ValueError("Full pipeline benchmark requires the canonical frame-130 fixture.")
    profile_ids = payload.get("profile_ids")
    if profile_ids is not None and (
        not isinstance(profile_ids, list)
        or not all(isinstance(profile_id, str) for profile_id in profile_ids)
    ):
        raise ValueError("Full pipeline benchmark profile_ids must be a list of strings.")
    profiles = resolve_pipeline_profiles(profile_ids)
    assets_payload = payload.get("assets")
    if not isinstance(assets_payload, dict) or set(assets_payload) != set(_FULL_ASSET_ENCODINGS):
        raise ValueError("Full pipeline benchmark requires the exact canonical assets.")
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    assets = {
        name: _decode_full_asset(
            name,
            assets_payload[name],
            expected_sha256=fixture.asset_sha256[name],
        )
        for name in _FULL_ASSET_ENCODINGS
    }
    prompt_validation = validate_prompt_for_fixture(
        assets["person_prompt_json"],
        fixture=fixture,
    )
    if not prompt_validation["raw_hash_matches"]:
        raise ValueError("Full benchmark prompt failed raw SHA-256 verification.")
    return profiles, assets


def materialize_pipeline_fixture(
    *,
    run_dir: Path,
    source_path: Path,
    assets: dict[str, bytes],
    profile_id: str,
) -> Path:
    required_assets = {
        "person_prompt_json",
        "tracklets_jsonl",
        "baseline_runner_mask_mp4",
    }
    if set(assets) != required_assets:
        raise ValueError("Pipeline fixture requires the exact canonical input assets.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run = RunningClipRun(run_dir)
    paths = run.canonical_paths()
    source_target = Path(paths["source_segment"])
    shutil.copyfile(source_path, source_target)
    Path(paths["person_prompt"]).write_bytes(assets["person_prompt_json"])
    Path(paths["tracklets_jsonl"]).write_bytes(assets["tracklets_jsonl"])
    Path(paths["runner_mask"]).write_bytes(assets["baseline_runner_mask_mp4"])
    write_masks_jsonl_from_video(Path(paths["runner_mask"]), Path(paths["masks_jsonl"]))

    candidate_id = f"benchmark-{profile_id.replace('_', '-')}"
    video_meta = inspect_video(source_target)
    (run_dir / "track_seed.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "runner_name": "Cole Hocker",
                "prompt_path": paths["person_prompt"],
                "target_lock_method": "canonical_frame130_fixture",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "view_bucket.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "runner_name": "Cole Hocker",
                "view_bucket": "unknown",
                "source": "sam31_pipeline_benchmark",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "created_at": "1970-01-01T00:00:00Z",
        "candidate_id": candidate_id,
        "runner_name": "Cole Hocker",
        "runner_slug": "cole-hocker",
        "implementation_goal": "identity_stable_runner_clip",
        "source": {
            "platform": "sam31_pipeline_benchmark",
            "filename": "cole-source.mp4",
            "content_type": "video/mp4",
            "size_bytes": source_target.stat().st_size,
            "sha256": hashlib.sha256(source_target.read_bytes()).hexdigest(),
        },
        "review": {
            "quality": "canonical_benchmark",
            "camera_angle": "unknown",
            "primary_bucket": "running",
            "duration_seconds": video_meta.get("duration_seconds"),
        },
        "video": video_meta,
        "target_prompt_source": "canonical_frame130_fixture",
        "paths": paths,
        "stages": {
            "upload": {"status": "complete", "output": paths["source_segment"]},
            "person_prompt": {"status": "fixed", "output": paths["person_prompt"]},
            "detector_tracker": {
                "status": "fixed",
                "tracklets_jsonl": paths["tracklets_jsonl"],
            },
            "whole_runner_mask": {
                "status": "fixed",
                "backend": "sam31_gpu_production_baseline",
                "output": paths["runner_mask"],
                "masks_jsonl": paths["masks_jsonl"],
            },
            "pose": {"status": "pending"},
            "densepose": {"status": "pending"},
            "fused_form": {"status": "pending"},
            "form_features": {"status": "pending"},
            "qc_metrics": {"status": "pending"},
        },
    }
    return run.write_manifest(manifest)


def install_candidate_mask_artifacts(
    *,
    candidate_run_dir: Path,
    downstream_run_dir: Path,
) -> dict[str, str]:
    """Copy one generated candidate mask into an independent downstream arm."""

    candidate_run = RunningClipRun(candidate_run_dir)
    downstream_run = RunningClipRun(downstream_run_dir)
    candidate_manifest = candidate_run.read_manifest()
    downstream_manifest = downstream_run.read_manifest()
    copied: dict[str, str] = {}
    for key in ("runner_mask", "masks_jsonl"):
        source = candidate_run.artifact_path(key, candidate_manifest)
        destination = downstream_run.artifact_path(key, downstream_manifest)
        if not source.is_file():
            raise FileNotFoundError(f"Candidate mask artifact is unavailable: {key}")
        shutil.copy2(source, destination)
        copied[key] = hashlib.sha256(destination.read_bytes()).hexdigest()
    downstream_manifest["updated_at"] = "1970-01-01T00:00:00Z"
    downstream_run.update_stage(
        "whole_runner_mask",
        {
            "status": "fixed_candidate",
            "backend": "sam31_gpu_public_candidate",
            "output": str(downstream_run.artifact_path("runner_mask", downstream_manifest)),
            "masks_jsonl": str(downstream_run.artifact_path("masks_jsonl", downstream_manifest)),
        },
        downstream_manifest,
    )
    return copied


_SUMMARY_RESULT_FIELDS = frozenset(
    {
        "status",
        "frame_count",
        "usable_frames",
        "usable_frame_count",
        "quality",
        "summary",
        "exports",
        "uncertainty_score",
    }
)
_SUMMARY_VOLATILE_FIELDS = frozenset(
    {
        "candidate_id",
        "created_at",
        "updated_at",
        "completed_at",
        "elapsed_seconds",
        "input",
        "output",
        "metadata",
        "arrays",
    }
)


def _normalize_summary_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_summary_value(child)
            for key, child in value.items()
            if str(key) not in _SUMMARY_VOLATILE_FIELDS and not str(key).endswith(("_path", "_url"))
        }
    if isinstance(value, list):
        return [_normalize_summary_value(item) for item in value[:100]]
    if isinstance(value, Path):
        return value.name
    return value


def _summarize_stage_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: _normalize_summary_value(value)
        for key, value in result.items()
        if key in _SUMMARY_RESULT_FIELDS
    }
    steps = result.get("steps")
    if isinstance(steps, list):
        summarized_steps: list[dict[str, Any]] = []
        for step in steps[:20]:
            if not isinstance(step, dict) or not isinstance(step.get("result"), dict):
                continue
            step_result = step["result"]
            summarized = {
                "stage": str(step.get("stage") or "unknown"),
                **{
                    key: step_result[key]
                    for key in ("status", "elapsed_seconds", "frame_count", "usable_frames")
                    if key in step_result
                },
            }
            summarized_steps.append(summarized)
        summary["steps"] = summarized_steps
    return summary


def _call_with_supported_kwargs(function: Any, **kwargs: Any) -> dict[str, Any]:
    import inspect

    supported = inspect.signature(function).parameters
    return function(**{key: value for key, value in kwargs.items() if key in supported})


def _default_stage_functions(
    profile: PipelineBenchmarkProfile,
) -> PipelineStageFunctions:
    from whodoirunlike.artifact_tables import export_cv_tables
    from whodoirunlike.densepose_runner import run_densepose
    from whodoirunlike.form_features import compile_form_features
    from whodoirunlike.full_pipeline import run_full_cv_pipeline
    from whodoirunlike.fusion_runner import run_fused_form
    from whodoirunlike.mmpose_runner import run_mmpose_pose
    from whodoirunlike.qc import run_qc_metrics

    def pose(run_dir: Path) -> dict[str, Any]:
        return _call_with_supported_kwargs(
            run_mmpose_pose,
            run_dir=run_dir,
            model_id="mmpose_rtmpose_l_384",
            isolate_qa_overlay=True,
        )

    def densepose(run_dir: Path) -> dict[str, Any]:
        return _call_with_supported_kwargs(
            run_densepose,
            run_dir=run_dir,
            write_qa_overlay=True,
            **_densepose_benchmark_kwargs(),
        )

    def full_pipeline(run_dir: Path) -> dict[str, Any]:
        return _call_with_supported_kwargs(
            run_full_cv_pipeline,
            run_dir=run_dir,
            pose_backend="mmpose_rtmpose_l_384",
            mask_backend="sam31_gpu",
            skip_densepose=False,
            parallel_mask_presentation=profile.parallel_mask_presentation,
            parallel_pose_densepose=profile.parallel_pose_densepose,
            parallel_post_fusion=profile.parallel_post_fusion,
        )

    return PipelineStageFunctions(
        pose=pose,
        densepose=densepose,
        fusion=lambda run_dir: run_fused_form(run_dir=run_dir),
        features=lambda run_dir: compile_form_features(run_dir=run_dir),
        tables=export_cv_tables,
        qc=run_qc_metrics,
        full_pipeline=full_pipeline,
    )


def _run_timed_stage(
    action: PipelineStageCallable,
    run_dir: Path,
) -> tuple[dict[str, Any], float]:
    started_at = time.perf_counter()
    result = action(run_dir)
    if not isinstance(result, dict):
        raise TypeError("Pipeline benchmark stage must return an object.")
    status = str(result.get("status") or "complete").lower()
    if status in {"failed", "failure", "error", "unavailable"}:
        raise RuntimeError(f"Pipeline benchmark stage returned {status}.")
    return result, time.perf_counter() - started_at


def _collect_parallel_stages(
    actions: dict[str, PipelineStageCallable],
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    results: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    errors: list[BaseException] = []
    with ThreadPoolExecutor(
        max_workers=len(actions),
        thread_name_prefix="pipeline-parity",
    ) as executor:
        futures = {
            name: executor.submit(_run_timed_stage, action, run_dir)
            for name, action in actions.items()
        }
        for name, future in futures.items():
            try:
                results[name], timings[name] = future.result()
            except BaseException as exc:
                errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        if all(isinstance(error, Exception) for error in errors):
            raise ExceptionGroup(
                "Pipeline benchmark stages failed",
                [error for error in errors if isinstance(error, Exception)],
            )
        raise errors[0]
    return results, timings


_PROFILE_ENVIRONMENT_LOCK = threading.RLock()


@contextmanager
def _profile_environment(profile: PipelineBenchmarkProfile):
    previous = {name: os.environ.get(name) for name, _ in profile.environment_overrides}
    with _PROFILE_ENVIRONMENT_LOCK:
        try:
            for name, value in profile.environment_overrides:
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _execute_pipeline_profile(
    run_dir: Path,
    profile: PipelineBenchmarkProfile,
    *,
    stage_functions: PipelineStageFunctions | None = None,
) -> dict[str, Any]:
    functions = stage_functions or _default_stage_functions(profile)
    overall_started_at = time.perf_counter()
    timings: dict[str, float] = {}
    stage_results: dict[str, dict[str, Any]] = {}

    if profile.execution_mode == "production_full_pipeline":
        result, elapsed = _run_timed_stage(functions.full_pipeline, run_dir)
        timings["production_full_pipeline"] = elapsed
        for step in result.get("steps") or []:
            if not isinstance(step, dict) or not isinstance(step.get("result"), dict):
                continue
            stage_name = str(step.get("stage") or "unknown")
            stage_elapsed = step["result"].get("elapsed_seconds")
            if isinstance(stage_elapsed, (int, float)) and math.isfinite(float(stage_elapsed)):
                timings[f"pipeline_{stage_name}"] = max(0.0, float(stage_elapsed))
        stage_results["production_full_pipeline"] = result
    elif profile.execution_mode == "fixed_mask_downstream":
        analysis_started_at = time.perf_counter()
        if profile.parallel_pose_densepose:
            analysis_results, analysis_timings = _collect_parallel_stages(
                {"pose": functions.pose, "densepose": functions.densepose},
                run_dir,
            )
            stage_results.update(analysis_results)
            timings.update(analysis_timings)
        else:
            for name, action in (("pose", functions.pose), ("densepose", functions.densepose)):
                stage_results[name], timings[name] = _run_timed_stage(action, run_dir)
        timings["analysis_wall"] = time.perf_counter() - analysis_started_at

        stage_results["fusion"], timings["fusion"] = _run_timed_stage(
            functions.fusion,
            run_dir,
        )
        post_started_at = time.perf_counter()
        post_actions = {
            "features": functions.features,
            "tables": functions.tables,
            "qc": functions.qc,
        }
        if profile.parallel_post_fusion:
            post_results, post_timings = _collect_parallel_stages(post_actions, run_dir)
            stage_results.update(post_results)
            timings.update(post_timings)
        else:
            for name, action in post_actions.items():
                stage_results[name], timings[name] = _run_timed_stage(action, run_dir)
        timings["post_fusion_wall"] = time.perf_counter() - post_started_at
    else:
        raise ValueError(f"Unsupported pipeline benchmark execution mode: {profile.execution_mode}")

    timings["total"] = time.perf_counter() - overall_started_at
    return {
        "profile_id": profile.profile_id,
        "execution_mode": profile.execution_mode,
        "status": "complete",
        "config": {
            "mask_source": profile.mask_source,
            "parallel_mask_presentation": profile.parallel_mask_presentation,
            "parallel_pose_densepose": profile.parallel_pose_densepose,
            "parallel_post_fusion": profile.parallel_post_fusion,
            "deploy_environment": {name: value for name, value in profile.environment_overrides},
        },
        "timings_seconds": {key: round(value, 6) for key, value in timings.items()},
        "stages": {name: _summarize_stage_result(result) for name, result in stage_results.items()},
        "artifact_inventory": sorted(_artifact_inventory(run_dir)),
    }


def run_pipeline_profile(
    run_dir: Path,
    profile: PipelineBenchmarkProfile,
    *,
    stage_functions: PipelineStageFunctions | None = None,
) -> dict[str, Any]:
    with _profile_environment(profile):
        return _execute_pipeline_profile(
            run_dir,
            profile,
            stage_functions=stage_functions,
        )


FULL_BENCHMARK_RESPONSE_MAX_BYTES = 256 * 1024
FULL_BENCHMARK_TYPE = "full_pipeline_parity"
FULL_BENCHMARK_UNAVAILABLE_METRICS = (
    {
        "metric": "ground_truth_pose_accuracy",
        "reason": "fixture_has_no_ground_truth_pose_labels; PCK is control agreement",
    },
    {
        "metric": "densepose_pixel_part_iou",
        "reason": "DensePose label and IUV arrays are not persisted",
    },
    {
        "metric": "physical_stride_metrics",
        "reason": (
            "stride length, cadence, ground contact, flight time, and physical displacement "
            "cannot be recovered from the current 2D artifacts"
        ),
    },
    {
        "metric": "3d_gait_metrics",
        "reason": "MMPose emits no world landmarks in this pipeline",
    },
    {
        "metric": "artifact_publish_parallelism",
        "reason": (
            "the isolated benchmark calls the CV pipeline entrypoint directly and does not "
            "publish hosted artifacts; validate this flag in a hosted canary"
        ),
    },
)


def _resolve_full_source_path() -> Path:
    configured = os.getenv("WHODOIRUNLIKE_SAM31_BENCHMARK_SOURCE", "").strip()
    repository_root = Path(__file__).resolve().parents[2]
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            repository_root / "site/public/assets/demos/cole-source.mp4",
            Path("/app/site/public/assets/demos/cole-source.mp4"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("The canonical full pipeline benchmark source is unavailable.")


def _full_runtime_metadata() -> dict[str, Any]:
    image_role = os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate")
    base_processor_commit = os.getenv(
        "WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT",
        EXACT_CANDIDATE_COMMIT,
    )
    base_processor_image_digest = os.getenv(
        "WHODOIRUNLIKE_BASE_PROCESSOR_IMAGE_DIGEST",
        EXACT_CANDIDATE_IMAGE_DIGEST,
    )
    result = {
        "processor_version": os.getenv("WHODOIRUNLIKE_PROCESSOR_VERSION", "unknown"),
        "benchmark_version": os.getenv("WHODOIRUNLIKE_BENCHMARK_VERSION", "unknown"),
        "candidate_commit": os.getenv(
            "WHODOIRUNLIKE_CANDIDATE_COMMIT",
            EXACT_CANDIDATE_COMMIT,
        ),
        "candidate_image_digest": os.getenv(
            "WHODOIRUNLIKE_CANDIDATE_IMAGE_DIGEST",
            EXACT_CANDIDATE_IMAGE_DIGEST,
        ),
        "base_image_role": image_role,
        "base_processor_commit": base_processor_commit,
        "base_processor_image_digest": base_processor_image_digest,
        "code_overlay_commit": os.getenv(
            "WHODOIRUNLIKE_CODE_OVERLAY_COMMIT",
            base_processor_commit,
        ),
        "code_overlay_source": os.getenv(
            "WHODOIRUNLIKE_CODE_OVERLAY_SOURCE",
            "base_image",
        ),
        "code_overlay_reference_image_digest": os.getenv(
            "WHODOIRUNLIKE_CODE_OVERLAY_REFERENCE_IMAGE_DIGEST",
            base_processor_image_digest,
        ),
        "dependency_base_role": os.getenv(
            "WHODOIRUNLIKE_DEPENDENCY_BASE_ROLE",
            image_role,
        ),
        "dependency_base_commit": os.getenv(
            "WHODOIRUNLIKE_DEPENDENCY_BASE_COMMIT",
            base_processor_commit,
        ),
        "dependency_base_image_digest": os.getenv(
            "WHODOIRUNLIKE_DEPENDENCY_BASE_IMAGE_DIGEST",
            base_processor_image_digest,
        ),
    }
    if os.getenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        result["base_contract"] = verify_non_overlay_production_files(image_role)
    try:
        import torch

        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except ModuleNotFoundError:
        pass
    return result


PipelineProfileRunner = Callable[[Path, PipelineBenchmarkProfile], dict[str, Any]]
CandidateMaskStageRunner = Callable[[Path], dict[str, Any]]

_SINK_RUN_ID_PATTERN = re.compile(r"^[a-f0-9-]{32,36}$", re.IGNORECASE)
_SINK_ATTEMPT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FORBIDDEN_PARITY_SINK_HOSTS = frozenset({"api.whodoirunlike.com", "staging-api.whodoirunlike.com"})
_PARITY_SINK_ORIGIN_ENV = "WHODOIRUNLIKE_PARITY_SINK_ORIGIN"
_HANDOFF_BUNDLE_MAX_BYTES = 1024 * 1024 * 1024
_HANDOFF_EXTRACTED_MAX_BYTES = 2 * 1024 * 1024 * 1024
_HANDOFF_MEMBER_MAX_COUNT = 1000
_HANDOFF_MANIFEST_MAX_BYTES = 256 * 1024


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_exact_https_origin(value: Any, *, label: str) -> str:
    origin = str(value or "")
    parsed = urllib.parse.urlsplit(origin)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} must be an exact HTTPS origin.") from exc
    canonical_netloc = str(parsed.hostname or "").lower()
    if port is not None:
        canonical_netloc = f"{canonical_netloc}:{port}"
    canonical_origin = f"https://{canonical_netloc}"
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or origin != canonical_origin
    ):
        raise ValueError(f"{label} must be an exact HTTPS origin.")
    if parsed.hostname.lower() in _FORBIDDEN_PARITY_SINK_HOSTS:
        raise ValueError(f"{label} must not target production or staging.")
    return canonical_origin


def _configured_parity_sink_origin() -> str:
    configured = os.getenv(_PARITY_SINK_ORIGIN_ENV, "")
    if not configured:
        raise RuntimeError(
            f"{_PARITY_SINK_ORIGIN_ENV} must name the exact scratch sink HTTPS origin."
        )
    try:
        return _validate_exact_https_origin(
            configured,
            label=_PARITY_SINK_ORIGIN_ENV,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _validate_artifact_sink(payload: Any) -> dict[str, str] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "callback_base_url",
        "run_id",
        "attempt_id",
    }:
        raise ValueError("artifact_sink must contain callback_base_url, run_id, and attempt_id.")
    callback_base_url = _validate_exact_https_origin(
        payload.get("callback_base_url"),
        label="artifact_sink callback_base_url",
    )
    configured_origin = _configured_parity_sink_origin()
    if callback_base_url != configured_origin:
        raise ValueError(
            "artifact_sink callback_base_url does not match the configured scratch origin."
        )
    run_id = str(payload.get("run_id") or "")
    attempt_id = str(payload.get("attempt_id") or "")
    if not _SINK_RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("artifact_sink run_id is invalid.")
    if not _SINK_ATTEMPT_ID_PATTERN.fullmatch(attempt_id):
        raise ValueError("artifact_sink attempt_id is invalid.")
    return {
        "callback_base_url": callback_base_url,
        "run_id": run_id,
        "attempt_id": attempt_id,
    }


def _sink_secret() -> str:
    secret = os.getenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET is required for artifact handoff."
        )
    return secret


def _sink_artifact_url(sink: dict[str, str], name: str) -> str:
    return (
        f"{sink['callback_base_url']}/v1/artifacts/{sink['run_id']}/"
        f"{urllib.parse.quote(name, safe='')}"
    )


def _put_sink_artifact(
    sink: dict[str, str],
    *,
    name: str,
    path: Path,
) -> None:
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with path.open("rb") as body:
        request = urllib.request.Request(
            (
                f"{sink['callback_base_url']}/v1/jobs/{sink['run_id']}/artifacts/"
                f"{urllib.parse.quote(name, safe='')}"
            ),
            data=body,
            method="PUT",
            headers={
                "Authorization": f"Bearer {_sink_secret()}",
                "X-Processing-Attempt-Id": sink["attempt_id"],
                "Content-Type": content_type,
                "Content-Length": str(path.stat().st_size),
                "User-Agent": "wdirl-parity-handoff/1",
            },
        )
        with urllib.request.urlopen(request, timeout=600):
            return


def _finalize_sink(sink: dict[str, str], *, image_role: str) -> None:
    body = json.dumps(
        {
            "attempt_id": sink["attempt_id"],
            "status": "complete",
            "progress": {"phase": "parity_handoff_complete"},
            "summary": {"benchmark_image_role": image_role},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{sink['callback_base_url']}/v1/jobs/{sink['run_id']}/report",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {_sink_secret()}",
            "X-Processing-Attempt-Id": sink["attempt_id"],
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "User-Agent": "wdirl-parity-handoff/1",
        },
    )
    with urllib.request.urlopen(request, timeout=120):
        return


def _download_sink_artifact(
    sink: dict[str, str],
    *,
    name: str,
    destination: Path,
    max_bytes: int,
) -> None:
    request = urllib.request.Request(
        _sink_artifact_url(sink, name),
        headers={"User-Agent": "wdirl-parity-handoff/1"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with urllib.request.urlopen(request, timeout=600) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Handoff artifact exceeds its size limit: {name}")
            output.write(chunk)


def _bundle_file_inventory(run_dir: Path, profile_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Handoff bundle cannot contain symbolic links.")
        if not path.is_file():
            continue
        relative = Path(profile_id) / path.relative_to(run_dir)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    if not records or len(records) > _HANDOFF_MEMBER_MAX_COUNT:
        raise ValueError("Handoff bundle has an invalid file count.")
    if sum(int(record["bytes"]) for record in records) > _HANDOFF_EXTRACTED_MAX_BYTES:
        raise ValueError("Handoff bundle exceeds its uncompressed size limit.")
    return records


def _create_handoff_bundle(
    *,
    workspace: Path,
    run_dir: Path,
    profile_id: str,
    image_role: str,
    runtime: dict[str, Any],
    fixture: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    handoff_dir = workspace / "_handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"parity_{image_role}.tar.gz"
    manifest_name = f"parity_{image_role}_manifest.json"
    bundle_path = handoff_dir / bundle_name
    records = _bundle_file_inventory(run_dir, profile_id)
    with (
        bundle_path.open("wb") as raw_bundle,
        gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            mtime=0,
            fileobj=raw_bundle,
        ) as compressed_bundle,
        tarfile.open(
            fileobj=compressed_bundle,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive,
    ):
        for record in records:
            source = run_dir / Path(record["path"]).relative_to(profile_id)
            info = tarfile.TarInfo(str(record["path"]))
            info.size = source.stat().st_size
            info.mode = 0o600
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as file_body:
                archive.addfile(info, file_body)
    if bundle_path.stat().st_size > _HANDOFF_BUNDLE_MAX_BYTES:
        raise ValueError("Compressed handoff bundle exceeds its size limit.")
    manifest = {
        "schema_version": 1,
        "image_role": image_role,
        "base_processor_commit": runtime.get("base_processor_commit"),
        "base_processor_image_digest": runtime.get("base_processor_image_digest"),
        "code_overlay_commit": runtime.get("code_overlay_commit"),
        "code_overlay_source": runtime.get("code_overlay_source"),
        "code_overlay_reference_image_digest": runtime.get("code_overlay_reference_image_digest"),
        "dependency_base_role": runtime.get("dependency_base_role"),
        "dependency_base_commit": runtime.get("dependency_base_commit"),
        "dependency_base_image_digest": runtime.get("dependency_base_image_digest"),
        "base_contract": runtime.get("base_contract"),
        "gpu_name": runtime.get("gpu_name"),
        "fixture": {
            "id": fixture.fixture_id,
            "source_sha256": fixture.source_sha256,
            "prompt_sha256": fixture.prompt.raw_sha256,
            "tracklets_sha256": fixture.asset_sha256["tracklets_jsonl"],
            "baseline_mask_sha256": fixture.asset_sha256["baseline_runner_mask_mp4"],
        },
        "profile_id": profile_id,
        "bundle": {
            "name": bundle_name,
            "bytes": bundle_path.stat().st_size,
            "sha256": _sha256_path(bundle_path),
        },
        "files": records,
    }
    manifest_path = handoff_dir / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if manifest_path.stat().st_size > _HANDOFF_MANIFEST_MAX_BYTES:
        raise ValueError("Handoff manifest exceeds its size limit.")
    return bundle_path, manifest_path, manifest


def _safe_extract_handoff_bundle(
    *,
    bundle_path: Path,
    destination: Path,
    manifest: dict[str, Any],
) -> Path:
    expected_records = manifest.get("files")
    if not isinstance(expected_records, list):
        raise ValueError("Handoff manifest files must be a list.")
    expected = {
        str(record.get("path")): record for record in expected_records if isinstance(record, dict)
    }
    if len(expected) != len(expected_records) or len(expected) > _HANDOFF_MEMBER_MAX_COUNT:
        raise ValueError("Handoff manifest has an invalid file inventory.")
    total_bytes = 0
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected):
            raise ValueError("Handoff archive member count does not match its manifest.")
        for member in members:
            member_path = Path(member.name)
            if (
                not member.isfile()
                or member.name not in expected
                or member_path.is_absolute()
                or ".." in member_path.parts
            ):
                raise ValueError("Handoff archive contains an unsafe member.")
            total_bytes += int(member.size)
            if total_bytes > _HANDOFF_EXTRACTED_MAX_BYTES:
                raise ValueError("Handoff archive exceeds its extracted size limit.")
            archive.extract(member, destination, filter="data")
    observed_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if observed_paths != set(expected):
        raise ValueError("Extracted handoff inventory does not match its manifest.")
    for relative, record in expected.items():
        path = destination / relative
        if path.stat().st_size != int(record.get("bytes") or -1) or _sha256_path(
            path
        ) != record.get("sha256"):
            raise ValueError(f"Extracted handoff file failed verification: {relative}")
    profile_id = str(manifest.get("profile_id") or "")
    profile_dir = destination / profile_id
    if not profile_dir.is_dir():
        raise ValueError("Handoff archive is missing its profile directory.")
    return profile_dir


def _publish_handoff_bundle(
    *,
    sink: dict[str, str],
    workspace: Path,
    run_dir: Path,
    profile_id: str,
    image_role: str,
    runtime: dict[str, Any],
    fixture: Any,
) -> dict[str, Any]:
    bundle_path, manifest_path, manifest = _create_handoff_bundle(
        workspace=workspace,
        run_dir=run_dir,
        profile_id=profile_id,
        image_role=image_role,
        runtime=runtime,
        fixture=fixture,
    )
    _put_sink_artifact(sink, name=manifest["bundle"]["name"], path=bundle_path)
    _put_sink_artifact(sink, name=manifest_path.name, path=manifest_path)
    return {
        "status": "published",
        "transport": "cloudflare_r2_scratch_job",
        "image_role": image_role,
        "profile_id": profile_id,
        "bundle": manifest["bundle"],
        "manifest_name": manifest_path.name,
        "file_count": len(manifest["files"]),
    }


def _load_control_handoff(
    *,
    sink: dict[str, str],
    workspace: Path,
    runtime: dict[str, Any],
    fixture: Any,
) -> tuple[Path, dict[str, Any]]:
    handoff_dir = workspace / "_control_handoff"
    manifest_path = handoff_dir / "parity_control_manifest.json"
    _download_sink_artifact(
        sink,
        name=manifest_path.name,
        destination=manifest_path,
        max_bytes=_HANDOFF_MANIFEST_MAX_BYTES,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Control handoff manifest is not valid JSON.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Control handoff manifest schema is unsupported.")
    expected_fixture = {
        "id": fixture.fixture_id,
        "source_sha256": fixture.source_sha256,
        "prompt_sha256": fixture.prompt.raw_sha256,
        "tracklets_sha256": fixture.asset_sha256["tracklets_jsonl"],
        "baseline_mask_sha256": fixture.asset_sha256["baseline_runner_mask_mp4"],
    }
    checks = {
        "role": manifest.get("image_role") == "control",
        "commit": manifest.get("base_processor_commit") == EXACT_CONTROL_COMMIT,
        "digest": manifest.get("base_processor_image_digest") == EXACT_CONTROL_IMAGE_DIGEST,
        "fixture": manifest.get("fixture") == expected_fixture,
        "profile": manifest.get("profile_id") == "production_control",
        "base_contract": isinstance(manifest.get("base_contract"), dict)
        and manifest["base_contract"].get("passed") is True
        and manifest["base_contract"].get("image_role") == "control",
        "gpu": bool(runtime.get("gpu_name"))
        and manifest.get("gpu_name") == runtime.get("gpu_name"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Control handoff provenance failed: " + ", ".join(failed))
    bundle = manifest.get("bundle")
    if not isinstance(bundle, dict) or bundle.get("name") != "parity_control.tar.gz":
        raise ValueError("Control handoff bundle metadata is invalid.")
    bundle_path = handoff_dir / "parity_control.tar.gz"
    _download_sink_artifact(
        sink,
        name=bundle_path.name,
        destination=bundle_path,
        max_bytes=_HANDOFF_BUNDLE_MAX_BYTES,
    )
    if bundle_path.stat().st_size != int(bundle.get("bytes") or -1) or _sha256_path(
        bundle_path
    ) != bundle.get("sha256"):
        raise ValueError("Control handoff bundle failed SHA-256 verification.")
    profile_dir = _safe_extract_handoff_bundle(
        bundle_path=bundle_path,
        destination=handoff_dir / "extracted",
        manifest=manifest,
    )
    source_path = profile_dir / "source_segment.mp4"
    prompt_path = profile_dir / "person_prompt.json"
    if (
        not source_path.is_file()
        or _sha256_path(source_path) != fixture.source_sha256
        or not prompt_path.is_file()
        or _sha256_path(prompt_path) != fixture.prompt.raw_sha256
    ):
        raise ValueError("Control handoff canonical source or prompt hash is invalid.")
    return profile_dir, {
        "status": "verified",
        "transport": "cloudflare_r2_scratch_job",
        "control_commit": manifest["base_processor_commit"],
        "control_image_digest": manifest["base_processor_image_digest"],
        "gpu_name": manifest["gpu_name"],
        "bundle_sha256": bundle["sha256"],
        "file_count": len(manifest["files"]),
        "provenance_checks": checks,
    }


def _publish_gate_result(
    *,
    sink: dict[str, str],
    workspace: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    path = workspace / "_handoff" / "parity_gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if path.stat().st_size > FULL_BENCHMARK_RESPONSE_MAX_BYTES:
        raise ValueError("Parity gate artifact exceeds its size limit.")
    _put_sink_artifact(sink, name=path.name, path=path)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


@contextmanager
def _full_benchmark_workspace():
    configured = os.getenv("WHODOIRUNLIKE_SAM31_BENCHMARK_OUTPUT_ROOT", "").strip()
    bundle_id = uuid.uuid4().hex
    if configured:
        workspace = Path(configured).expanduser() / bundle_id
        workspace.mkdir(parents=True, mode=0o700)
        yield workspace, bundle_id, True
        return
    with tempfile.TemporaryDirectory(prefix="wdirl-full-pipeline-parity-") as temp_name:
        yield Path(temp_name), bundle_id, False


def _run_candidate_mask_publicly(run_dir: Path) -> dict[str, Any]:
    from whodoirunlike.sam31_benchmark import run_candidate_mask_stage

    return run_candidate_mask_stage(run_dir)


def _compare_profile_pair(
    *,
    control_profile: PipelineBenchmarkProfile,
    candidate_profile: PipelineBenchmarkProfile,
    run_dirs: dict[str, Path],
    fixture: Any,
    expected_fps: float,
) -> dict[str, Any]:
    comparison = compare_pipeline_runs(
        run_dirs[control_profile.profile_id],
        run_dirs[candidate_profile.profile_id],
        expected_width=fixture.width,
        expected_height=fixture.height,
        expected_frame_count=fixture.frame_count,
        expected_fps=expected_fps,
    )
    comparison["control_profile_id"] = control_profile.profile_id
    comparison["candidate_profile_id"] = candidate_profile.profile_id
    return comparison


def _comparison_pairs(
    profiles: list[PipelineBenchmarkProfile],
) -> list[tuple[str, PipelineBenchmarkProfile, PipelineBenchmarkProfile]]:
    by_id = {profile.profile_id: profile for profile in profiles}
    three_arm_ids = set(DEFAULT_PIPELINE_PROFILE_MATRIX)
    if three_arm_ids.issubset(by_id):
        baseline = by_id["downstream_baseline_control"]
        candidate_control = by_id["downstream_candidate_control"]
        candidate_optimized = by_id["downstream_candidate_optimized"]
        return [
            ("mask_effect", baseline, candidate_control),
            ("optimization_effect", candidate_control, candidate_optimized),
            ("end_to_end", baseline, candidate_optimized),
        ]
    if len(profiles) == 2:
        return [("profile_comparison", profiles[0], profiles[1])]
    return []


def run_full_pipeline_benchmark(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    profile_runner: PipelineProfileRunner | None = None,
    candidate_mask_runner: CandidateMaskStageRunner | None = None,
) -> dict[str, Any]:
    profiles, assets = validate_full_benchmark_request(payload)
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    source_path = Path(source_path) if source_path is not None else _resolve_full_source_path()
    source_data = source_path.read_bytes()
    if hashlib.sha256(source_data).hexdigest() != fixture.source_sha256:
        raise ValueError("Canonical full pipeline source failed SHA-256 verification.")
    source_meta = inspect_video(source_path)
    if (
        int(source_meta.get("width") or 0) != fixture.width
        or int(source_meta.get("height") or 0) != fixture.height
        or int(source_meta.get("frame_count") or 0) != fixture.frame_count
    ):
        raise ValueError("Canonical full pipeline source metadata does not match the fixture.")

    sink = _validate_artifact_sink(payload.get("artifact_sink"))
    runtime = _full_runtime_metadata()
    image_role = str(runtime.get("base_image_role") or "candidate")
    is_candidate_handoff = image_role in {"candidate", "schedule_only"}
    profile_ids = [profile.profile_id for profile in profiles]
    if sink is not None:
        if image_role == "control":
            if profile_ids != ["production_control"]:
                raise ValueError("Control handoff requires exactly the production_control profile.")
            if (
                runtime.get("base_processor_commit") != EXACT_CONTROL_COMMIT
                or runtime.get("base_processor_image_digest") != EXACT_CONTROL_IMAGE_DIGEST
            ):
                raise ValueError("Control handoff is not running on the exact control image.")
        elif image_role == "candidate":
            if profile_ids != ["production_candidate"]:
                raise ValueError("Candidate handoff requires production_candidate.")
            if (
                runtime.get("base_processor_commit") != EXACT_CANDIDATE_COMMIT
                or runtime.get("base_processor_image_digest") != EXACT_CANDIDATE_IMAGE_DIGEST
            ):
                raise ValueError("Candidate handoff is not running on the exact candidate image.")
        elif image_role == "schedule_only":
            if profile_ids != ["production_candidate_schedule_only"]:
                raise ValueError(
                    "Schedule-only handoff requires production_candidate_schedule_only."
                )
            schedule_provenance = {
                "base_processor_commit": runtime.get("base_processor_commit")
                == EXACT_CONTROL_COMMIT,
                "base_processor_image_digest": runtime.get("base_processor_image_digest")
                == EXACT_CONTROL_IMAGE_DIGEST,
                "code_overlay_commit": runtime.get("code_overlay_commit") == EXACT_CANDIDATE_COMMIT,
                "code_overlay_source": runtime.get("code_overlay_source") == "git_commit",
                "code_overlay_reference_image_digest": runtime.get(
                    "code_overlay_reference_image_digest"
                )
                == EXACT_CANDIDATE_IMAGE_DIGEST,
                "dependency_base_role": runtime.get("dependency_base_role") == "control",
                "dependency_base_commit": runtime.get("dependency_base_commit")
                == EXACT_CONTROL_COMMIT,
                "dependency_base_image_digest": runtime.get("dependency_base_image_digest")
                == EXACT_CONTROL_IMAGE_DIGEST,
            }
            if not all(schedule_provenance.values()):
                raise ValueError("Schedule-only handoff provenance is not exact.")
        else:
            raise ValueError(
                "Artifact handoff requires a control, candidate, or schedule-only image role."
            )

    execute_profile = profile_runner or (
        lambda run_dir, profile: run_pipeline_profile(run_dir, profile)
    )
    execute_candidate_mask = candidate_mask_runner or _run_candidate_mask_publicly
    profile_results: list[dict[str, Any]] = []
    with _full_benchmark_workspace() as (temp_root, bundle_id, artifacts_persisted):
        baseline_probe_dir = temp_root / "_baseline_probe"
        materialize_pipeline_fixture(
            run_dir=baseline_probe_dir,
            source_path=source_path,
            assets=assets,
            profile_id="baseline_probe",
        )
        mask_meta = inspect_video(baseline_probe_dir / "runner_mask.mp4")
        if (
            int(mask_meta.get("width") or 0) != fixture.width
            or int(mask_meta.get("height") or 0) != fixture.height
            or int(mask_meta.get("frame_count") or 0) != fixture.frame_count
        ):
            raise ValueError("Canonical full pipeline mask metadata does not match the fixture.")

        candidate_mask_result: dict[str, Any] | None = None
        candidate_mask_dir: Path | None = None
        if any(profile.mask_source == "candidate" for profile in profiles):
            candidate_mask_dir = temp_root / "_candidate_mask_generation"
            materialize_pipeline_fixture(
                run_dir=candidate_mask_dir,
                source_path=source_path,
                assets=assets,
                profile_id="candidate_mask_generation",
            )
            candidate_mask_result = execute_candidate_mask(candidate_mask_dir)
            if not isinstance(candidate_mask_result, dict):
                raise TypeError("Candidate mask stage runner must return an object.")
            mask_gate = (candidate_mask_result.get("quality_vs_production_baseline") or {}).get(
                "strict_mask_agreement_gate"
            )
            if not isinstance(mask_gate, dict) or not isinstance(mask_gate.get("passed"), bool):
                raise ValueError("Candidate mask stage did not return the strict mask gate.")

        run_dirs: dict[str, Path] = {}
        for profile in profiles:
            run_dir = temp_root / profile.profile_id
            materialize_pipeline_fixture(
                run_dir=run_dir,
                source_path=source_path,
                assets=assets,
                profile_id=profile.profile_id,
            )
            input_mask_sha256: str | None = None
            if profile.mask_source == "baseline":
                input_mask_sha256 = fixture.asset_sha256["baseline_runner_mask_mp4"]
            elif profile.mask_source == "candidate":
                if candidate_mask_dir is None:
                    raise RuntimeError("Candidate mask was not materialized.")
                installed = install_candidate_mask_artifacts(
                    candidate_run_dir=candidate_mask_dir,
                    downstream_run_dir=run_dir,
                )
                input_mask_sha256 = installed["runner_mask"]

            profile_result = execute_profile(run_dir, profile)
            if not isinstance(profile_result, dict):
                raise TypeError("Full pipeline profile runner must return an object.")
            if profile_result.get("profile_id") != profile.profile_id:
                raise ValueError("Full pipeline profile result ID does not match its request.")
            profile_result = dict(profile_result)
            profile_result["mask_source"] = profile.mask_source
            if input_mask_sha256 is not None:
                profile_result["input_mask_sha256"] = input_mask_sha256
            profile_results.append(profile_result)
            run_dirs[profile.profile_id] = run_dir

        comparisons = {
            comparison_id: _compare_profile_pair(
                control_profile=control,
                candidate_profile=candidate,
                run_dirs=run_dirs,
                fixture=fixture,
                expected_fps=float(source_meta.get("fps") or 0.0),
            )
            for comparison_id, control, candidate in _comparison_pairs(profiles)
        }
        handoff: dict[str, Any] = {"status": "not_requested"}
        if sink is not None and is_candidate_handoff:
            candidate_profile_id = profile_ids[0]
            control_profile_dir, control_handoff = _load_control_handoff(
                sink=sink,
                workspace=temp_root,
                runtime=runtime,
                fixture=fixture,
            )
            cross_image = compare_pipeline_runs(
                control_profile_dir,
                run_dirs[candidate_profile_id],
                expected_width=fixture.width,
                expected_height=fixture.height,
                expected_frame_count=fixture.frame_count,
                expected_fps=float(source_meta.get("fps") or 0.0),
            )
            cross_image["control_profile_id"] = "production_control"
            cross_image["candidate_profile_id"] = candidate_profile_id
            cross_image["provenance"] = control_handoff
            comparisons["authoritative_cross_image"] = cross_image

        if sink is not None:
            handoff_profile_id = profile_ids[0]
            handoff = _publish_handoff_bundle(
                sink=sink,
                workspace=temp_root,
                run_dir=run_dirs[handoff_profile_id],
                profile_id=handoff_profile_id,
                image_role=image_role,
                runtime=runtime,
                fixture=fixture,
            )
        downstream_parity = (
            all(bool(comparison["passed"]) for comparison in comparisons.values())
            if comparisons
            else None
        )
        mask_parity = (
            bool(
                candidate_mask_result["quality_vs_production_baseline"][
                    "strict_mask_agreement_gate"
                ]["passed"]
            )
            if candidate_mask_result is not None
            else None
        )
        parity_components = [
            value for value in (mask_parity, downstream_parity) if value is not None
        ]
        parity_passed: bool | None = (
            all(parity_components) if parity_components and comparisons else None
        )
        comparison_summary = {
            "passed": downstream_parity,
            "pair_ids": list(comparisons),
            "availability": "available" if comparisons else "not_requested",
            **({} if comparisons else {"reason": "single_profile_timing_mode"}),
        }

        result: dict[str, Any] = {
            "schema_version": 1,
            "type": FULL_BENCHMARK_TYPE,
            "scope": "full",
            "fixture": {
                "id": fixture.fixture_id,
                "source_sha256": fixture.source_sha256,
                "prompt_sha256": fixture.prompt.raw_sha256,
                "tracklets_sha256": fixture.asset_sha256["tracklets_jsonl"],
                "baseline_mask_sha256": fixture.asset_sha256["baseline_runner_mask_mp4"],
                "frame_count": fixture.frame_count,
                "width": fixture.width,
                "height": fixture.height,
                "fps": round(float(source_meta.get("fps") or 0.0), 6),
            },
            "runtime": runtime,
            "profile_order": [profile.profile_id for profile in profiles],
            "timing_note": (
                "Profiles execute in listed order in one process; use production-reversed "
                "jobs to separate cold-build and hot-cache timing."
            ),
            "candidate_mask": candidate_mask_result
            or {
                "status": "not_requested",
            },
            "artifact_bundle": {
                "id": bundle_id,
                "persisted": artifacts_persisted,
                "profiles": [profile.profile_id for profile in profiles],
            },
            "handoff": handoff,
            "profiles": profile_results,
            "comparisons": comparisons,
            "comparison": comparison_summary,
            "parity_passed": parity_passed,
            "unavailable_metrics": list(FULL_BENCHMARK_UNAVAILABLE_METRICS),
            "response_bytes": 0,
        }
        if sink is not None and is_candidate_handoff:
            gate_payload = {
                "schema_version": 1,
                "type": "authoritative_cross_image_parity",
                "fixture": result["fixture"],
                "runtime": runtime,
                "comparison": comparisons["authoritative_cross_image"],
                "parity_passed": parity_passed,
            }
            result["handoff"]["gates"] = _publish_gate_result(
                sink=sink,
                workspace=temp_root,
                result=gate_payload,
            )
        if sink is not None:
            _finalize_sink(sink, image_role=image_role)
        encoded = json.dumps(
            result,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        result["response_bytes"] = len(encoded)
        encoded = json.dumps(
            result,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        result["response_bytes"] = len(encoded)
        if len(encoded) > FULL_BENCHMARK_RESPONSE_MAX_BYTES:
            raise RuntimeError("Full pipeline benchmark response exceeded its size limit.")
        if artifacts_persisted:
            (temp_root / "benchmark_result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        return result


POSE_PARITY_THRESHOLDS = {
    "usable_agreement_rate_min": 0.99,
    "new_unusable_frame_count_max": 1,
    "pck_at_001_diagonal_min": 0.99,
    "joint_error_median_normalized_max": 0.002,
    "joint_error_p95_normalized_max": 0.01,
    "visibility_mae_max": 0.01,
}

DENSEPOSE_PARITY_THRESHOLDS = {
    "usable_agreement_rate_min": 0.99,
    "new_unusable_frame_count_max": 1,
    "part_jaccard_mean_min": 0.99,
    "part_jaccard_p05_min": 0.95,
    "centroid_error_normalized_mean_max": 0.005,
    "centroid_error_normalized_p95_max": 0.015,
    "bbox_iou_p05_min": 0.95,
    "coverage_mae_max": 0.01,
    "mask_overlap_mae_max": 0.01,
}

FUSION_PARITY_THRESHOLDS = {
    "frame_state_agreement_rate_min": 0.99,
    "risk_state_increase_count_max": 0,
    "usable_agreement_rate_min": 0.99,
    "confidence_mae_max": 0.01,
    "confidence_mean_drop_max": 0.01,
    "joint_weight_mae_max": 0.01,
    "joint_weight_p95_error_max": 0.03,
}

FEATURE_PARITY_THRESHOLDS = {
    "array_max_abs_delta_max": 0.001,
    "valid_frame_loss_count_max": 1,
    "joint_angle_median_abs_error_max": 0.5,
    "joint_angle_p95_abs_error_max": 2.0,
}

QC_PARITY_THRESHOLDS = {
    "numeric_max_abs_delta_max": 0.01,
    "mask_churn_abs_delta_max": 0.01,
    "uncertainty_increase_max": 0.01,
}
VIDEO_PARITY_THRESHOLDS = {"fps_max_abs_delta_max": 0.01}
RUNNER_MASK_PARITY_THRESHOLDS = {
    "iou_mean_min": float(STRICT_MASK_GATE_THRESHOLDS["iou_mean_min"]),
    "iou_p05_min": float(STRICT_MASK_GATE_THRESHOLDS["iou_p05_min"]),
    "boundary_f1_mean_min": float(STRICT_MASK_GATE_THRESHOLDS["boundary_f1_mean_min"]),
    "centroid_error_normalized_mean_max": float(
        STRICT_MASK_GATE_THRESHOLDS["centroid_error_normalized_mean_max"]
    ),
    "coverage_mae_max": 0.01,
    "mask_churn_abs_delta_max": float(
        STRICT_MASK_GATE_THRESHOLDS["temporal_iou_absolute_delta_max"]
    ),
}


def evaluate_pose_parity(
    measurements: PoseParityMeasurements,
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    if expected_frame_count <= 0:
        raise ValueError("Pose parity expected_frame_count must be positive.")

    checks = {
        "control_frame_count_exact": measurements.control_frame_count == expected_frame_count,
        "candidate_frame_count_exact": measurements.candidate_frame_count == expected_frame_count,
        "control_schema_preserved": measurements.control_schema_preserved,
        "required_fields_present": measurements.required_fields_present,
        "frame_indices_aligned": measurements.aligned_frame_count == expected_frame_count,
        "usable_agreement": measurements.usable_agreement_rate is not None
        and measurements.usable_agreement_rate
        >= POSE_PARITY_THRESHOLDS["usable_agreement_rate_min"],
        "new_unusable_frames": measurements.new_unusable_frame_count
        <= POSE_PARITY_THRESHOLDS["new_unusable_frame_count_max"],
        "common_visible_point_evidence": measurements.common_visible_point_count > 0,
        "pck_at_001_diagonal": measurements.pck_at_001_diagonal is not None
        and measurements.pck_at_001_diagonal >= POSE_PARITY_THRESHOLDS["pck_at_001_diagonal_min"],
        "joint_error_median_normalized": measurements.joint_error_median_normalized is not None
        and measurements.joint_error_median_normalized
        <= POSE_PARITY_THRESHOLDS["joint_error_median_normalized_max"],
        "joint_error_p95_normalized": measurements.joint_error_p95_normalized is not None
        and measurements.joint_error_p95_normalized
        <= POSE_PARITY_THRESHOLDS["joint_error_p95_normalized_max"],
        "visibility_mae": measurements.visibility_mae is not None
        and measurements.visibility_mae <= POSE_PARITY_THRESHOLDS["visibility_mae_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {
            "expected_frame_count": expected_frame_count,
            **POSE_PARITY_THRESHOLDS,
        },
    }


def evaluate_densepose_parity(
    measurements: DensePoseParityMeasurements,
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    if expected_frame_count <= 0:
        raise ValueError("DensePose parity expected_frame_count must be positive.")

    checks = {
        "control_frame_count_exact": measurements.control_frame_count == expected_frame_count,
        "candidate_frame_count_exact": measurements.candidate_frame_count == expected_frame_count,
        "control_schema_preserved": measurements.control_schema_preserved,
        "required_fields_present": measurements.required_fields_present,
        "frame_indices_aligned": measurements.aligned_frame_count == expected_frame_count,
        "usable_agreement": measurements.usable_agreement_rate is not None
        and measurements.usable_agreement_rate
        >= DENSEPOSE_PARITY_THRESHOLDS["usable_agreement_rate_min"],
        "new_unusable_frames": measurements.new_unusable_frame_count
        <= DENSEPOSE_PARITY_THRESHOLDS["new_unusable_frame_count_max"],
        "common_usable_frame_evidence": measurements.common_usable_frame_count > 0,
        "part_jaccard_mean": measurements.part_jaccard_mean is not None
        and measurements.part_jaccard_mean >= DENSEPOSE_PARITY_THRESHOLDS["part_jaccard_mean_min"],
        "part_jaccard_p05": measurements.part_jaccard_p05 is not None
        and measurements.part_jaccard_p05 >= DENSEPOSE_PARITY_THRESHOLDS["part_jaccard_p05_min"],
        "centroid_error_normalized_mean": measurements.centroid_error_normalized_mean is not None
        and measurements.centroid_error_normalized_mean
        <= DENSEPOSE_PARITY_THRESHOLDS["centroid_error_normalized_mean_max"],
        "centroid_error_normalized_p95": measurements.centroid_error_normalized_p95 is not None
        and measurements.centroid_error_normalized_p95
        <= DENSEPOSE_PARITY_THRESHOLDS["centroid_error_normalized_p95_max"],
        "bbox_iou_p05": measurements.bbox_iou_p05 is not None
        and measurements.bbox_iou_p05 >= DENSEPOSE_PARITY_THRESHOLDS["bbox_iou_p05_min"],
        "coverage_mae": measurements.coverage_mae is not None
        and measurements.coverage_mae <= DENSEPOSE_PARITY_THRESHOLDS["coverage_mae_max"],
        "mask_overlap_mae": measurements.mask_overlap_mae is not None
        and measurements.mask_overlap_mae <= DENSEPOSE_PARITY_THRESHOLDS["mask_overlap_mae_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {
            "expected_frame_count": expected_frame_count,
            **DENSEPOSE_PARITY_THRESHOLDS,
        },
    }


def evaluate_fusion_parity(
    measurements: FusionParityMeasurements,
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    if expected_frame_count <= 0:
        raise ValueError("Fusion parity expected_frame_count must be positive.")

    checks = {
        "control_frame_count_exact": measurements.control_frame_count == expected_frame_count,
        "candidate_frame_count_exact": measurements.candidate_frame_count == expected_frame_count,
        "schema_match": measurements.schema_match,
        "required_fields_present": measurements.required_fields_present,
        "frame_indices_aligned": measurements.aligned_frame_count == expected_frame_count,
        "frame_state_agreement": measurements.frame_state_agreement_rate is not None
        and measurements.frame_state_agreement_rate
        >= FUSION_PARITY_THRESHOLDS["frame_state_agreement_rate_min"],
        "no_new_risk_states": measurements.risk_state_increase_count
        <= FUSION_PARITY_THRESHOLDS["risk_state_increase_count_max"],
        "usable_agreement": measurements.usable_agreement_rate is not None
        and measurements.usable_agreement_rate
        >= FUSION_PARITY_THRESHOLDS["usable_agreement_rate_min"],
        "confidence_mae": measurements.confidence_mae is not None
        and measurements.confidence_mae <= FUSION_PARITY_THRESHOLDS["confidence_mae_max"],
        "confidence_mean_drop": measurements.confidence_mean_drop is not None
        and measurements.confidence_mean_drop
        <= FUSION_PARITY_THRESHOLDS["confidence_mean_drop_max"],
        "common_joint_weight_evidence": measurements.common_joint_weight_count > 0,
        "joint_weight_mae": measurements.joint_weight_mae is not None
        and measurements.joint_weight_mae <= FUSION_PARITY_THRESHOLDS["joint_weight_mae_max"],
        "joint_weight_p95_error": measurements.joint_weight_p95_error is not None
        and measurements.joint_weight_p95_error
        <= FUSION_PARITY_THRESHOLDS["joint_weight_p95_error_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {
            "expected_frame_count": expected_frame_count,
            **FUSION_PARITY_THRESHOLDS,
        },
    }


def evaluate_feature_parity(
    measurements: FeatureParityMeasurements,
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    if expected_frame_count <= 0:
        raise ValueError("Feature parity expected_frame_count must be positive.")

    checks = {
        "control_frame_count_exact": measurements.control_frame_count == expected_frame_count,
        "candidate_frame_count_exact": measurements.candidate_frame_count == expected_frame_count,
        "npz_keys_match": measurements.npz_keys_match,
        "npz_shapes_match": measurements.npz_shapes_match,
        "npz_dtypes_match": measurements.npz_dtypes_match,
        "array_schema_match": measurements.array_schema_match,
        "comparable_array_evidence": measurements.comparable_array_count > 0,
        "array_max_abs_delta": measurements.array_max_abs_delta is not None
        and measurements.array_max_abs_delta
        <= FEATURE_PARITY_THRESHOLDS["array_max_abs_delta_max"],
        "valid_frame_loss": measurements.valid_frame_loss_count
        <= FEATURE_PARITY_THRESHOLDS["valid_frame_loss_count_max"],
        "joint_angle_evidence": measurements.joint_angle_common_value_count > 0,
        "joint_angle_median_abs_error": measurements.joint_angle_median_abs_error is not None
        and measurements.joint_angle_median_abs_error
        <= FEATURE_PARITY_THRESHOLDS["joint_angle_median_abs_error_max"],
        "joint_angle_p95_abs_error": measurements.joint_angle_p95_abs_error is not None
        and measurements.joint_angle_p95_abs_error
        <= FEATURE_PARITY_THRESHOLDS["joint_angle_p95_abs_error_max"],
        "runner_metric_keys_match": measurements.runner_metric_keys_match,
        "runner_metric_evidence": measurements.comparable_runner_metric_count > 0,
        "runner_metrics_within_tolerance": measurements.runner_metrics_within_tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {
            "expected_frame_count": expected_frame_count,
            **FEATURE_PARITY_THRESHOLDS,
        },
    }


def evaluate_qc_parity(measurements: QcParityMeasurements) -> dict[str, Any]:
    checks = {
        "schema_match": measurements.schema_match,
        "required_components_present": measurements.required_components_present,
        "categorical_match": measurements.categorical_match,
        "numeric_field_evidence": measurements.numeric_field_count > 0,
        "numeric_max_abs_delta": measurements.numeric_max_abs_delta is not None
        and measurements.numeric_max_abs_delta <= QC_PARITY_THRESHOLDS["numeric_max_abs_delta_max"],
        "identity_exact": measurements.identity_exact,
        "mask_churn_abs_delta": measurements.mask_churn_abs_delta is not None
        and measurements.mask_churn_abs_delta <= QC_PARITY_THRESHOLDS["mask_churn_abs_delta_max"],
        "uncertainty_increase": measurements.uncertainty_increase is not None
        and measurements.uncertainty_increase <= QC_PARITY_THRESHOLDS["uncertainty_increase_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": dict(QC_PARITY_THRESHOLDS),
    }


def evaluate_artifact_parity(
    measurements: ArtifactParityMeasurements,
) -> dict[str, Any]:
    checks = {
        "control_required_artifacts_present": measurements.control_required_artifacts_present,
        "candidate_required_artifacts_present": (measurements.candidate_required_artifacts_present),
        "control_inventory_preserved": measurements.control_inventory_preserved,
        "schema_artifact_evidence": measurements.schema_artifact_count > 0,
        "json_control_schema_preserved": measurements.json_control_schema_preserved,
        "parquet_control_schema_preserved": measurements.parquet_control_schema_preserved,
        "parquet_row_counts_match": measurements.parquet_row_counts_match,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {},
    }


def evaluate_video_parity(
    measurements: VideoParityMeasurements,
    *,
    expected_video_count: int,
) -> dict[str, Any]:
    if expected_video_count <= 0:
        raise ValueError("Video parity expected_video_count must be positive.")
    checks = {
        "control_required_videos_present": measurements.control_required_videos_present,
        "candidate_required_videos_present": measurements.candidate_required_videos_present,
        "decoded_video_count_exact": measurements.decoded_video_count == expected_video_count,
        "all_videos_playable": measurements.all_videos_playable,
        "no_blank_frames": measurements.no_blank_frames,
        "dimensions_exact": measurements.dimensions_exact,
        "frame_counts_exact": measurements.frame_counts_exact,
        "profile_metadata_match": measurements.profile_metadata_match,
        "fps_expected_match": measurements.fps_expected_match,
        "fps_max_abs_delta": measurements.fps_max_abs_delta is not None
        and measurements.fps_max_abs_delta <= VIDEO_PARITY_THRESHOLDS["fps_max_abs_delta_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {
            "expected_video_count": expected_video_count,
            **VIDEO_PARITY_THRESHOLDS,
        },
    }


def evaluate_runner_mask_parity(
    measurements: RunnerMaskParityMeasurements,
    *,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
) -> dict[str, Any]:
    if expected_width <= 0 or expected_height <= 0 or expected_frame_count <= 0:
        raise ValueError("Runner-mask parity expectations must be positive.")
    checks = {
        "control_dimensions_exact": (
            measurements.control_width == expected_width
            and measurements.control_height == expected_height
        ),
        "candidate_dimensions_exact": (
            measurements.candidate_width == expected_width
            and measurements.candidate_height == expected_height
        ),
        "control_frame_count_exact": measurements.control_frame_count == expected_frame_count,
        "candidate_frame_count_exact": measurements.candidate_frame_count == expected_frame_count,
        "control_nonempty_frame_count_exact": measurements.control_nonempty_frame_count
        == expected_frame_count,
        "candidate_nonempty_frame_count_exact": measurements.candidate_nonempty_frame_count
        == expected_frame_count,
        "iou_mean": measurements.iou_mean is not None
        and measurements.iou_mean >= RUNNER_MASK_PARITY_THRESHOLDS["iou_mean_min"],
        "iou_p05": measurements.iou_p05 is not None
        and measurements.iou_p05 >= RUNNER_MASK_PARITY_THRESHOLDS["iou_p05_min"],
        "boundary_f1_mean": measurements.boundary_f1_mean is not None
        and measurements.boundary_f1_mean >= RUNNER_MASK_PARITY_THRESHOLDS["boundary_f1_mean_min"],
        "centroid_error_normalized_mean": (
            measurements.centroid_error_normalized_mean is not None
            and measurements.centroid_error_normalized_mean
            <= RUNNER_MASK_PARITY_THRESHOLDS["centroid_error_normalized_mean_max"]
        ),
        "coverage_mae": measurements.coverage_mae is not None
        and measurements.coverage_mae <= RUNNER_MASK_PARITY_THRESHOLDS["coverage_mae_max"],
        "mask_churn_abs_delta": measurements.mask_churn_abs_delta is not None
        and measurements.mask_churn_abs_delta
        <= RUNNER_MASK_PARITY_THRESHOLDS["mask_churn_abs_delta_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": asdict(measurements),
        "thresholds": {
            "expected_width": expected_width,
            "expected_height": expected_height,
            "expected_frame_count": expected_frame_count,
            **RUNNER_MASK_PARITY_THRESHOLDS,
        },
    }


_POSE_ROW_REQUIRED_FIELDS = frozenset({"frame_index", "usable", "visibility_mean", "landmarks"})
_POSE_LANDMARK_REQUIRED_FIELDS = frozenset({"index", "name", "x", "y", "visibility"})
_SCHEMA_DIAGNOSTIC_MAX_PATHS = 16


def _schema_type(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _json_pointer_token(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _value_schema_paths(value: Any) -> dict[str, frozenset[str]]:
    observed: dict[str, set[str]] = {}

    def visit(child: Any, path: str) -> None:
        observed.setdefault(path, set()).add(_schema_type(child))
        if isinstance(child, dict):
            for key, item in child.items():
                visit(item, f"{path}/{_json_pointer_token(key)}")
        elif isinstance(child, list):
            for item in child:
                visit(item, f"{path}/*")

    visit(value, "$")
    return {path: frozenset(types) for path, types in observed.items()}


def _schema_compatibility(control: Any, candidate: Any) -> dict[str, Any]:
    control_schema = _value_schema_paths(control)
    candidate_schema = _value_schema_paths(candidate)
    missing_paths = sorted(set(control_schema) - set(candidate_schema))
    candidate_only_paths = sorted(set(candidate_schema) - set(control_schema))
    type_change_paths = sorted(
        path
        for path in set(control_schema) & set(candidate_schema)
        if control_schema[path] != candidate_schema[path]
    )
    return {
        "exact_match": control_schema == candidate_schema,
        "control_preserved": not missing_paths and not type_change_paths,
        "control_path_count": len(control_schema),
        "candidate_path_count": len(candidate_schema),
        "missing_control_path_count": len(missing_paths),
        "missing_control_paths": missing_paths[:_SCHEMA_DIAGNOSTIC_MAX_PATHS],
        "type_change_count": len(type_change_paths),
        "type_changes": {
            path: {
                "control": sorted(control_schema[path]),
                "candidate": sorted(candidate_schema[path]),
            }
            for path in type_change_paths[:_SCHEMA_DIAGNOSTIC_MAX_PATHS]
        },
        "candidate_only_path_count": len(candidate_only_paths),
        "candidate_only_paths": candidate_only_paths[:_SCHEMA_DIAGNOSTIC_MAX_PATHS],
    }


def _row_and_item_schema(
    rows: list[dict[str, Any]],
    *,
    item_field: str,
) -> tuple[frozenset[str], frozenset[str]]:
    row_fields: set[str] = set()
    item_fields: set[str] = set()
    for row in rows:
        row_fields.update(str(key) for key in row)
        for item in row.get(item_field) or []:
            if isinstance(item, dict):
                item_fields.update(str(key) for key in item)
    return frozenset(row_fields), frozenset(item_fields)


def _rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row.get("frame_index") or 0): row for row in rows}


def _landmarks_by_index(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(landmark.get("index") or 0): landmark
        for landmark in row.get("landmarks") or []
        if isinstance(landmark, dict)
    }


def _visible_xy(landmark: dict[str, Any] | None) -> tuple[float, float, float] | None:
    if not landmark or landmark.get("missing"):
        return None
    try:
        x = float(landmark["x"])
        y = float(landmark["y"])
        visibility = float(landmark.get("visibility") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(visibility)):
        return None
    if visibility < 0.05:
        return None
    return x, y, visibility


def compare_pose_rows(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    control_schema = _row_and_item_schema(control_rows, item_field="landmarks")
    candidate_schema = _row_and_item_schema(candidate_rows, item_field="landmarks")
    schema_compatibility = _schema_compatibility(control_rows, candidate_rows)
    required_fields_present = (
        _POSE_ROW_REQUIRED_FIELDS.issubset(control_schema[0])
        and _POSE_ROW_REQUIRED_FIELDS.issubset(candidate_schema[0])
        and _POSE_LANDMARK_REQUIRED_FIELDS.issubset(control_schema[1])
        and _POSE_LANDMARK_REQUIRED_FIELDS.issubset(candidate_schema[1])
    )

    control_by_frame = _rows_by_frame(control_rows)
    candidate_by_frame = _rows_by_frame(candidate_rows)
    aligned_indices = sorted(set(control_by_frame) & set(candidate_by_frame))
    usable_matches: list[bool] = []
    new_unusable_frames = 0
    pck_hits: list[bool] = []
    normalized_joint_errors: list[float] = []
    visibility_errors: list[float] = []
    for frame_index in aligned_indices:
        control_row = control_by_frame[frame_index]
        candidate_row = candidate_by_frame[frame_index]
        control_usable = bool(control_row.get("usable"))
        candidate_usable = bool(candidate_row.get("usable"))
        usable_matches.append(control_usable == candidate_usable)
        new_unusable_frames += int(control_usable and not candidate_usable)
        control_landmarks = _landmarks_by_index(control_row)
        candidate_landmarks = _landmarks_by_index(candidate_row)
        for landmark_index in sorted(set(control_landmarks) & set(candidate_landmarks)):
            control_point = _visible_xy(control_landmarks[landmark_index])
            candidate_point = _visible_xy(candidate_landmarks[landmark_index])
            if control_point is None or candidate_point is None:
                continue
            normalized_distance = math.hypot(
                control_point[0] - candidate_point[0],
                control_point[1] - candidate_point[1],
            ) / math.sqrt(2.0)
            normalized_joint_errors.append(normalized_distance)
            pck_hits.append(normalized_distance <= 0.01)
            visibility_errors.append(abs(control_point[2] - candidate_point[2]))

    measurements = PoseParityMeasurements(
        control_frame_count=len(control_rows),
        candidate_frame_count=len(candidate_rows),
        schema_match=bool(schema_compatibility["exact_match"]),
        control_schema_preserved=bool(schema_compatibility["control_preserved"]),
        required_fields_present=required_fields_present,
        aligned_frame_count=len(aligned_indices),
        usable_agreement_rate=(float(statistics.fmean(usable_matches)) if usable_matches else None),
        new_unusable_frame_count=new_unusable_frames,
        common_visible_point_count=len(pck_hits),
        pck_at_001_diagonal=(float(statistics.fmean(pck_hits)) if pck_hits else None),
        joint_error_median_normalized=(
            float(statistics.median(normalized_joint_errors)) if normalized_joint_errors else None
        ),
        joint_error_p95_normalized=(
            float(np.percentile(np.asarray(normalized_joint_errors), 95))
            if normalized_joint_errors
            else None
        ),
        visibility_mae=(float(statistics.fmean(visibility_errors)) if visibility_errors else None),
    )
    result = evaluate_pose_parity(measurements, expected_frame_count=expected_frame_count)
    result["schema_compatibility"] = schema_compatibility
    return result


_DENSEPOSE_REQUIRED_FIELDS = frozenset(
    {
        "frame_index",
        "usable",
        "part_ids",
        "part_centroids",
        "densepose_coverage",
        "mask_overlap",
        "bbox",
    }
)


def _row_schema(rows: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(str(key) for row in rows for key in row)


def _centroid_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        x = float(value["x"])
        y = float(value["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _xywh_iou(control: Any, candidate: Any) -> float | None:
    if not isinstance(control, (list, tuple)) or not isinstance(candidate, (list, tuple)):
        return None
    if len(control) != 4 or len(candidate) != 4:
        return None
    try:
        ax, ay, aw, ah = (float(value) for value in control)
        bx, by, bw, bh = (float(value) for value in candidate)
    except (TypeError, ValueError):
        return None
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else None


def compare_densepose_rows(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    control_schema = _row_schema(control_rows)
    candidate_schema = _row_schema(candidate_rows)
    schema_compatibility = _schema_compatibility(control_rows, candidate_rows)
    control_by_frame = _rows_by_frame(control_rows)
    candidate_by_frame = _rows_by_frame(candidate_rows)
    aligned_indices = sorted(set(control_by_frame) & set(candidate_by_frame))
    usable_matches: list[bool] = []
    new_unusable_frames = 0
    common_usable_frames = 0
    part_jaccards: list[float] = []
    centroid_errors: list[float] = []
    bbox_ious: list[float] = []
    coverage_errors: list[float] = []
    mask_overlap_errors: list[float] = []
    for frame_index in aligned_indices:
        control_row = control_by_frame[frame_index]
        candidate_row = candidate_by_frame[frame_index]
        control_usable = bool(control_row.get("usable"))
        candidate_usable = bool(candidate_row.get("usable"))
        usable_matches.append(control_usable == candidate_usable)
        new_unusable_frames += int(control_usable and not candidate_usable)
        if not (control_usable and candidate_usable):
            continue
        common_usable_frames += 1
        control_parts = {int(value) for value in control_row.get("part_ids") or []}
        candidate_parts = {int(value) for value in candidate_row.get("part_ids") or []}
        union = control_parts | candidate_parts
        part_jaccards.append(len(control_parts & candidate_parts) / len(union) if union else 1.0)
        bbox_iou = _xywh_iou(control_row.get("bbox"), candidate_row.get("bbox"))
        if bbox_iou is not None:
            bbox_ious.append(bbox_iou)
        control_centroids = control_row.get("part_centroids") or {}
        candidate_centroids = candidate_row.get("part_centroids") or {}
        if isinstance(control_centroids, dict) and isinstance(candidate_centroids, dict):
            for part_id in sorted(set(control_centroids) & set(candidate_centroids)):
                control_centroid = _centroid_xy(control_centroids[part_id])
                candidate_centroid = _centroid_xy(candidate_centroids[part_id])
                if control_centroid is not None and candidate_centroid is not None:
                    centroid_errors.append(
                        math.hypot(
                            control_centroid[0] - candidate_centroid[0],
                            control_centroid[1] - candidate_centroid[1],
                        )
                    )
        coverage_errors.append(
            abs(
                float(control_row.get("densepose_coverage") or 0.0)
                - float(candidate_row.get("densepose_coverage") or 0.0)
            )
        )
        mask_overlap_errors.append(
            abs(
                float(control_row.get("mask_overlap") or 0.0)
                - float(candidate_row.get("mask_overlap") or 0.0)
            )
        )

    measurements = DensePoseParityMeasurements(
        control_frame_count=len(control_rows),
        candidate_frame_count=len(candidate_rows),
        schema_match=bool(schema_compatibility["exact_match"]),
        control_schema_preserved=bool(schema_compatibility["control_preserved"]),
        required_fields_present=(
            _DENSEPOSE_REQUIRED_FIELDS.issubset(control_schema)
            and _DENSEPOSE_REQUIRED_FIELDS.issubset(candidate_schema)
        ),
        aligned_frame_count=len(aligned_indices),
        usable_agreement_rate=(float(statistics.fmean(usable_matches)) if usable_matches else None),
        new_unusable_frame_count=new_unusable_frames,
        common_usable_frame_count=common_usable_frames,
        part_jaccard_mean=(float(statistics.fmean(part_jaccards)) if part_jaccards else None),
        part_jaccard_p05=(
            float(np.percentile(np.asarray(part_jaccards), 5)) if part_jaccards else None
        ),
        centroid_error_normalized_mean=(
            float(statistics.fmean(centroid_errors)) if centroid_errors else None
        ),
        centroid_error_normalized_p95=(
            float(np.percentile(np.asarray(centroid_errors), 95)) if centroid_errors else None
        ),
        bbox_iou_p05=(float(np.percentile(np.asarray(bbox_ious), 5)) if bbox_ious else None),
        coverage_mae=(float(statistics.fmean(coverage_errors)) if coverage_errors else None),
        mask_overlap_mae=(
            float(statistics.fmean(mask_overlap_errors)) if mask_overlap_errors else None
        ),
    )
    result = evaluate_densepose_parity(
        measurements,
        expected_frame_count=expected_frame_count,
    )
    result["schema_compatibility"] = schema_compatibility
    return result


_FUSION_REQUIRED_FIELDS = frozenset(
    {"frame_index", "frame_state", "usable", "frame_confidence", "joint_weights"}
)
_FUSION_JOINT_REQUIRED_FIELDS = frozenset({"index", "name", "weight"})


def _joint_weights_by_index(row: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for joint in row.get("joint_weights") or []:
        if not isinstance(joint, dict):
            continue
        try:
            result[int(joint.get("index") or 0)] = float(joint["weight"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def compare_fusion_rows(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    control_schema = _row_and_item_schema(control_rows, item_field="joint_weights")
    candidate_schema = _row_and_item_schema(candidate_rows, item_field="joint_weights")
    schema_compatibility = _schema_compatibility(control_rows, candidate_rows)
    control_by_frame = _rows_by_frame(control_rows)
    candidate_by_frame = _rows_by_frame(candidate_rows)
    aligned_indices = sorted(set(control_by_frame) & set(candidate_by_frame))
    state_matches: list[bool] = []
    usable_matches: list[bool] = []
    confidence_errors: list[float] = []
    control_confidences: list[float] = []
    candidate_confidences: list[float] = []
    joint_weight_errors: list[float] = []
    control_risk_states = {"identity_risk": 0, "pose_rejected": 0}
    candidate_risk_states = {"identity_risk": 0, "pose_rejected": 0}
    for frame_index in aligned_indices:
        control_row = control_by_frame[frame_index]
        candidate_row = candidate_by_frame[frame_index]
        control_state = str(control_row.get("frame_state"))
        candidate_state = str(candidate_row.get("frame_state"))
        state_matches.append(control_state == candidate_state)
        if control_state in control_risk_states:
            control_risk_states[control_state] += 1
        if candidate_state in candidate_risk_states:
            candidate_risk_states[candidate_state] += 1
        usable_matches.append(bool(control_row.get("usable")) == bool(candidate_row.get("usable")))
        control_confidence = float(control_row.get("frame_confidence") or 0.0)
        candidate_confidence = float(candidate_row.get("frame_confidence") or 0.0)
        control_confidences.append(control_confidence)
        candidate_confidences.append(candidate_confidence)
        confidence_errors.append(abs(control_confidence - candidate_confidence))
        control_weights = _joint_weights_by_index(control_row)
        candidate_weights = _joint_weights_by_index(candidate_row)
        for joint_index in set(control_weights) & set(candidate_weights):
            joint_weight_errors.append(
                abs(control_weights[joint_index] - candidate_weights[joint_index])
            )

    measurements = FusionParityMeasurements(
        control_frame_count=len(control_rows),
        candidate_frame_count=len(candidate_rows),
        schema_match=bool(schema_compatibility["exact_match"]),
        required_fields_present=(
            _FUSION_REQUIRED_FIELDS.issubset(control_schema[0])
            and _FUSION_REQUIRED_FIELDS.issubset(candidate_schema[0])
            and _FUSION_JOINT_REQUIRED_FIELDS.issubset(control_schema[1])
            and _FUSION_JOINT_REQUIRED_FIELDS.issubset(candidate_schema[1])
        ),
        aligned_frame_count=len(aligned_indices),
        frame_state_agreement_rate=(
            float(statistics.fmean(state_matches)) if state_matches else None
        ),
        risk_state_increase_count=sum(
            max(0, candidate_risk_states[state] - control_risk_states[state])
            for state in control_risk_states
        ),
        usable_agreement_rate=(float(statistics.fmean(usable_matches)) if usable_matches else None),
        confidence_mae=(float(statistics.fmean(confidence_errors)) if confidence_errors else None),
        confidence_mean_drop=(
            float(statistics.fmean(control_confidences))
            - float(statistics.fmean(candidate_confidences))
            if control_confidences and candidate_confidences
            else None
        ),
        common_joint_weight_count=len(joint_weight_errors),
        joint_weight_mae=(
            float(statistics.fmean(joint_weight_errors)) if joint_weight_errors else None
        ),
        joint_weight_p95_error=(
            float(np.percentile(np.asarray(joint_weight_errors), 95))
            if joint_weight_errors
            else None
        ),
    )
    result = evaluate_fusion_parity(
        measurements,
        expected_frame_count=expected_frame_count,
    )
    result["schema_compatibility"] = schema_compatibility
    return result


def compare_feature_artifacts(
    control_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    control_npz_path: Path,
    candidate_npz_path: Path,
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    with (
        np.load(control_npz_path, allow_pickle=False) as control_npz,
        np.load(
            candidate_npz_path,
            allow_pickle=False,
        ) as candidate_npz,
    ):
        control_keys = set(control_npz.files)
        candidate_keys = set(candidate_npz.files)
        common_keys = sorted(control_keys & candidate_keys)
        shapes_match = control_keys == candidate_keys and all(
            control_npz[key].shape == candidate_npz[key].shape for key in common_keys
        )
        dtypes_match = control_keys == candidate_keys and all(
            control_npz[key].dtype == candidate_npz[key].dtype for key in common_keys
        )
        array_deltas: list[float] = []
        array_comparison_invalid = False
        for key in common_keys:
            control_array = np.asarray(control_npz[key])
            candidate_array = np.asarray(candidate_npz[key])
            if control_array.shape != candidate_array.shape:
                array_comparison_invalid = True
                continue
            try:
                control_values = control_array.astype(np.float64, copy=False)
                candidate_values = candidate_array.astype(np.float64, copy=False)
            except (TypeError, ValueError):
                array_comparison_invalid = True
                continue
            control_finite = np.isfinite(control_values)
            candidate_finite = np.isfinite(candidate_values)
            if not np.array_equal(control_finite, candidate_finite):
                array_comparison_invalid = True
                continue
            if key in {"joint_angles", "angular_velocity"}:
                continue
            if control_finite.any():
                array_deltas.append(
                    float(
                        np.max(
                            np.abs(
                                control_values[control_finite] - candidate_values[candidate_finite]
                            )
                        )
                    )
                )
            else:
                array_deltas.append(0.0)

        valid_frame_loss_count = expected_frame_count
        if (
            "valid_frames" in common_keys
            and control_npz["valid_frames"].shape == candidate_npz["valid_frames"].shape
        ):
            control_valid = np.asarray(control_npz["valid_frames"], dtype=bool)
            candidate_valid = np.asarray(candidate_npz["valid_frames"], dtype=bool)
            valid_frame_loss_count = int(np.logical_and(control_valid, ~candidate_valid).sum())

        joint_angle_errors: list[float] = []
        if (
            "joint_angles" in common_keys
            and control_npz["joint_angles"].shape == candidate_npz["joint_angles"].shape
        ):
            control_angles = np.asarray(control_npz["joint_angles"], dtype=np.float64)
            candidate_angles = np.asarray(candidate_npz["joint_angles"], dtype=np.float64)
            common_finite = np.isfinite(control_angles) & np.isfinite(candidate_angles)
            if common_finite.any():
                joint_angle_errors = np.abs(
                    control_angles[common_finite] - candidate_angles[common_finite]
                ).tolist()

    control_metrics = control_metadata.get("summary_features") or {}
    candidate_metrics = candidate_metadata.get("summary_features") or {}
    if not isinstance(control_metrics, dict):
        control_metrics = {}
    if not isinstance(candidate_metrics, dict):
        candidate_metrics = {}
    control_metric_keys = set(control_metrics)
    candidate_metric_keys = set(candidate_metrics)
    metric_deltas: list[float] = []
    metrics_within_tolerance = control_metric_keys == candidate_metric_keys
    duration_seconds = max(
        float(control_metadata.get("duration_seconds") or 0.0),
        1e-9,
    )
    for key in sorted(control_metric_keys & candidate_metric_keys):
        try:
            control_value = float(control_metrics[key])
            candidate_value = float(candidate_metrics[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(control_value) and math.isfinite(candidate_value):
            delta = abs(control_value - candidate_value)
            metric_deltas.append(delta)
            if key == "stride_rhythm_proxy":
                tolerance = 1.0 / duration_seconds
            elif "angle" in key or key.startswith("torso_lean"):
                tolerance = max(1.0, abs(control_value) * 0.02)
            elif key.endswith(("_coverage_mean", "_visibility_rate")):
                tolerance = 0.01
            else:
                tolerance = max(0.01, abs(control_value) * 0.02)
            metrics_within_tolerance = metrics_within_tolerance and delta <= tolerance

    measurements = FeatureParityMeasurements(
        control_frame_count=int(control_metadata.get("frame_count") or 0),
        candidate_frame_count=int(candidate_metadata.get("frame_count") or 0),
        npz_keys_match=control_keys == candidate_keys,
        npz_shapes_match=shapes_match,
        npz_dtypes_match=dtypes_match,
        array_schema_match=(
            control_metadata.get("array_schema") == candidate_metadata.get("array_schema")
            and isinstance(control_metadata.get("array_schema"), dict)
        ),
        comparable_array_count=len(array_deltas),
        array_max_abs_delta=(
            None if array_comparison_invalid else max(array_deltas) if array_deltas else None
        ),
        valid_frame_loss_count=valid_frame_loss_count,
        joint_angle_common_value_count=len(joint_angle_errors),
        joint_angle_median_abs_error=(
            float(statistics.median(joint_angle_errors)) if joint_angle_errors else None
        ),
        joint_angle_p95_abs_error=(
            float(np.percentile(np.asarray(joint_angle_errors), 95)) if joint_angle_errors else None
        ),
        runner_metric_keys_match=control_metric_keys == candidate_metric_keys,
        comparable_runner_metric_count=len(metric_deltas),
        runner_metric_max_abs_delta=max(metric_deltas) if metric_deltas else None,
        runner_metrics_within_tolerance=metrics_within_tolerance and bool(metric_deltas),
    )
    return evaluate_feature_parity(
        measurements,
        expected_frame_count=expected_frame_count,
    )


_VOLATILE_PAYLOAD_FIELDS = frozenset(
    {
        "candidate_id",
        "created_at",
        "updated_at",
        "completed_at",
        "output_path",
        "input_video",
        "matching_clip_path",
    }
)
_QC_REQUIRED_COMPONENTS = frozenset({"identity", "mask", "pose", "fused", "uncertainty_score"})


def _flatten_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, str], dict[str, float], dict[str, Any]]:
    schema: dict[str, str] = {}
    numeric: dict[str, float] = {}
    categorical: dict[str, Any] = {}

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key)
                if normalized_key in _VOLATILE_PAYLOAD_FIELDS or normalized_key.endswith("_path"):
                    continue
                visit(child, (*path, normalized_key))
            return
        dotted = ".".join(path)
        if isinstance(value, bool):
            schema[dotted] = "boolean"
            categorical[dotted] = value
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            schema[dotted] = "number"
            numeric[dotted] = float(value)
        else:
            schema[dotted] = "categorical"
            categorical[dotted] = value

    visit(payload, ())
    return schema, numeric, categorical


def compare_qc_payloads(
    control_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    control_schema, control_numeric, control_categorical = _flatten_payload(control_payload)
    candidate_schema, candidate_numeric, candidate_categorical = _flatten_payload(candidate_payload)
    numeric_keys = set(control_numeric) & set(candidate_numeric)
    numeric_deltas = [abs(control_numeric[key] - candidate_numeric[key]) for key in numeric_keys]
    control_mask = control_payload.get("mask") or {}
    candidate_mask = candidate_payload.get("mask") or {}
    try:
        mask_churn_abs_delta = abs(
            float(control_mask["mean_mask_churn"]) - float(candidate_mask["mean_mask_churn"])
        )
    except (KeyError, TypeError, ValueError):
        mask_churn_abs_delta = None
    try:
        uncertainty_increase = float(candidate_payload["uncertainty_score"]) - float(
            control_payload["uncertainty_score"]
        )
    except (KeyError, TypeError, ValueError):
        uncertainty_increase = None
    measurements = QcParityMeasurements(
        schema_match=control_schema == candidate_schema,
        required_components_present=(
            _QC_REQUIRED_COMPONENTS.issubset(control_payload)
            and _QC_REQUIRED_COMPONENTS.issubset(candidate_payload)
            and all(
                isinstance(control_payload.get(component), dict)
                and isinstance(candidate_payload.get(component), dict)
                for component in ("identity", "mask", "pose", "fused")
            )
        ),
        categorical_match=control_categorical == candidate_categorical,
        numeric_field_count=len(numeric_deltas),
        numeric_max_abs_delta=max(numeric_deltas) if numeric_deltas else None,
        identity_exact=(
            isinstance(control_payload.get("identity"), dict)
            and control_payload.get("identity") == candidate_payload.get("identity")
        ),
        mask_churn_abs_delta=mask_churn_abs_delta,
        uncertainty_increase=uncertainty_increase,
    )
    return evaluate_qc_parity(measurements)


DEFAULT_REQUIRED_ARTIFACTS = frozenset(
    {
        "cv_run_manifest.json",
        "source_segment.mp4",
        "person_prompt.json",
        "tracklets.jsonl",
        "runner_mask.mp4",
        "masks.jsonl",
        "pose_landmarks.jsonl",
        "mmpose_landmarks.jsonl",
        "skeleton_render.mp4",
        "densepose.jsonl",
        "qa_overlay.mp4",
        "fused_form.jsonl",
        "fused_overlay.mp4",
        "features.json",
        "form_features.json",
        "form_features.npz",
        "poses.parquet",
        "densepose.parquet",
        "fused_form.parquet",
        "qc_metrics.json",
    }
)
DEFAULT_REQUIRED_VIDEOS = frozenset(
    {"runner_mask.mp4", "skeleton_render.mp4", "qa_overlay.mp4", "fused_overlay.mp4"}
)


def _artifact_inventory(run_dir: Path) -> set[str]:
    return {path.name for path in Path(run_dir).iterdir() if path.is_file()}


def _json_artifact_value(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def _parquet_schema_compatibility(control_schema: Any, candidate_schema: Any) -> dict[str, Any]:
    control_fields = list(control_schema)
    candidate_fields = list(candidate_schema)
    candidate_by_name = {field.name: field for field in candidate_fields}
    candidate_positions = {field.name: index for index, field in enumerate(candidate_fields)}
    missing_fields = [field.name for field in control_fields if field.name not in candidate_by_name]
    type_change_fields = [
        field.name
        for field in control_fields
        if field.name in candidate_by_name
        and (
            not field.type.equals(candidate_by_name[field.name].type)
            or field.nullable != candidate_by_name[field.name].nullable
        )
    ]
    retained_positions = [
        candidate_positions[field.name]
        for field in control_fields
        if field.name in candidate_positions
    ]
    order_preserved = retained_positions == sorted(retained_positions)
    control_names = {field.name for field in control_fields}
    candidate_only_fields = [
        field.name for field in candidate_fields if field.name not in control_names
    ]
    exact_match = control_schema.equals(candidate_schema, check_metadata=False)
    return {
        "exact_match": exact_match,
        "control_preserved": not missing_fields and not type_change_fields and order_preserved,
        "control_field_count": len(control_fields),
        "candidate_field_count": len(candidate_fields),
        "missing_control_field_count": len(missing_fields),
        "missing_control_fields": missing_fields[:_SCHEMA_DIAGNOSTIC_MAX_PATHS],
        "type_change_count": len(type_change_fields),
        "type_changes": {
            name: {
                "control": str(control_schema.field(name)),
                "candidate": str(candidate_schema.field(name)),
            }
            for name in type_change_fields[:_SCHEMA_DIAGNOSTIC_MAX_PATHS]
        },
        "control_field_order_preserved": order_preserved,
        "candidate_only_field_count": len(candidate_only_fields),
        "candidate_only_fields": candidate_only_fields[:_SCHEMA_DIAGNOSTIC_MAX_PATHS],
    }


def compare_artifact_contracts(
    control_dir: Path,
    candidate_dir: Path,
    *,
    required_artifacts: set[str] | frozenset[str] = DEFAULT_REQUIRED_ARTIFACTS,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    control_inventory = _artifact_inventory(control_dir)
    candidate_inventory = _artifact_inventory(candidate_dir)
    json_names = sorted(
        name for name in control_inventory if Path(name).suffix in {".json", ".jsonl"}
    )
    parquet_names = sorted(name for name in control_inventory if Path(name).suffix == ".parquet")
    json_schema_match = True
    json_control_schema_preserved = True
    json_schema_compatibility: dict[str, dict[str, Any]] = {}
    for name in json_names:
        try:
            compatibility = _schema_compatibility(
                _json_artifact_value(control_dir / name),
                _json_artifact_value(candidate_dir / name),
            )
            json_schema_compatibility[name] = compatibility
            json_schema_match = json_schema_match and bool(compatibility["exact_match"])
            json_control_schema_preserved = json_control_schema_preserved and bool(
                compatibility["control_preserved"]
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            json_schema_match = False
            json_control_schema_preserved = False
            json_schema_compatibility[name] = {
                "exact_match": False,
                "control_preserved": False,
                "reason": "invalid_or_missing_json_artifact",
            }

    parquet_schema_match = True
    parquet_control_schema_preserved = True
    parquet_row_counts_match = True
    parquet_schema_compatibility: dict[str, dict[str, Any]] = {}
    for name in parquet_names:
        try:
            control_table = pq.read_table(control_dir / name)
            candidate_table = pq.read_table(candidate_dir / name)
        except Exception:
            parquet_schema_match = False
            parquet_control_schema_preserved = False
            parquet_row_counts_match = False
            parquet_schema_compatibility[name] = {
                "exact_match": False,
                "control_preserved": False,
                "reason": "invalid_or_missing_parquet_artifact",
            }
            continue
        compatibility = _parquet_schema_compatibility(
            control_table.schema,
            candidate_table.schema,
        )
        parquet_schema_compatibility[name] = compatibility
        parquet_schema_match = parquet_schema_match and bool(compatibility["exact_match"])
        parquet_control_schema_preserved = parquet_control_schema_preserved and bool(
            compatibility["control_preserved"]
        )
        if control_table.num_rows != candidate_table.num_rows:
            parquet_row_counts_match = False

    measurements = ArtifactParityMeasurements(
        control_required_artifacts_present=set(required_artifacts).issubset(control_inventory),
        candidate_required_artifacts_present=set(required_artifacts).issubset(candidate_inventory),
        inventory_match=control_inventory == candidate_inventory,
        control_inventory_preserved=control_inventory.issubset(candidate_inventory),
        schema_artifact_count=len(json_names) + len(parquet_names),
        json_schema_match=json_schema_match,
        json_control_schema_preserved=json_control_schema_preserved,
        parquet_schema_match=parquet_schema_match,
        parquet_control_schema_preserved=parquet_control_schema_preserved,
        parquet_row_counts_match=parquet_row_counts_match,
    )
    result = evaluate_artifact_parity(measurements)
    result["inventory"] = sorted(control_inventory & candidate_inventory)
    result["inventory_only_in_control"] = sorted(control_inventory - candidate_inventory)
    result["inventory_only_in_candidate"] = sorted(candidate_inventory - control_inventory)
    result["schema_compatibility"] = {
        "json": json_schema_compatibility,
        "parquet": parquet_schema_compatibility,
    }
    return result


def compare_runner_mask_videos(
    control_path: Path,
    candidate_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
) -> dict[str, Any]:
    from whodoirunlike.sam31_benchmark import compare_masks_to_production_baseline

    control_meta, control_masks = iter_mask_video(control_path)
    candidate_meta, candidate_masks = iter_mask_video(candidate_path)
    quality = compare_masks_to_production_baseline(
        candidate_masks,
        control_masks,
        track_boxes={},
        expected_frame_count=expected_frame_count,
    )
    coverage_errors = [
        abs(float(np.mean(control_mask > 0)) - float(np.mean(candidate_mask > 0)))
        for control_mask, candidate_mask in zip(control_masks, candidate_masks)
    ]
    measurements = RunnerMaskParityMeasurements(
        control_width=int(control_meta["width"]),
        control_height=int(control_meta["height"]),
        candidate_width=int(candidate_meta["width"]),
        candidate_height=int(candidate_meta["height"]),
        control_frame_count=len(control_masks),
        candidate_frame_count=len(candidate_masks),
        control_nonempty_frame_count=sum(int(bool(np.any(mask))) for mask in control_masks),
        candidate_nonempty_frame_count=sum(int(bool(np.any(mask))) for mask in candidate_masks),
        iou_mean=quality["iou"]["mean"],
        iou_p05=quality["iou"]["p05"],
        boundary_f1_mean=quality["boundary_f1_2px_mean"],
        centroid_error_normalized_mean=quality["centroid_error_normalized_mean"],
        coverage_mae=(float(statistics.fmean(coverage_errors)) if coverage_errors else None),
        mask_churn_abs_delta=quality["temporal_iou"]["absolute_delta"],
    )
    result = evaluate_runner_mask_parity(
        measurements,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_frame_count=expected_frame_count,
    )
    result["worst_frame_indices"] = quality["worst_frame_indices"]
    result["baseline_is_lossy_mp4_not_ground_truth"] = True
    return result


def _decode_video_contract(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "playable": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "decoded_frames": 0,
            "nonblank_frames": 0,
            "consistent_dimensions": False,
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = 0
    height = 0
    decoded_frames = 0
    nonblank_frames = 0
    consistent_dimensions = True
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_height, frame_width = frame.shape[:2]
            if decoded_frames == 0:
                width = int(frame_width)
                height = int(frame_height)
            elif (frame_width, frame_height) != (width, height):
                consistent_dimensions = False
            decoded_frames += 1
            nonblank_frames += int(bool(np.any(frame)))
    finally:
        capture.release()
    return {
        "playable": decoded_frames > 0,
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frames": decoded_frames,
        "nonblank_frames": nonblank_frames,
        "consistent_dimensions": consistent_dimensions,
    }


def compare_video_contracts(
    control_dir: Path,
    candidate_dir: Path,
    *,
    required_videos: set[str] | frozenset[str] = DEFAULT_REQUIRED_VIDEOS,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
    expected_fps: float,
) -> dict[str, Any]:
    control_inventory = _artifact_inventory(control_dir)
    candidate_inventory = _artifact_inventory(candidate_dir)
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    decoded_summaries: list[dict[str, Any]] = []
    profile_metadata_match = True
    fps_deltas: list[float] = []
    for name in sorted(required_videos):
        control_summary = _decode_video_contract(control_dir / name)
        candidate_summary = _decode_video_contract(candidate_dir / name)
        summaries[name] = {
            "control": control_summary,
            "candidate": candidate_summary,
        }
        decoded_summaries.extend([control_summary, candidate_summary])
        profile_metadata_match = profile_metadata_match and (
            control_summary["width"] == candidate_summary["width"]
            and control_summary["height"] == candidate_summary["height"]
            and control_summary["decoded_frames"] == candidate_summary["decoded_frames"]
        )
        if control_summary["playable"] and candidate_summary["playable"]:
            fps_deltas.append(abs(control_summary["fps"] - candidate_summary["fps"]))

    measurements = VideoParityMeasurements(
        control_required_videos_present=set(required_videos).issubset(control_inventory),
        candidate_required_videos_present=set(required_videos).issubset(candidate_inventory),
        decoded_video_count=sum(int(summary["playable"]) for summary in decoded_summaries),
        all_videos_playable=all(summary["playable"] for summary in decoded_summaries),
        no_blank_frames=all(
            summary["nonblank_frames"] == summary["decoded_frames"] for summary in decoded_summaries
        ),
        dimensions_exact=all(
            summary["consistent_dimensions"]
            and summary["width"] == expected_width
            and summary["height"] == expected_height
            for summary in decoded_summaries
        ),
        frame_counts_exact=all(
            summary["decoded_frames"] == expected_frame_count for summary in decoded_summaries
        ),
        profile_metadata_match=profile_metadata_match,
        fps_expected_match=all(
            abs(summary["fps"] - expected_fps) <= VIDEO_PARITY_THRESHOLDS["fps_max_abs_delta_max"]
            for summary in decoded_summaries
        ),
        fps_max_abs_delta=max(fps_deltas) if fps_deltas else None,
    )
    result = evaluate_video_parity(
        measurements,
        expected_video_count=len(required_videos) * 2,
    )
    result["videos"] = summaries
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected JSONL object rows")
    return rows


def _unavailable_comparison(*artifact_names: str) -> dict[str, Any]:
    return {
        "passed": False,
        "availability": "unavailable",
        "reason": "missing_required_artifacts",
        "missing_artifacts": sorted(set(artifact_names)),
    }


def _available_comparison(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "availability": "available"}


def _invalid_comparison(exc: BaseException) -> dict[str, Any]:
    return {
        "passed": False,
        "availability": "invalid",
        "reason": type(exc).__name__,
    }


def compare_pipeline_runs(
    control_dir: Path,
    candidate_dir: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
    expected_fps: float,
) -> dict[str, Any]:
    control_dir = Path(control_dir)
    candidate_dir = Path(candidate_dir)
    comparisons: dict[str, dict[str, Any]] = {}

    def compare_jsonl_family(
        name: str,
        filename: str,
        comparison: Any,
    ) -> None:
        control_path = control_dir / filename
        candidate_path = candidate_dir / filename
        missing = [filename for path in (control_path, candidate_path) if not path.is_file()]
        if missing:
            comparisons[name] = _unavailable_comparison(*missing)
            return
        try:
            comparisons[name] = _available_comparison(
                comparison(
                    _read_jsonl(control_path),
                    _read_jsonl(candidate_path),
                    expected_frame_count=expected_frame_count,
                )
            )
        except Exception as exc:
            comparisons[name] = _invalid_comparison(exc)

    runner_mask_name = "runner_mask.mp4"
    control_runner_mask = control_dir / runner_mask_name
    candidate_runner_mask = candidate_dir / runner_mask_name
    if not control_runner_mask.is_file() or not candidate_runner_mask.is_file():
        comparisons["runner_mask"] = _unavailable_comparison(runner_mask_name)
    else:
        try:
            comparisons["runner_mask"] = _available_comparison(
                compare_runner_mask_videos(
                    control_runner_mask,
                    candidate_runner_mask,
                    expected_width=expected_width,
                    expected_height=expected_height,
                    expected_frame_count=expected_frame_count,
                )
            )
        except Exception as exc:
            comparisons["runner_mask"] = _invalid_comparison(exc)

    compare_jsonl_family("pose", "pose_landmarks.jsonl", compare_pose_rows)
    compare_jsonl_family("densepose", "densepose.jsonl", compare_densepose_rows)
    compare_jsonl_family("fusion", "fused_form.jsonl", compare_fusion_rows)

    feature_names = ("form_features.json", "form_features.npz")
    if not all(
        (directory / filename).is_file()
        for directory in (control_dir, candidate_dir)
        for filename in feature_names
    ):
        comparisons["features"] = _unavailable_comparison(*feature_names)
    else:
        try:
            comparisons["features"] = _available_comparison(
                compare_feature_artifacts(
                    _read_json(control_dir / feature_names[0]),
                    _read_json(candidate_dir / feature_names[0]),
                    control_dir / feature_names[1],
                    candidate_dir / feature_names[1],
                    expected_frame_count=expected_frame_count,
                )
            )
        except Exception as exc:
            comparisons["features"] = _invalid_comparison(exc)

    qc_name = "qc_metrics.json"
    if not (control_dir / qc_name).is_file() or not (candidate_dir / qc_name).is_file():
        comparisons["qc"] = _unavailable_comparison(qc_name)
    else:
        try:
            comparisons["qc"] = _available_comparison(
                compare_qc_payloads(
                    _read_json(control_dir / qc_name),
                    _read_json(candidate_dir / qc_name),
                )
            )
        except Exception as exc:
            comparisons["qc"] = _invalid_comparison(exc)

    try:
        comparisons["artifacts"] = _available_comparison(
            compare_artifact_contracts(control_dir, candidate_dir)
        )
    except Exception as exc:
        comparisons["artifacts"] = _invalid_comparison(exc)
    try:
        comparisons["videos"] = _available_comparison(
            compare_video_contracts(
                control_dir,
                candidate_dir,
                expected_width=expected_width,
                expected_height=expected_height,
                expected_frame_count=expected_frame_count,
                expected_fps=expected_fps,
            )
        )
    except Exception as exc:
        comparisons["videos"] = _invalid_comparison(exc)

    return {
        "passed": all(comparison.get("passed") is True for comparison in comparisons.values()),
        "comparisons": comparisons,
    }
