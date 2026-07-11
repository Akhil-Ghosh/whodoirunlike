from __future__ import annotations

import copy
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from whodoirunlike.artifact_tables import export_cv_tables
from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.form_features import compile_form_features
from whodoirunlike.fusion_runner import run_fused_form
from whodoirunlike.identity_runner import DEFAULT_IDENTITY_BACKEND, run_identity_tracking
from whodoirunlike.processing_telemetry import ProcessingTelemetry
from whodoirunlike.qc import run_qc_metrics
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam31_mlx_runner import run_sam31_mlx_mask
from whodoirunlike.video_io import make_browser_playable_mp4


REPO_ROOT = Path(__file__).resolve().parents[2]
DENSEPOSE_CONFIG_ENV = "DENSEPOSE_CONFIG"
DENSEPOSE_WEIGHTS_ENV = "DENSEPOSE_WEIGHTS"
DENSEPOSE_DEVICE_ENV = "DENSEPOSE_DEVICE"
DENSEPOSE_INPUT_MIN_SIZE_TEST_ENV = "DENSEPOSE_INPUT_MIN_SIZE_TEST"
DENSEPOSE_INPUT_MAX_SIZE_TEST_ENV = "DENSEPOSE_INPUT_MAX_SIZE_TEST"
DENSEPOSE_TARGET_CROP_ENABLED_ENV = "DENSEPOSE_TARGET_CROP_ENABLED"
DENSEPOSE_TARGET_CROP_PADDING_RATIO_ENV = "DENSEPOSE_TARGET_CROP_PADDING_RATIO"
DENSEPOSE_TARGET_CROP_PADDING_PIXELS_ENV = "DENSEPOSE_TARGET_CROP_PADDING_PIXELS"
DENSEPOSE_DEFAULT_CONFIG = (
    REPO_ROOT / "models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
)
DENSEPOSE_DEFAULT_WEIGHTS = (
    REPO_ROOT / "models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl"
)
INLINE_MASK_BACKENDS = frozenset(
    {"yolo26n_seg_inline", "yolo26n-seg-inline", "yolo26n_seg", "yolo26n-seg"}
)
DEFAULT_INLINE_SEGMENTATION_MODEL = "yolo26n-seg.pt"


def _env_value(name: str) -> str:
    import os

    return os.getenv(name, "").strip()


def _env_bool_value(name: str, default: bool = False) -> bool:
    value = _env_value(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_optional_positive_int(name: str) -> int | None:
    value = _env_value(name)
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _env_nonnegative_int(name: str, default: int) -> int:
    value = _env_value(name)
    parsed = int(value) if value else default
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _env_nonnegative_float(name: str, default: float) -> float:
    value = _env_value(name)
    parsed = float(value) if value else default
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _resolve_repo_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (REPO_ROOT / raw_path).resolve()


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _uses_inline_segmentation(mask_backend: str) -> bool:
    return mask_backend.strip().lower() in INLINE_MASK_BACKENDS


def _densepose_runtime_kwargs() -> dict[str, Any]:
    config_value = _env_value(DENSEPOSE_CONFIG_ENV)
    weights_value = _env_value(DENSEPOSE_WEIGHTS_ENV)
    device = _env_value(DENSEPOSE_DEVICE_ENV) or "cpu"

    config_path = _resolve_repo_path(config_value) if config_value else DENSEPOSE_DEFAULT_CONFIG
    if weights_value:
        weights_path = weights_value if _is_url(weights_value) else str(_resolve_repo_path(weights_value))
    else:
        weights_path = str(DENSEPOSE_DEFAULT_WEIGHTS)

    return {
        "config_path": config_path,
        "weights_path": weights_path,
        "device": device,
        "input_min_size_test": _env_optional_positive_int(
            DENSEPOSE_INPUT_MIN_SIZE_TEST_ENV
        ),
        "input_max_size_test": _env_optional_positive_int(
            DENSEPOSE_INPUT_MAX_SIZE_TEST_ENV
        ),
        "target_crop_enabled": _env_bool_value(
            DENSEPOSE_TARGET_CROP_ENABLED_ENV
        ),
        "target_crop_padding_ratio": _env_nonnegative_float(
            DENSEPOSE_TARGET_CROP_PADDING_RATIO_ENV,
            0.2,
        ),
        "target_crop_padding_pixels": _env_nonnegative_int(
            DENSEPOSE_TARGET_CROP_PADDING_PIXELS_ENV,
            16,
        ),
    }


def _run_mask_stage(
    *,
    run_dir: Path,
    mask_backend: str,
    mask_quality_mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    normalized = mask_backend.strip().lower()
    if normalized in {"sam31", "sam31_mlx", "sam3.1_mlx", "mlx"}:
        return run_sam31_mlx_mask(
            run_dir=run_dir,
            quality_mode=mask_quality_mode,
            progress_callback=progress_callback,
        )
    if normalized in {"sam31_gpu", "sam3.1_gpu", "sam31_cuda", "sam3.1_cuda"}:
        from whodoirunlike.sam31_gpu_runner import run_sam31_gpu_mask

        return run_sam31_gpu_mask(run_dir=run_dir, progress_callback=progress_callback)
    raise ValueError("mask_backend must be one of: sam31_mlx, sam31_gpu")


_FAILURE_STATUSES = frozenset({"failed", "failure", "unavailable", "error"})
_MASK_ARTIFACT_KEYS = (
    "runner_mask",
    "masked_runner",
    "qa_overlay",
    "runner_mask_metadata",
    "masks_jsonl",
)


def _rewrite_staged_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_staged_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_staged_paths(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _run_staged_mask_fallback(
    *,
    run_dir: Path,
    mask_backend: str,
    mask_quality_mode: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a conditional SAM fallback without mutating the valid inline result.

    SAM's runners normally write canonical artifacts in place. A scratch manifest
    redirects those writes into a temporary directory. Only a complete artifact
    contract is atomically copied over the inline files; failures leave the source
    manifest and YOLO artifacts unchanged.
    """

    source_run = RunningClipRun(run_dir)
    source_manifest = source_run.read_manifest()
    destination_paths = {
        key: source_run.artifact_path(key, source_manifest)
        for key in _MASK_ARTIFACT_KEYS
    }

    with tempfile.TemporaryDirectory(prefix=".sam-fallback-", dir=run_dir) as staging_name:
        staging_dir = Path(staging_name)
        staged_manifest = copy.deepcopy(source_manifest)
        staged_paths = dict(staged_manifest.get("paths") or {})
        staged_artifacts = staging_dir / "artifacts"
        staged_artifacts.mkdir(parents=True, exist_ok=True)
        for key, destination in destination_paths.items():
            staged_paths[key] = str(staged_artifacts / destination.name)
        staged_manifest["paths"] = staged_paths
        RunningClipRun(staging_dir).write_manifest(staged_manifest)

        staged_result = _run_mask_stage(
            run_dir=staging_dir,
            mask_backend=mask_backend,
            mask_quality_mode=mask_quality_mode,
            progress_callback=progress_callback,
        )
        result_status = str(staged_result.get("status") or "complete").lower()
        if result_status in _FAILURE_STATUSES:
            raise RuntimeError(
                str(staged_result.get("error") or f"runner_mask returned {result_status}")
            )

        missing = [
            key
            for key in _MASK_ARTIFACT_KEYS
            if not Path(staged_paths[key]).is_file()
        ]
        if missing:
            raise RuntimeError(
                "SAM fallback did not complete the mask artifact contract: "
                + ", ".join(missing)
            )

        replacements = {
            staged_paths[key]: str(destination_paths[key])
            for key in _MASK_ARTIFACT_KEYS
        }
        completed_manifest = RunningClipRun(staging_dir).read_manifest()
        completed_stages = completed_manifest.get("stages") or {}
        stage_updates = {
            key: _rewrite_staged_paths(completed_stages[key], replacements)
            for key in ("whole_runner_mask", "renders")
            if isinstance(completed_stages.get(key), dict)
        }
        promoted_mask_stage = stage_updates.get("whole_runner_mask")
        if isinstance(promoted_mask_stage, dict):
            promoted_mask_stage.pop("deferred_browser_encoding", None)

        backup_dir = staging_dir / "backups"
        backups: dict[str, Path | None] = {}
        for key, destination in destination_paths.items():
            if destination.is_file():
                backup_path = backup_dir / key / destination.name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup_path)
                backups[key] = backup_path
            else:
                backups[key] = None

        try:
            for key, destination in destination_paths.items():
                _atomic_copy(Path(staged_paths[key]), destination)

            manifest_for_promotion = source_run.read_manifest()
            promoted_paths = dict(manifest_for_promotion.get("paths") or {})
            promoted_paths.update(
                {key: str(path) for key, path in destination_paths.items()}
            )
            manifest_for_promotion["paths"] = promoted_paths
            manifest_mask_stage = (
                manifest_for_promotion.setdefault("stages", {}).setdefault(
                    "whole_runner_mask",
                    {},
                )
            )
            if isinstance(manifest_mask_stage, dict):
                manifest_mask_stage.pop("deferred_browser_encoding", None)
            if stage_updates:
                source_run.update_stages(stage_updates, manifest_for_promotion)
            else:
                source_run.write_manifest(manifest_for_promotion)
        except Exception:
            for key, destination in destination_paths.items():
                backup = backups[key]
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    _atomic_copy(backup, destination)
            raise

        return _rewrite_staged_paths(staged_result, replacements)


def _finalize_render_artifact_pointers(run_dir: Path) -> None:
    """Resolve QA pointers after the parallel pose/DensePose join."""

    run = RunningClipRun(run_dir)
    if not run.manifest_path.is_file():
        return
    manifest = run.read_manifest()
    values: dict[str, Any] = {}
    for key in ("qa_overlay", "pose_qa_overlay"):
        path = run.artifact_path(key, manifest)
        if path.is_file():
            values[key] = str(path)
    if values:
        run.update_stage("renders", values, manifest)


def _run_observed_stage(
    *,
    telemetry: ProcessingTelemetry | None,
    stage: str,
    action: Callable[[Callable[[dict[str, Any]], None] | None], dict[str, Any]],
    runtime: dict[str, Any] | None = None,
    phase_spans: dict[str, str | None] | None = None,
    default_span: str | None = "inference",
    single_span: str | None = None,
) -> dict[str, Any]:
    if telemetry is None:
        result = action(None)
        result_status = str(result.get("status") or "complete").lower()
        if result_status in _FAILURE_STATUSES:
            raise RuntimeError(str(result.get("error") or f"{stage} returned {result_status}"))
        return result

    reporter = telemetry.progress_reporter(
        stage=stage,
        phase_spans=phase_spans,
        default_span=default_span,
        runtime=runtime,
    )
    with telemetry.stage(stage, runtime=runtime) as stage_boundary:
        try:
            if single_span is None:
                result = action(reporter)
            else:
                with telemetry.span(stage, single_span) as span_boundary:
                    result = action(reporter)
                    span_boundary.set_result(result)
        except Exception as exc:
            reporter.close(exc)
            raise
        result_status = str(result.get("status") or "complete").lower()
        if result_status in _FAILURE_STATUSES:
            stage_error = RuntimeError(
                str(result.get("error") or f"{stage} returned {result_status}")
            )
            reporter.close(stage_error)
            stage_boundary.set_result(result)
            raise stage_error
        reporter.close()
        stage_boundary.set_result(result)
        return result


_IDENTITY_PHASE_SPANS = {
    "loading_model": "model_load",
    "decoding": "decode",
    "detect_track": "inference",
    "tracking": "inference",
    "postprocessing": "postprocess",
    "rendering_inline_mask": "render",
    "encoding_inline_mask": "encode",
    "writing_inline_mask_outputs": "write",
    "writing_outputs": "write",
    "completed": None,
}
_MASK_PHASE_SPANS = {
    "decoding": "decode",
    "preprocessing": "preprocess",
    "loading_model": "model_load",
    "detecting": "inference",
    "identity_gated": "inference",
    "running_sam31": "inference",
    "postprocessing": "postprocess",
    "rendering": "render",
    "encoding": "encode",
    "writing_outputs": "write",
    "completed": None,
}
_POSE_PHASE_SPANS = {
    "decoding": "decode",
    "preparation": "preprocess",
    "preparing_openpose_frames": "preprocess",
    "loading_model": "model_load",
    "loading_rtmw_model": "model_load",
    "detecting_pose": "inference",
    "running_openpose": "inference",
    "running_rtmw": "inference",
    "postprocessing": "postprocess",
    "reading_outputs": "postprocess",
    "reading_openpose_results": "postprocess",
    "rendering": "render",
    "encoding": "encode",
    "writing_outputs": "write",
    "completed": None,
}
_DENSEPOSE_PHASE_SPANS = {
    "loading_model": "model_load",
    "decoding": "decode",
    "running_densepose": "inference",
    "postprocessing": "postprocess",
    "rendering": "render",
    "encoding": "encode",
    "writing_outputs": "write",
    "completed": None,
}
_FUSION_PHASE_SPANS = {
    "reading_inputs": "decode",
    "fusing_form": "postprocess",
    "rendering": "render",
    "encoding": "encode",
    "writing_outputs": "write",
    "completed": None,
}
_FEATURE_PHASE_SPANS = {
    "reading_inputs": "decode",
    "compiling_features": "postprocess",
    "summarizing_features": "postprocess",
    "writing_outputs": "write",
    "completed": None,
}


def run_full_cv_pipeline(
    *,
    run_dir: Path,
    identity_backend: str = DEFAULT_IDENTITY_BACKEND,
    pose_backend: str = "mmpose_rtmpose_l_384",
    mask_backend: str = "sam31_mlx",
    mask_quality_mode: str = "native",
    skip_densepose: bool = False,
    identity_detector_model: str | None = None,
    inline_mask_dilation_pixels: int = 5,
    inline_mask_fallback_to_track_box: bool = True,
    inline_mask_defer_browser_encoding: bool = False,
    inline_mask_temporal_reset_gap_frames: int = 3,
    inline_mask_rescue_appearance_only_identity_risk: bool = False,
    inline_mask_sam_fallback: bool = True,
    inline_mask_fallback_backend: str = "sam31_gpu",
    parallel_pose_densepose: bool = False,
    parallel_post_fusion: bool = False,
    telemetry: ProcessingTelemetry | None = None,
) -> dict[str, Any]:
    if parallel_pose_densepose and not skip_densepose and not pose_backend.startswith("mmpose_"):
        raise ValueError(
            "parallel_pose_densepose requires an mmpose backend with isolated QA output"
        )
    result: dict[str, Any] = {"run_dir": str(run_dir), "steps": []}
    inline_segmentation = _uses_inline_segmentation(mask_backend)
    selected_detector_model = identity_detector_model or (
        DEFAULT_INLINE_SEGMENTATION_MODEL if inline_segmentation else None
    )

    def run_identity_stage(
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "run_dir": run_dir,
            "backend": identity_backend,
            "progress_callback": progress,
        }
        if selected_detector_model is not None:
            kwargs["detector_model"] = selected_detector_model
        if inline_segmentation:
            kwargs.update(
                {
                    "inline_segmentation": True,
                    "inline_mask_dilation_pixels": inline_mask_dilation_pixels,
                    "inline_mask_fallback_to_track_box": inline_mask_fallback_to_track_box,
                    "inline_mask_defer_browser_encoding": (
                        inline_mask_defer_browser_encoding
                    ),
                    "inline_mask_temporal_reset_gap_frames": (
                        inline_mask_temporal_reset_gap_frames
                    ),
                    "inline_mask_rescue_appearance_only_identity_risk": (
                        inline_mask_rescue_appearance_only_identity_risk
                    ),
                }
            )
        return run_identity_tracking(**kwargs)

    identity_runtime: dict[str, Any] = {
        "backend": identity_backend,
        "inline_segmentation": inline_segmentation,
    }
    if selected_detector_model is not None:
        identity_runtime["detector_model"] = selected_detector_model
    if inline_segmentation:
        identity_runtime["inline_mask_temporal_reset_gap_frames"] = (
            inline_mask_temporal_reset_gap_frames
        )
        identity_runtime["inline_mask_rescue_appearance_only_identity_risk"] = (
            inline_mask_rescue_appearance_only_identity_risk
        )
    identity = _run_observed_stage(
        telemetry=telemetry,
        stage="target_tracking",
        runtime=identity_runtime,
        phase_spans=_IDENTITY_PHASE_SPANS,
        action=run_identity_stage,
    )
    result["steps"].append({"stage": "identity", "result": identity})

    mask_runtime: dict[str, Any] = {
        "backend": mask_backend,
        "quality_mode": mask_quality_mode,
    }
    mask_phase_spans: dict[str, str | None] | None = _MASK_PHASE_SPANS
    mask_default_span: str | None = "inference"
    if inline_segmentation:
        inline_mask = identity.get("inline_mask")
        if not isinstance(inline_mask, dict) or inline_mask.get("status") != "complete":
            raise RuntimeError(
                "YOLO26 inline segmentation did not produce the runner-mask artifact contract"
            )
        inline_summary = inline_mask.get("summary")
        fallback_recommended = bool(
            isinstance(inline_summary, dict)
            and inline_summary.get("sam_fallback_recommended")
        )
        if inline_mask_sam_fallback and fallback_recommended:
            mask_runtime.update(
                {
                    "backend": inline_mask_fallback_backend,
                    "fallback_from": mask_backend,
                    "fallback_reason": "inline_mask_quality_gate",
                }
            )

            def run_mask_stage(
                progress: Callable[[dict[str, Any]], None] | None,
            ) -> dict[str, Any]:
                fallback_result = _run_staged_mask_fallback(
                    run_dir=run_dir,
                    mask_backend=inline_mask_fallback_backend,
                    mask_quality_mode=mask_quality_mode,
                    progress_callback=progress,
                )
                return {**fallback_result, "fallback_from": mask_backend}

        else:
            mask_phase_spans = None
            mask_default_span = None
            reused_inline_mask = {
                **inline_mask,
                "produced_during_stage": "target_tracking",
            }

            def run_mask_stage(
                _: Callable[[dict[str, Any]], None] | None,
            ) -> dict[str, Any]:
                return reused_inline_mask
    else:
        def run_mask_stage(
            progress: Callable[[dict[str, Any]], None] | None,
        ) -> dict[str, Any]:
            return _run_mask_stage(
                run_dir=run_dir,
                mask_backend=mask_backend,
                mask_quality_mode=mask_quality_mode,
                progress_callback=progress,
            )

    mask = _run_observed_stage(
        telemetry=telemetry,
        stage="runner_mask",
        runtime=mask_runtime,
        phase_spans=mask_phase_spans,
        default_span=mask_default_span,
        action=run_mask_stage,
    )
    if inline_segmentation and mask.get("fallback_from"):
        inline_identity_mask = identity.get("inline_mask")
        if isinstance(inline_identity_mask, dict):
            inline_identity_mask.pop("deferred_browser_encoding", None)
            inline_identity_mask["superseded_by"] = mask.get("backend")
    result["steps"].append({"stage": "mask", "result": mask})

    if pose_backend.startswith("mmpose_"):
        from whodoirunlike.mmpose_runner import run_mmpose_pose

        def pose_action(
            progress: Callable[[dict[str, Any]], None] | None,
        ) -> dict[str, Any]:
            return run_mmpose_pose(
                run_dir=run_dir,
                model_id=pose_backend,
                isolate_qa_overlay=True,
                progress_callback=progress,
            )
    elif pose_backend == "mediapipe":
        from whodoirunlike.pose_runner import run_pose_landmarks

        def pose_action(
            progress: Callable[[dict[str, Any]], None] | None,
        ) -> dict[str, Any]:
            return run_pose_landmarks(
                run_dir=run_dir,
                model_variant="heavy",
                input_mode="auto",
                progress_callback=progress,
            )
    else:
        from whodoirunlike.openpose_runner import run_openpose_pose

        def pose_action(
            progress: Callable[[dict[str, Any]], None] | None,
        ) -> dict[str, Any]:
            return run_openpose_pose(
                run_dir=run_dir,
                progress_callback=progress,
            )

    pose_runtime: dict[str, Any] = {"backend": pose_backend}
    if pose_backend.startswith("mmpose_"):
        pose_runtime.update(
            {
                "device": _env_value("MMPOSE_DEVICE") or "cpu",
                "runtime_backend": _env_value("RTMW_RUNTIME_BACKEND") or "onnxruntime",
                "use_detector": _env_bool_value("MMPOSE_USE_DETECTOR", True),
            }
        )

    def run_pose_stage() -> dict[str, Any]:
        return _run_observed_stage(
            telemetry=telemetry,
            stage="pose_sequence",
            runtime=pose_runtime,
            phase_spans=_POSE_PHASE_SPANS,
            action=pose_action,
        )

    if not skip_densepose:
        from whodoirunlike.densepose_runner import run_densepose

        densepose_kwargs = _densepose_runtime_kwargs()

        def run_densepose_stage() -> dict[str, Any]:
            return _run_observed_stage(
                telemetry=telemetry,
                stage="densepose_body_map",
                runtime={
                    "backend": "densepose",
                    "device": densepose_kwargs["device"],
                    "input_min_size_test": densepose_kwargs["input_min_size_test"],
                    "input_max_size_test": densepose_kwargs["input_max_size_test"],
                    "target_crop_enabled": densepose_kwargs["target_crop_enabled"],
                    "target_crop_padding_ratio": densepose_kwargs[
                        "target_crop_padding_ratio"
                    ],
                    "target_crop_padding_pixels": densepose_kwargs[
                        "target_crop_padding_pixels"
                    ],
                },
                phase_spans=_DENSEPOSE_PHASE_SPANS,
                action=lambda progress: run_densepose(
                    run_dir=run_dir,
                    progress_callback=progress,
                    **densepose_kwargs,
                ),
            )
    else:
        def write_skipped_densepose(
            _: Callable[[dict[str, Any]], None] | None,
        ) -> dict[str, Any]:
            run = RunningClipRun(run_dir)
            manifest = run.read_manifest()
            densepose_path = run.artifact_path("densepose", manifest)
            if not densepose_path.exists():
                densepose_path.parent.mkdir(parents=True, exist_ok=True)
                densepose_path.write_text("", encoding="utf-8")
            if pose_backend.startswith("mmpose_"):
                pose_qa_path = run.artifact_path("pose_qa_overlay", manifest)
                canonical_qa_path = run.artifact_path("qa_overlay", manifest)
                if pose_qa_path.exists():
                    canonical_qa_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(pose_qa_path, canonical_qa_path)
                    make_browser_playable_mp4(canonical_qa_path)
            densepose_stage = manifest.setdefault("stages", {}).setdefault("densepose", {})
            densepose_stage.pop("error", None)
            densepose_stage.pop("setup_instructions", None)
            manifest["updated_at"] = utc_now_iso()
            run.update_stage(
                "densepose",
                {"status": "skipped", "output": str(densepose_path)},
                manifest,
            )
            return {"status": "skipped"}

        def run_densepose_stage() -> dict[str, Any]:
            return _run_observed_stage(
                telemetry=telemetry,
                stage="densepose_body_map",
                runtime={"backend": "densepose", "skipped": True},
                action=write_skipped_densepose,
                default_span=None,
                single_span="write",
            )

    if parallel_pose_densepose and not skip_densepose:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-analysis") as executor:
            pose_future = executor.submit(run_pose_stage)
            densepose_future = executor.submit(run_densepose_stage)
            pose = pose_future.result()
            densepose = densepose_future.result()
    else:
        pose = run_pose_stage()
        densepose = run_densepose_stage()

    result["steps"].append({"stage": "pose", "result": pose})
    result["steps"].append({"stage": "densepose", "result": densepose})
    _finalize_render_artifact_pointers(run_dir)

    fusion = _run_observed_stage(
        telemetry=telemetry,
        stage="fused_form_signal",
        phase_spans=_FUSION_PHASE_SPANS,
        action=lambda progress: run_fused_form(
            run_dir=run_dir,
            progress_callback=progress,
        ),
    )
    result["steps"].append({"stage": "fusion", "result": fusion})
    def run_features_stage() -> dict[str, Any]:
        return _run_observed_stage(
            telemetry=telemetry,
            stage="form_feature_compilation",
            phase_spans=_FEATURE_PHASE_SPANS,
            action=lambda progress: compile_form_features(
                run_dir=run_dir,
                progress_callback=progress,
            ),
        )

    def run_tables_stage() -> dict[str, Any]:
        return _run_observed_stage(
            telemetry=telemetry,
            stage="artifact_table_export",
            action=lambda _: export_cv_tables(run_dir),
            default_span=None,
            single_span="write",
        )

    def run_qc_stage() -> dict[str, Any]:
        return _run_observed_stage(
            telemetry=telemetry,
            stage="quality_control",
            action=lambda _: run_qc_metrics(run_dir),
            default_span=None,
            single_span="postprocess",
        )

    if parallel_post_fusion:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="cv-postprocess") as executor:
            features_future = executor.submit(run_features_stage)
            tables_future = executor.submit(run_tables_stage)
            qc_future = executor.submit(run_qc_stage)
            features = features_future.result()
            tables = tables_future.result()
            qc = qc_future.result()
    else:
        features = run_features_stage()
        tables = run_tables_stage()
        qc = run_qc_stage()

    result["steps"].append({"stage": "features", "result": features})
    result["steps"].append({"stage": "artifact_tables", "result": tables})
    result["steps"].append({"stage": "qc", "result": qc})
    return result
