from __future__ import annotations

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DENSEPOSE_CONFIG_ENV = "DENSEPOSE_CONFIG"
DENSEPOSE_WEIGHTS_ENV = "DENSEPOSE_WEIGHTS"
DENSEPOSE_DEVICE_ENV = "DENSEPOSE_DEVICE"
DENSEPOSE_DEFAULT_CONFIG = (
    REPO_ROOT / "models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
)
DENSEPOSE_DEFAULT_WEIGHTS = (
    REPO_ROOT / "models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl"
)


def _env_value(name: str) -> str:
    import os

    return os.getenv(name, "").strip()


def _resolve_repo_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (REPO_ROOT / raw_path).resolve()


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


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
    telemetry: ProcessingTelemetry | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_dir": str(run_dir), "steps": []}
    identity = _run_observed_stage(
        telemetry=telemetry,
        stage="target_tracking",
        runtime={"backend": identity_backend},
        phase_spans=_IDENTITY_PHASE_SPANS,
        action=lambda progress: run_identity_tracking(
            run_dir=run_dir,
            backend=identity_backend,
            progress_callback=progress,
        ),
    )
    result["steps"].append({"stage": "identity", "result": identity})

    mask = _run_observed_stage(
        telemetry=telemetry,
        stage="runner_mask",
        runtime={"backend": mask_backend, "quality_mode": mask_quality_mode},
        phase_spans=_MASK_PHASE_SPANS,
        action=lambda progress: _run_mask_stage(
            run_dir=run_dir,
            mask_backend=mask_backend,
            mask_quality_mode=mask_quality_mode,
            progress_callback=progress,
        ),
    )
    result["steps"].append({"stage": "mask", "result": mask})

    if pose_backend.startswith("mmpose_"):
        from whodoirunlike.mmpose_runner import run_mmpose_pose

        def pose_action(
            progress: Callable[[dict[str, Any]], None] | None,
        ) -> dict[str, Any]:
            return run_mmpose_pose(
                run_dir=run_dir,
                model_id=pose_backend,
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
    pose = _run_observed_stage(
        telemetry=telemetry,
        stage="pose_sequence",
        runtime={"backend": pose_backend},
        phase_spans=_POSE_PHASE_SPANS,
        action=pose_action,
    )
    result["steps"].append({"stage": "pose", "result": pose})

    if not skip_densepose:
        from whodoirunlike.densepose_runner import run_densepose

        densepose_kwargs = _densepose_runtime_kwargs()
        densepose = _run_observed_stage(
            telemetry=telemetry,
            stage="densepose_body_map",
            runtime={"backend": "densepose", "device": densepose_kwargs["device"]},
            phase_spans=_DENSEPOSE_PHASE_SPANS,
            action=lambda progress: run_densepose(
                run_dir=run_dir,
                progress_callback=progress,
                **densepose_kwargs,
            ),
        )
        result["steps"].append({"stage": "densepose", "result": densepose})
    else:
        def skip_densepose(_: Callable[[dict[str, Any]], None] | None) -> dict[str, Any]:
            run = RunningClipRun(run_dir)
            manifest = run.read_manifest()
            densepose_path = run.artifact_path("densepose", manifest)
            if not densepose_path.exists():
                densepose_path.parent.mkdir(parents=True, exist_ok=True)
                densepose_path.write_text("", encoding="utf-8")
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

        densepose = _run_observed_stage(
            telemetry=telemetry,
            stage="densepose_body_map",
            runtime={"backend": "densepose", "skipped": True},
            action=skip_densepose,
            default_span=None,
            single_span="write",
        )
        result["steps"].append({"stage": "densepose", "result": densepose})

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
    features = _run_observed_stage(
        telemetry=telemetry,
        stage="form_feature_compilation",
        phase_spans=_FEATURE_PHASE_SPANS,
        action=lambda progress: compile_form_features(
            run_dir=run_dir,
            progress_callback=progress,
        ),
    )
    result["steps"].append({"stage": "features", "result": features})
    tables = _run_observed_stage(
        telemetry=telemetry,
        stage="artifact_table_export",
        action=lambda _: export_cv_tables(run_dir),
        default_span=None,
        single_span="write",
    )
    result["steps"].append({"stage": "artifact_tables", "result": tables})
    qc = _run_observed_stage(
        telemetry=telemetry,
        stage="quality_control",
        action=lambda _: run_qc_metrics(run_dir),
        default_span=None,
        single_span="postprocess",
    )
    result["steps"].append({"stage": "qc", "result": qc})
    return result
