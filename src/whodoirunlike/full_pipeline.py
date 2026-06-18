from __future__ import annotations

from pathlib import Path
from typing import Any

from whodoirunlike.artifact_tables import export_cv_tables
from whodoirunlike.form_features import compile_form_features
from whodoirunlike.fusion_runner import run_fused_form
from whodoirunlike.identity_runner import DEFAULT_IDENTITY_BACKEND, run_identity_tracking
from whodoirunlike.qc import run_qc_metrics
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
) -> dict[str, Any]:
    normalized = mask_backend.strip().lower()
    if normalized in {"sam31", "sam31_mlx", "sam3.1_mlx", "mlx"}:
        return run_sam31_mlx_mask(run_dir=run_dir, quality_mode=mask_quality_mode)
    if normalized in {"sam31_gpu", "sam3.1_gpu", "sam31_cuda", "sam3.1_cuda"}:
        from whodoirunlike.sam31_gpu_runner import run_sam31_gpu_mask

        return run_sam31_gpu_mask(run_dir=run_dir)
    raise ValueError("mask_backend must be one of: sam31_mlx, sam31_gpu")


def run_full_cv_pipeline(
    *,
    run_dir: Path,
    identity_backend: str = DEFAULT_IDENTITY_BACKEND,
    pose_backend: str = "mmpose_rtmpose_l_384",
    mask_backend: str = "sam31_mlx",
    mask_quality_mode: str = "native",
    skip_densepose: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_dir": str(run_dir), "steps": []}
    identity = run_identity_tracking(run_dir=run_dir, backend=identity_backend)
    result["steps"].append({"stage": "identity", "result": identity})

    mask = _run_mask_stage(
        run_dir=run_dir,
        mask_backend=mask_backend,
        mask_quality_mode=mask_quality_mode,
    )
    result["steps"].append({"stage": "mask", "result": mask})

    if pose_backend.startswith("mmpose_"):
        from whodoirunlike.mmpose_runner import run_mmpose_pose

        pose = run_mmpose_pose(run_dir=run_dir, model_id=pose_backend)
    elif pose_backend == "mediapipe":
        from whodoirunlike.pose_runner import run_pose_landmarks

        pose = run_pose_landmarks(run_dir=run_dir, model_variant="heavy", input_mode="auto")
    else:
        from whodoirunlike.openpose_runner import run_openpose_pose

        pose = run_openpose_pose(run_dir=run_dir)
    result["steps"].append({"stage": "pose", "result": pose})

    if not skip_densepose:
        from whodoirunlike.densepose_runner import run_densepose

        densepose = run_densepose(run_dir=run_dir, **_densepose_runtime_kwargs())
        result["steps"].append({"stage": "densepose", "result": densepose})
        if densepose.get("status") == "failed":
            raise RuntimeError(str(densepose.get("error") or "DensePose failed"))
    else:
        densepose_path = run_dir / "densepose.jsonl"
        if not densepose_path.exists():
            densepose_path.write_text("", encoding="utf-8")
        result["steps"].append({"stage": "densepose", "result": {"status": "skipped"}})

    fusion = run_fused_form(run_dir=run_dir)
    result["steps"].append({"stage": "fusion", "result": fusion})
    features = compile_form_features(run_dir=run_dir)
    result["steps"].append({"stage": "features", "result": features})
    tables = export_cv_tables(run_dir)
    result["steps"].append({"stage": "artifact_tables", "result": tables})
    qc = run_qc_metrics(run_dir)
    result["steps"].append({"stage": "qc", "result": qc})
    return result
