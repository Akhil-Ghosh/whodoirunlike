from __future__ import annotations

from pathlib import Path
from typing import Any

from whodoirunlike.artifact_tables import export_cv_tables
from whodoirunlike.form_features import compile_form_features
from whodoirunlike.fusion_runner import run_fused_form
from whodoirunlike.identity_runner import run_identity_tracking
from whodoirunlike.qc import run_qc_metrics
from whodoirunlike.sam31_mlx_runner import run_sam31_mlx_mask


def run_full_cv_pipeline(
    *,
    run_dir: Path,
    pose_backend: str = "mmpose_rtmpose_l_384",
    mask_quality_mode: str = "native",
    skip_densepose: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_dir": str(run_dir), "steps": []}
    identity = run_identity_tracking(run_dir=run_dir)
    result["steps"].append({"stage": "identity", "result": identity})

    mask = run_sam31_mlx_mask(run_dir=run_dir, quality_mode=mask_quality_mode)
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

        densepose = run_densepose(run_dir=run_dir)
        result["steps"].append({"stage": "densepose", "result": densepose})
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
