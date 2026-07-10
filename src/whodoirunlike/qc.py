from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from whodoirunlike.artifact_tables import read_jsonl
from whodoirunlike.cv_flow import utc_now_iso, write_json
from whodoirunlike.mask_artifacts import mask_rows_from_video, write_masks_jsonl_from_video
from whodoirunlike.running_clip_run import RunningClipRun


def _mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".parquet" and path.exists():
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("QC Parquet reads need pyarrow") from exc
        return pq.read_table(path).to_pylist()
    return []


def identity_metrics(tracklets_path: Path) -> dict[str, Any]:
    rows = _load_rows(tracklets_path)
    target_rows = [row for row in rows if row.get("is_target", True)]
    rows = target_rows or rows
    total = len(rows)
    if not rows:
        return {"frame_count": 0, "target_identity_stability_rate": 0.0}
    states = Counter(str(row.get("identity_state") or "missing") for row in rows)
    similarities = [
        float(row["reid_similarity"])
        for row in rows
        if row.get("reid_similarity") not in (None, "")
    ]
    return {
        "frame_count": total,
        "usable_frames": states.get("usable", 0),
        "identity_risk_frames": states.get("identity_risk", 0),
        "missing_frames": states.get("missing", 0),
        "target_identity_stability_rate": round(states.get("usable", 0) / total, 6),
        "identity_risk_rate": round(states.get("identity_risk", 0) / total, 6),
        "missing_rate": round(states.get("missing", 0) / total, 6),
        "mean_reid_similarity": _mean(similarities),
        "min_reid_similarity": round(min(similarities), 6) if similarities else 0.0,
    }


def mask_metrics(mask_video_path: Path, masks_jsonl_path: Path | None = None) -> dict[str, Any]:
    if not mask_video_path.exists():
        return {"frame_count": 0, "mask_available": False}
    if masks_jsonl_path:
        summary = write_masks_jsonl_from_video(mask_video_path, masks_jsonl_path)
    else:
        meta, rows = mask_rows_from_video(mask_video_path)
        temporal = [
            float(row["temporal_iou_prev"])
            for row in rows
            if row.get("temporal_iou_prev") is not None
        ]
        summary = {
            **meta,
            "mean_temporal_iou": _mean(temporal),
            "mean_mask_churn": round(1.0 - mean(temporal), 6) if temporal else None,
            "nonempty_frames": sum(1 for row in rows if int(row["area"]) > 0),
        }
    return {"mask_available": True, **summary}


def pose_metrics(pose_path: Path) -> dict[str, Any]:
    rows = read_jsonl(pose_path)
    if not rows:
        return {"frame_count": 0, "pose_available": False}
    visibility = [
        float(row.get("visibility_mean") or row.get("pose_confidence_mean") or 0.0)
        for row in rows
    ]
    usable = sum(1 for row in rows if bool(row.get("usable", row.get("detected", False))))
    key_missing = sum(1 for row in rows if row.get("drop_reason") or row.get("rejection_reason"))
    return {
        "pose_available": True,
        "frame_count": len(rows),
        "usable_frames": usable,
        "usable_rate": round(usable / len(rows), 6),
        "visibility_mean": _mean(visibility),
        "dropped_or_rejected_frames": key_missing,
    }


def fused_metrics(fused_path: Path) -> dict[str, Any]:
    rows = read_jsonl(fused_path)
    if not rows:
        return {"frame_count": 0, "fused_available": False}
    confidence = [float(row.get("frame_confidence") or row.get("confidence") or 0.0) for row in rows]
    states = Counter(str(row.get("frame_state") or row.get("state") or "unknown") for row in rows)
    return {
        "fused_available": True,
        "frame_count": len(rows),
        "mean_frame_confidence": _mean(confidence),
        "min_frame_confidence": round(min(confidence), 6) if confidence else 0.0,
        "frame_states": dict(states),
    }


def overall_uncertainty_score(metrics: dict[str, Any]) -> float:
    identity = metrics.get("identity", {})
    mask = metrics.get("mask", {})
    pose = metrics.get("pose", {})
    fused = metrics.get("fused", {})
    score = (
        0.35 * float(identity.get("identity_risk_rate") or 0.0)
        + 0.20 * float(identity.get("missing_rate") or 0.0)
        + 0.20 * float(mask.get("mean_mask_churn") or 0.0)
        + 0.15 * (1.0 - float(pose.get("usable_rate") or 0.0))
        + 0.10 * (1.0 - float(fused.get("mean_frame_confidence") or 0.0))
    )
    return round(max(0.0, min(1.0, score)), 6)


def run_qc_metrics(run_dir: Path) -> dict[str, Any]:
    run = RunningClipRun(run_dir)
    manifest = run.read_manifest()
    tracklets = run.artifact_path("tracklets_jsonl", manifest)
    if not tracklets.exists():
        tracklets = run.artifact_path("tracklets", manifest)
    masks_jsonl = run.artifact_path("masks_jsonl", manifest)
    runner_mask = run.artifact_path("runner_mask", manifest)
    pose_path = run.artifact_path("pose_landmarks", manifest)
    fused_path = run.artifact_path("fused_form", manifest)
    payload = {
        "version": 1,
        "candidate_id": manifest.get("candidate_id"),
        "updated_at": utc_now_iso(),
        "identity": identity_metrics(tracklets),
        "mask": mask_metrics(runner_mask, masks_jsonl),
        "pose": pose_metrics(pose_path),
        "fused": fused_metrics(fused_path),
    }
    payload["uncertainty_score"] = overall_uncertainty_score(payload)

    qc_path = run.artifact_path("qc_metrics", manifest)
    write_json(qc_path, payload)

    manifest["updated_at"] = utc_now_iso()
    run.update_stage(
        "qc_metrics",
        {
            "status": "complete",
            "output": str(qc_path),
            "uncertainty_score": payload["uncertainty_score"],
            "completed_at": utc_now_iso(),
        },
        manifest,
    )
    return payload
