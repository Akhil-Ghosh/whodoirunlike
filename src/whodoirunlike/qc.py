from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from whodoirunlike.artifact_tables import read_jsonl
from whodoirunlike.cv_flow import utc_now_iso, write_json
from whodoirunlike.mask_artifacts import mask_rows_from_video, write_masks_jsonl_from_video
from whodoirunlike.running_clip_run import RunningClipRun


_MASK_SUMMARY_FIELDS = (
    "fps",
    "width",
    "height",
    "frame_count",
    "output_path",
    "mean_temporal_iou",
    "mean_mask_churn",
    "nonempty_frames",
)
_SAM31_MASK_BACKENDS = frozenset({"sam31_gpu", "sam31_mlx"})


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


def mask_metrics(
    mask_video_path: Path,
    masks_jsonl_path: Path | None = None,
    *,
    mask_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not mask_video_path.exists():
        return {"frame_count": 0, "mask_available": False}
    if mask_summary is not None:
        summary = dict(mask_summary)
    elif masks_jsonl_path:
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


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mask_summary_matches_jsonl(summary: Mapping[str, Any], masks_jsonl: Path) -> bool:
    fps = summary.get("fps")
    width = summary.get("width")
    height = summary.get("height")
    frame_count = summary.get("frame_count")
    nonempty_frames = summary.get("nonempty_frames")
    if not _is_finite_number(fps) or float(fps) <= 0:
        return False
    if not all(_is_plain_int(value) and value >= 0 for value in (width, height, frame_count)):
        return False
    if not _is_plain_int(nonempty_frames) or not 0 <= nonempty_frames <= frame_count:
        return False

    try:
        rows = read_jsonl(masks_jsonl)
    except (OSError, TypeError, UnicodeError, ValueError):
        return False
    if len(rows) != frame_count:
        return False

    temporal_ious: list[float] = []
    observed_nonempty_frames = 0
    pixel_count = width * height
    for expected_frame_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            return False
        if row.get("frame_index") != expected_frame_index:
            return False
        if row.get("width") != width or row.get("height") != height:
            return False
        area = row.get("area")
        if not _is_plain_int(area) or not 0 <= area <= pixel_count:
            return False
        observed_nonempty_frames += int(area > 0)

        rle = row.get("rle")
        if not isinstance(rle, Mapping) or rle.get("size") != [height, width]:
            return False
        counts = rle.get("counts")
        if not isinstance(counts, list) or not all(
            _is_plain_int(count) and count >= 0 for count in counts
        ):
            return False
        if sum(counts) != pixel_count or sum(counts[1::2]) != area:
            return False

        temporal_iou = row.get("temporal_iou_prev")
        if expected_frame_index == 0:
            if temporal_iou is not None:
                return False
        else:
            if not _is_finite_number(temporal_iou) or not 0 <= float(temporal_iou) <= 1:
                return False
            temporal_ious.append(float(temporal_iou))

    if observed_nonempty_frames != nonempty_frames:
        return False
    expected_temporal_iou = _mean(temporal_ious) if temporal_ious else None
    expected_mask_churn = (
        round(1.0 - mean(temporal_ious), 6) if temporal_ious else None
    )
    return (
        summary.get("mean_temporal_iou") == expected_temporal_iou
        and summary.get("mean_mask_churn") == expected_mask_churn
    )


def _current_sam31_mask_summary(
    manifest: Mapping[str, Any],
    *,
    runner_mask: Path,
    masks_jsonl: Path,
) -> dict[str, Any] | None:
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        return None
    stage = stages.get("whole_runner_mask")
    if not isinstance(stage, Mapping):
        return None
    if stage.get("status") != "complete" or stage.get("backend") not in _SAM31_MASK_BACKENDS:
        return None
    if not runner_mask.is_file() or not masks_jsonl.is_file():
        return None
    try:
        if masks_jsonl.stat().st_mtime_ns < runner_mask.stat().st_mtime_ns:
            return None
    except OSError:
        return None

    stage_masks_jsonl = stage.get("masks_jsonl")
    summary = stage.get("mask_summary")
    if not isinstance(stage_masks_jsonl, str) or not isinstance(summary, Mapping):
        return None
    if Path(stage_masks_jsonl).resolve(strict=False) != masks_jsonl.resolve(strict=False):
        return None
    if any(field not in summary for field in _MASK_SUMMARY_FIELDS):
        return None
    if Path(str(summary["output_path"])).resolve(strict=False) != masks_jsonl.resolve(
        strict=False
    ):
        return None
    if not _mask_summary_matches_jsonl(summary, masks_jsonl):
        return None
    return {field: summary[field] for field in _MASK_SUMMARY_FIELDS}


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
    mask_summary = _current_sam31_mask_summary(
        manifest,
        runner_mask=runner_mask,
        masks_jsonl=masks_jsonl,
    )
    pose_path = run.artifact_path("pose_landmarks", manifest)
    fused_path = run.artifact_path("fused_form", manifest)
    payload = {
        "version": 1,
        "candidate_id": manifest.get("candidate_id"),
        "updated_at": utc_now_iso(),
        "identity": identity_metrics(tracklets),
        "mask": mask_metrics(runner_mask, masks_jsonl, mask_summary=mask_summary),
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
