from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.qc import run_qc_metrics
from whodoirunlike.running_clip_run import RunningClipRun


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _reason_tags(qc: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    identity = qc.get("identity", {})
    mask = qc.get("mask", {})
    pose = qc.get("pose", {})
    fused = qc.get("fused", {})
    if float(identity.get("identity_risk_rate") or 0.0) >= 0.05:
        tags.append("identity_risk")
    if float(identity.get("missing_rate") or 0.0) >= 0.05:
        tags.append("target_missing")
    if float(mask.get("mean_mask_churn") or 0.0) >= 0.25:
        tags.append("mask_churn")
    if float(pose.get("usable_rate") or 1.0) < 0.8:
        tags.append("pose_drop")
    if float(fused.get("mean_frame_confidence") or 1.0) < 0.65:
        tags.append("low_fused_confidence")
    return tags or ["review_sample"]


def build_uncertainty_queue(cv_run_root: Path, output_path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(cv_run_root.glob("*/cv_run_manifest.json")):
        run_dir = manifest_path.parent
        run = RunningClipRun(run_dir)
        manifest = run.read_manifest()
        qc_path = run.artifact_path("qc_metrics", manifest)
        try:
            qc = run_qc_metrics(run_dir)
        except Exception:
            qc = _read_json(qc_path)
        score = float(qc.get("uncertainty_score") or 0.0)
        entries.append(
            {
                "candidate_id": manifest.get("candidate_id") or run_dir.name,
                "runner_name": manifest.get("runner_name"),
                "run_dir": str(run_dir),
                "uncertainty_score": round(score, 6),
                "reason_tags": _reason_tags(qc),
                "review_url_hint": f"/subject.html?candidate_id={run_dir.name}",
                "qc_metrics": str(qc_path),
            }
        )
    entries.sort(key=lambda row: (-float(row["uncertainty_score"]), str(row["candidate_id"])))
    payload = {
        "version": 1,
        "created_at": utc_now_iso(),
        "queue_type": "active_learning_uncertainty",
        "entry_count": len(entries),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
