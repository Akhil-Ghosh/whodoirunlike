from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from whodoirunlike.artifact_tables import read_jsonl
from whodoirunlike.cv_flow import read_json, utc_now_iso
from whodoirunlike.running_clip_run import RunningClipRun


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def representative_reid_embedding(run_dir: Path) -> list[float] | None:
    run = RunningClipRun(run_dir)
    manifest = run.read_manifest()
    reid_jsonl = run.artifact_path("reid_jsonl", manifest)
    rows = read_jsonl(reid_jsonl)
    embeddings = [row.get("embedding") for row in rows if isinstance(row.get("embedding"), list)]
    if not embeddings:
        return None
    return np.asarray(embeddings, dtype=np.float32).mean(axis=0).tolist()


def view_bucket(run_dir: Path) -> str:
    run = RunningClipRun(run_dir)
    manifest = run.read_manifest()
    bucket_path = run.artifact_path("view_bucket", manifest)
    if bucket_path.exists():
        payload = read_json(bucket_path)
        return str(payload.get("view_bucket") or "unknown")
    return str(manifest.get("review", {}).get("camera_angle") or "unknown")


def cross_view_cost(
    run_a: Path,
    run_b: Path,
    *,
    synchronized: bool = False,
    temporal_offset_seconds: float | None = None,
    reprojection_error_px: float | None = None,
) -> dict[str, Any]:
    emb_a = representative_reid_embedding(run_a)
    emb_b = representative_reid_embedding(run_b)
    appearance_cost = 1.0 if emb_a is None or emb_b is None else 1.0 - cosine_similarity(emb_a, emb_b)
    view_cost = 0.0 if view_bucket(run_a) == view_bucket(run_b) else 0.15
    if synchronized:
        time_cost = min(abs(float(temporal_offset_seconds or 0.0)) / 0.25, 1.0)
        geom_cost = min(float(reprojection_error_px or 0.0) / 25.0, 1.0)
        total = 0.40 * appearance_cost + 0.35 * geom_cost + 0.10 * time_cost + 0.15 * view_cost
    else:
        time_cost = None
        geom_cost = None
        total = 0.70 * appearance_cost + 0.30 * view_cost
    return {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "synchronized": synchronized,
        "appearance_cost": round(float(appearance_cost), 6),
        "view_cost": round(float(view_cost), 6),
        "temporal_offset_cost": round(float(time_cost), 6) if time_cost is not None else None,
        "geometry_cost": round(float(geom_cost), 6) if geom_cost is not None else None,
        "total_cost": round(max(0.0, min(1.0, float(total))), 6),
    }


def write_cross_view_match(
    run_a: Path,
    run_b: Path,
    output_path: Path,
    *,
    synchronized: bool = False,
    temporal_offset_seconds: float | None = None,
    reprojection_error_px: float | None = None,
) -> dict[str, Any]:
    match = cross_view_cost(
        run_a,
        run_b,
        synchronized=synchronized,
        temporal_offset_seconds=temporal_offset_seconds,
        reprojection_error_px=reprojection_error_px,
    )
    payload = {"version": 1, "created_at": utc_now_iso(), "match": match}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
