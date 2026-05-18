from __future__ import annotations

import csv
import json
import mimetypes
import re
import socket
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
BEST_QUALITY_REVIEW_SOURCE = REPO_ROOT / "artifacts/evaluation/video_candidates.review20_best.json"
FULL_HD_REVIEW_SOURCE = REPO_ROOT / "artifacts/evaluation/video_candidates.review20_1080.json"
HIGH_QUALITY_REVIEW_SOURCE = REPO_ROOT / "artifacts/evaluation/video_candidates.review20_720.json"
TRIAGE_REVIEW_SOURCE = REPO_ROOT / "artifacts/evaluation/video_candidates.top30.json"
REVIEW_SOURCE_PRIORITY = [
    BEST_QUALITY_REVIEW_SOURCE,
    FULL_HD_REVIEW_SOURCE,
    HIGH_QUALITY_REVIEW_SOURCE,
    TRIAGE_REVIEW_SOURCE,
]
DEFAULT_SOURCE = next(
    (source_path for source_path in REVIEW_SOURCE_PRIORITY if source_path.exists()),
    TRIAGE_REVIEW_SOURCE,
)
DEFAULT_ANNOTATIONS = REPO_ROOT / "artifacts/review/clip_reviews.json"
DEFAULT_STATIC_DIR = REPO_ROOT / "review_ui"
QUALITY_VALUES = {"", "good", "mid", "bad"}
CAMERA_ANGLE_VALUES = {"", "side", "diagonal", "front", "rear", "mixed", "unknown"}
PROMPT_SELECTION_TYPES = {"unset", "point", "box", "mask"}
MAX_PROMPT_POINTS = 128
CV_ARTIFACT_LABELS = {
    "source_segment": "Source segment",
    "prompt_frame": "Prompt frame",
    "person_prompt": "Person prompt",
    "pose_landmarks": "Pose landmarks",
    "runner_mask": "Runner mask",
    "densepose": "DensePose",
    "skeleton_render": "Skeleton render",
    "masked_runner": "Masked runner",
    "qa_overlay": "QA overlay",
    "features": "Features",
}
SAM2_CHECKPOINT = Path("models/sam2/sam2.1_hiera_tiny.pt")
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MASK_BACKENDS = {
    "sam2": "SAM 2.1 local",
    "sam31_mlx": "SAM 3.1 MLX",
}
MASK_QUALITY_MODES = {
    "max": "Highest quality",
    "native": "1008",
    "fast": "224",
}
MASK_JOBS: dict[str, dict[str, Any]] = {}
MASK_JOBS_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReviewAppConfig:
    source_path: Path = DEFAULT_SOURCE
    annotations_path: Path = DEFAULT_ANNOTATIONS
    static_dir: Path = DEFAULT_STATIC_DIR
    repo_root: Path = REPO_ROOT
    limit: int = 20


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_source_rows(source_path: Path) -> list[dict[str, Any]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Review source not found: {source_path}")

    if source_path.suffix.lower() == ".json":
        rows = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {source_path}")
        return [dict(row) for row in rows]

    with source_path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    return int(_as_float(value, float(default)))


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (repo_root / raw_path).resolve()


def _candidate_priority(row: dict[str, Any], source_index: int) -> tuple[float, float, int, int]:
    return (
        -_as_float(row.get("cv_score")),
        -_as_float(row.get("score")),
        -_as_int(row.get("view_count")),
        source_index,
    )


def load_review_clips(config: ReviewAppConfig) -> list[dict[str, Any]]:
    rows = _read_source_rows(config.source_path)
    enriched: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id") or "").strip()
        video_path_value = str(row.get("video_path") or "").strip()
        if not candidate_id or not video_path_value:
            continue

        local_video_path = _resolve_repo_path(video_path_value, config.repo_root)
        if not local_video_path.exists():
            continue

        duration = _as_float(row.get("duration_seconds_local") or row.get("duration_seconds"))
        clip = {
            **row,
            "source_index": source_index,
            "candidate_id": candidate_id,
            "runner_name": str(row.get("runner_name") or "Unknown runner"),
            "primary_bucket": str(row.get("primary_bucket") or "running"),
            "title": str(row.get("title") or "Untitled clip"),
            "channel": str(row.get("channel") or "Unknown channel"),
            "url": str(row.get("url") or ""),
            "video_path": str(local_video_path),
            "duration_seconds_local": round(duration, 2),
            "cv_score": _as_int(row.get("cv_score")),
            "score": _as_int(row.get("score")),
            "view_count": _as_int(row.get("view_count")),
            "pose_hit_rate": _as_float(row.get("pose_hit_rate")),
            "full_body_rate": _as_float(row.get("full_body_rate")),
            "size_ok_rate": _as_float(row.get("size_ok_rate")),
            "motion_score": _as_float(row.get("motion_score")),
            "visibility_mean": _as_float(row.get("visibility_mean")),
            "camera_angle_proxy": str(row.get("camera_angle_proxy") or "unknown"),
            "review_file_size_mb": _as_float(row.get("review_file_size_mb")),
            "review_video_max_height": _as_int(row.get("review_video_max_height")),
            "review_video_quality": str(row.get("review_video_quality") or ""),
        }
        enriched.append(clip)

    enriched.sort(key=lambda row: _candidate_priority(row, int(row["source_index"])))
    return enriched[: config.limit]


def load_annotations(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "annotations": {}}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected annotation object in {path}")
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected annotations map in {path}")
    return {"version": payload.get("version", 1), **payload, "annotations": annotations}


def write_annotations(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": utc_now_iso()}
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=str(path.parent),
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        temp_path = Path(f.name)
    temp_path.replace(path)


def _clamp_time(value: Any, duration: float | None) -> float | None:
    if value in (None, ""):
        return None
    parsed = max(0.0, _as_float(value))
    if duration and duration > 0:
        parsed = min(parsed, duration)
    return round(parsed, 2)


def sanitize_annotation(
    data: dict[str, Any],
    clip_lookup: dict[str, dict[str, Any]],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(data.get("candidate_id") or "").strip()
    if candidate_id not in clip_lookup:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")

    clip = clip_lookup[candidate_id]
    duration = _as_float(clip.get("duration_seconds_local")) or None
    quality = str(data.get("quality") or "").strip().lower()
    if quality not in QUALITY_VALUES:
        raise ValueError(f"quality must be one of: {', '.join(sorted(QUALITY_VALUES - {''}))}")

    existing = existing or {}
    camera_angle = str(
        data.get("camera_angle") or existing.get("camera_angle") or "unknown"
    ).strip().lower()
    if camera_angle not in CAMERA_ANGLE_VALUES:
        valid_angles = ", ".join(sorted(CAMERA_ANGLE_VALUES - {""}))
        raise ValueError(f"camera_angle must be one of: {valid_angles}")
    if not camera_angle:
        camera_angle = "unknown"

    start_seconds = _clamp_time(data.get("start_seconds"), duration)
    end_seconds = _clamp_time(data.get("end_seconds"), duration)
    if start_seconds is not None and end_seconds is not None and end_seconds < start_seconds:
        start_seconds, end_seconds = end_seconds, start_seconds

    annotation = {
        **existing,
        "candidate_id": candidate_id,
        "quality": quality,
        "camera_angle": camera_angle,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "notes": str(data.get("notes") or "").strip()[:4000],
        "updated_at": utc_now_iso(),
    }
    return annotation


def save_annotation(
    config: ReviewAppConfig,
    data: dict[str, Any],
    clip_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clips = load_review_clips(config) if clip_lookup is None else list(clip_lookup.values())
    lookup = {str(clip["candidate_id"]): clip for clip in clips} if clip_lookup is None else clip_lookup
    payload = load_annotations(config.annotations_path)
    annotations = payload.setdefault("annotations", {})
    candidate_id = str(data.get("candidate_id") or "").strip()
    annotation = sanitize_annotation(data, lookup, annotations.get(candidate_id))
    annotations[candidate_id] = annotation
    write_annotations(config.annotations_path, payload)
    return annotation


def build_clips_payload(config: ReviewAppConfig) -> dict[str, Any]:
    clips = load_review_clips(config)
    annotations = load_annotations(config.annotations_path).get("annotations", {})
    counts = {"good": 0, "mid": 0, "bad": 0, "unreviewed": 0}
    hydrated_clips: list[dict[str, Any]] = []

    for index, clip in enumerate(clips, start=1):
        annotation = annotations.get(clip["candidate_id"], {})
        quality = annotation.get("quality") or "unreviewed"
        if quality not in counts:
            quality = "unreviewed"
        counts[quality] += 1
        hydrated_clips.append(
            {
                **clip,
                "rank": index,
                "video_url": f"/video/{clip['candidate_id']}{Path(str(clip['video_path'])).suffix}",
                "annotation": annotation,
                "review_quality": quality,
            }
        )

    reviewed = counts["good"] + counts["mid"] + counts["bad"]
    return {
        "clips": hydrated_clips,
        "counts": counts,
        "reviewed": reviewed,
        "total": len(hydrated_clips),
        "source_path": str(config.source_path),
        "annotations_path": str(config.annotations_path),
    }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=str(path.parent),
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        temp_path = Path(f.name)
    temp_path.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _cv_run_root(config: ReviewAppConfig) -> Path:
    return (config.repo_root / "artifacts/cv_runs").resolve()


def _safe_candidate_id(candidate_id: str) -> str:
    candidate_id = candidate_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_id):
        raise ValueError(f"Invalid candidate id: {candidate_id!r}")
    return candidate_id


def _cv_run_dir(config: ReviewAppConfig, candidate_id: str) -> Path:
    candidate_id = _safe_candidate_id(candidate_id)
    root = _cv_run_root(config)
    run_dir = (root / candidate_id).resolve()
    if root not in run_dir.parents and run_dir != root:
        raise ValueError(f"Invalid run path for candidate id: {candidate_id}")
    return run_dir


def _is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "image"
    if suffix in {".mp4", ".webm", ".mov"}:
        return "video"
    if suffix in {".json", ".jsonl"}:
        return "data"
    return "file"


def _artifact_url(candidate_id: str, path: Path, run_dir: Path) -> str | None:
    resolved = path.resolve()
    if not _is_relative_to(resolved, run_dir) or not resolved.exists():
        return None
    return f"/cv-artifacts/{candidate_id}/{resolved.name}"


def _stage_payload(manifest: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    stages = json.loads(json.dumps(manifest.get("stages", {})))
    selection = prompt.get("selection", {}) if isinstance(prompt, dict) else {}
    prompt_ready = selection.get("type") not in (None, "", "unset")
    if prompt_ready and isinstance(stages.get("person_prompt"), dict):
        stages["person_prompt"]["status"] = "ready"
    if prompt_ready and isinstance(stages.get("whole_runner_mask"), dict):
        if stages["whole_runner_mask"].get("status") == "pending_prompt":
            stages["whole_runner_mask"]["status"] = "pending_run"
    return stages


def load_cv_run_payload(config: ReviewAppConfig, candidate_id: str) -> dict[str, Any]:
    candidate_id = _safe_candidate_id(candidate_id)
    run_dir = _cv_run_dir(config, candidate_id)
    manifest_path = run_dir / "cv_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CV run not found: {candidate_id}")

    manifest = _read_json(manifest_path, {})
    paths = manifest.get("paths", {})
    prompt_path = Path(str(paths.get("person_prompt") or run_dir / "person_prompt.json")).resolve()
    prompt = _read_json(prompt_path, {"selection": {"type": "unset"}})
    artifacts: dict[str, dict[str, Any]] = {}

    for key, label in CV_ARTIFACT_LABELS.items():
        raw_path = paths.get(key)
        artifact_path = Path(str(raw_path)).resolve() if raw_path else (run_dir / f"{key}.missing")
        artifacts[key] = {
            "key": key,
            "label": label,
            "path": str(artifact_path),
            "exists": artifact_path.exists() and _is_relative_to(artifact_path, run_dir),
            "kind": _artifact_kind(artifact_path),
            "url": _artifact_url(candidate_id, artifact_path, run_dir),
        }

    return {
        "candidate_id": candidate_id,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "prompt": prompt,
        "stages": _stage_payload(manifest, prompt),
        "artifacts": artifacts,
    }


def list_cv_runs(config: ReviewAppConfig) -> list[dict[str, Any]]:
    root = _cv_run_root(config)
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/cv_run_manifest.json"), key=lambda p: p.stat().st_mtime):
        candidate_id = manifest_path.parent.name
        try:
            payload = load_cv_run_payload(config, candidate_id)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        manifest = payload["manifest"]
        runs.append(
            {
                "candidate_id": candidate_id,
                "runner_name": manifest.get("runner_name", "Unknown runner"),
                "camera_angle": manifest.get("review", {}).get("camera_angle", "unknown"),
                "duration_seconds": manifest.get("review", {}).get("duration_seconds"),
                "title": manifest.get("source", {}).get("title", "Untitled run"),
                "prompt_ready": payload["prompt"].get("selection", {}).get("type") not in (None, "", "unset"),
            }
        )
    return list(reversed(runs))


def _clamp_unit(value: Any) -> float:
    return round(max(0.0, min(1.0, _as_float(value))), 6)


def _sanitize_point(point: Any) -> dict[str, float]:
    if not isinstance(point, dict):
        raise ValueError("Prompt points must be objects with x and y")
    return {"x": _clamp_unit(point.get("x")), "y": _clamp_unit(point.get("y"))}


def _sanitize_box(box: Any) -> dict[str, float] | None:
    if box in (None, ""):
        return None
    if not isinstance(box, dict):
        raise ValueError("Prompt box must be an object")
    x = _clamp_unit(box.get("x"))
    y = _clamp_unit(box.get("y"))
    width = min(_clamp_unit(box.get("width")), 1.0 - x)
    height = min(_clamp_unit(box.get("height")), 1.0 - y)
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": round(width, 6), "height": round(height, 6)}


def sanitize_prompt_selection(selection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")

    positive_points = [
        _sanitize_point(point) for point in selection.get("positive_points", [])[:MAX_PROMPT_POINTS]
    ]
    negative_points = [
        _sanitize_point(point) for point in selection.get("negative_points", [])[:MAX_PROMPT_POINTS]
    ]
    box = _sanitize_box(selection.get("box"))
    selection_type = str(selection.get("type") or "unset").strip().lower()
    if box:
        selection_type = "box"
    elif positive_points or negative_points:
        selection_type = "point"
    elif selection.get("mask_path"):
        selection_type = "mask"
    else:
        selection_type = "unset"

    if selection_type not in PROMPT_SELECTION_TYPES:
        raise ValueError(f"selection.type must be one of: {', '.join(sorted(PROMPT_SELECTION_TYPES))}")

    return {
        "type": selection_type,
        "positive_points": positive_points,
        "negative_points": negative_points,
        "box": box,
        "mask_path": selection.get("mask_path") or None,
    }


def save_cv_prompt(config: ReviewAppConfig, candidate_id: str, data: dict[str, Any]) -> dict[str, Any]:
    run_dir = _cv_run_dir(config, candidate_id)
    manifest_path = run_dir / "cv_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CV run not found: {candidate_id}")

    manifest = _read_json(manifest_path, {})
    prompt_path = Path(str(manifest.get("paths", {}).get("person_prompt") or run_dir / "person_prompt.json"))
    prompt_path = prompt_path.resolve()
    if not _is_relative_to(prompt_path, run_dir):
        raise ValueError("Prompt path must stay inside the CV run directory")

    prompt = _read_json(prompt_path, {"version": 1, "candidate_id": candidate_id})
    prompt["selection"] = sanitize_prompt_selection(data.get("selection", data))
    prompt["updated_at"] = utc_now_iso()
    _write_json(prompt_path, prompt)

    stages = manifest.setdefault("stages", {})
    stages.setdefault("person_prompt", {})["status"] = "ready"
    whole_runner_mask = stages.setdefault("whole_runner_mask", {})
    if whole_runner_mask.get("status") in (None, "", "pending_prompt"):
        whole_runner_mask["status"] = "pending_run"
    manifest["updated_at"] = utc_now_iso()
    _write_json(manifest_path, manifest)
    return load_cv_run_payload(config, candidate_id)


def _sam2_checkpoint(config: ReviewAppConfig) -> Path:
    return (config.repo_root / SAM2_CHECKPOINT).resolve()


def _validate_mask_backend(backend: str) -> str:
    backend = backend.strip().lower()
    if backend not in MASK_BACKENDS:
        valid = ", ".join(sorted(MASK_BACKENDS))
        raise ValueError(f"Mask backend must be one of: {valid}")
    return backend


def _mask_default_job(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "backend": None,
        "options": {},
        "status": "idle",
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None,
    }


def mask_job_status(candidate_id: str) -> dict[str, Any]:
    candidate_id = _safe_candidate_id(candidate_id)
    with MASK_JOBS_LOCK:
        return dict(MASK_JOBS.get(candidate_id, _mask_default_job(candidate_id)))


def sam2_job_status(candidate_id: str) -> dict[str, Any]:
    return mask_job_status(candidate_id)


def _set_mask_job(candidate_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    with MASK_JOBS_LOCK:
        current = MASK_JOBS.get(candidate_id, _mask_default_job(candidate_id))
        MASK_JOBS[candidate_id] = {**current, **updates}
        return dict(MASK_JOBS[candidate_id])


def _set_mask_stage_status(
    config: ReviewAppConfig,
    candidate_id: str,
    status: str,
    *,
    backend: str | None = None,
    error: str | None = None,
) -> None:
    run_dir = _cv_run_dir(config, candidate_id)
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = _read_json(manifest_path, {})
    stages = manifest.setdefault("stages", {})
    whole_runner_mask = stages.setdefault("whole_runner_mask", {})
    whole_runner_mask["status"] = status
    if backend:
        whole_runner_mask["backend"] = backend
    if error:
        whole_runner_mask["error"] = error
    elif "error" in whole_runner_mask:
        del whole_runner_mask["error"]
    manifest["updated_at"] = utc_now_iso()
    _write_json(manifest_path, manifest)


def _sanitize_mask_options(data: dict[str, Any]) -> dict[str, Any]:
    quality_mode = str(data.get("quality_mode") or data.get("mode") or "native").strip().lower()
    if quality_mode not in MASK_QUALITY_MODES:
        valid = ", ".join(sorted(MASK_QUALITY_MODES))
        raise ValueError(f"quality_mode must be one of: {valid}")

    options: dict[str, Any] = {"quality_mode": quality_mode}
    if data.get("resolution") not in (None, ""):
        resolution = max(1, _as_int(data.get("resolution")))
        options["resolution"] = resolution
    return options


def _run_mask_job(
    config: ReviewAppConfig,
    candidate_id: str,
    backend: str,
    options: dict[str, Any],
) -> None:
    try:
        if backend == "sam2":
            from whodoirunlike.sam2_runner import run_sam2_mask

            result = run_sam2_mask(
                run_dir=_cv_run_dir(config, candidate_id),
                checkpoint=_sam2_checkpoint(config),
                model_cfg=SAM2_MODEL_CFG,
            )
            result["backend"] = "sam2"
        elif backend == "sam31_mlx":
            from whodoirunlike.sam31_mlx_runner import run_sam31_mlx_mask

            result = run_sam31_mlx_mask(
                run_dir=_cv_run_dir(config, candidate_id),
                quality_mode=str(options.get("quality_mode") or "native"),
                resolution=options.get("resolution"),
            )
        else:
            raise ValueError(f"Unsupported mask backend: {backend}")

        _set_mask_job(
            candidate_id,
            {
                "status": "completed",
                "completed_at": utc_now_iso(),
                "error": None,
                "result": result,
            },
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the local review UI.
        error = str(exc)
        _set_mask_stage_status(config, candidate_id, "failed", backend=backend, error=error)
        _set_mask_job(
            candidate_id,
            {
                "status": "failed",
                "completed_at": utc_now_iso(),
                "error": error,
                "result": None,
            },
        )


def start_mask_job(
    config: ReviewAppConfig,
    candidate_id: str,
    *,
    backend: str = "sam2",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = _validate_mask_backend(backend)
    options = options or {}
    options = _sanitize_mask_options(options) if backend == "sam31_mlx" else {}
    candidate_id = _safe_candidate_id(candidate_id)
    run_dir = _cv_run_dir(config, candidate_id)
    if not (run_dir / "cv_run_manifest.json").exists():
        raise FileNotFoundError(f"CV run not found: {candidate_id}")
    if backend == "sam2":
        checkpoint = _sam2_checkpoint(config)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")

    existing = mask_job_status(candidate_id)
    if existing["status"] == "running":
        return existing

    _set_mask_stage_status(config, candidate_id, "running", backend=backend)
    _set_mask_job(
        candidate_id,
        {
            "candidate_id": candidate_id,
            "backend": backend,
            "options": options,
            "status": "running",
            "started_at": utc_now_iso(),
            "completed_at": None,
            "error": None,
            "result": None,
        },
    )
    thread = threading.Thread(
        target=_run_mask_job,
        args=(config, candidate_id, backend, options),
        name=f"mask-{backend}-{candidate_id}",
        daemon=True,
    )
    thread.start()
    return mask_job_status(candidate_id)


def start_sam2_job(config: ReviewAppConfig, candidate_id: str) -> dict[str, Any]:
    return start_mask_job(config, candidate_id, backend="sam2")


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "WhoDoIRunLikeReview/0.1"
    config: ReviewAppConfig

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/clips":
            self._send_json(build_clips_payload(self.config))
            return
        if parsed.path == "/api/cv-runs":
            self._send_json({"runs": list_cv_runs(self.config)})
            return
        if parsed.path.startswith("/api/cv-runs/") and parsed.path.endswith("/mask"):
            self._handle_get_mask_job(parsed.path)
            return
        if parsed.path.startswith("/api/cv-runs/") and parsed.path.endswith("/sam2"):
            self._handle_get_sam2_job(parsed.path)
            return
        if parsed.path.startswith("/api/cv-runs/"):
            self._handle_get_cv_run(parsed.path)
            return
        if parsed.path.startswith("/video/"):
            self._serve_video(parsed.path)
            return
        if parsed.path.startswith("/cv-artifacts/"):
            self._serve_cv_artifact(parsed.path)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/cv-runs/") and parsed.path.endswith("/mask"):
            self._handle_start_mask_job(parsed.path)
            return
        if parsed.path.startswith("/api/cv-runs/") and parsed.path.endswith("/sam2"):
            self._handle_start_sam2_job(parsed.path)
            return
        if parsed.path.startswith("/api/cv-runs/") and parsed.path.endswith("/prompt"):
            self._handle_save_cv_prompt(parsed.path)
            return

        if parsed.path != "/api/annotations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = min(int(self.headers.get("Content-Length", "0")), 64_000)
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            clips = load_review_clips(self.config)
            lookup = {str(clip["candidate_id"]): clip for clip in clips}
            annotation = save_annotation(self.config, data, lookup)
            self._send_json({"annotation": annotation, **build_clips_payload(self.config)})
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/video/"):
            self._serve_video(parsed.path, head_only=True)
            return
        if parsed.path.startswith("/cv-artifacts/"):
            self._serve_cv_artifact(parsed.path, head_only=True)
            return
        self._serve_static(parsed.path, head_only=True)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_get_cv_run(self, path: str) -> None:
        candidate_id = unquote(path.removeprefix("/api/cv-runs/")).strip("/")
        try:
            self._send_json(load_cv_run_payload(self.config, candidate_id))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)

    def _handle_save_cv_prompt(self, path: str) -> None:
        candidate_id = unquote(path.removeprefix("/api/cv-runs/").removesuffix("/prompt")).strip("/")
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 64_000)
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send_json(save_cv_prompt(self.config, candidate_id, data))
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)

    def _handle_get_sam2_job(self, path: str) -> None:
        candidate_id = unquote(path.removeprefix("/api/cv-runs/").removesuffix("/sam2")).strip("/")
        try:
            run_dir = _cv_run_dir(self.config, candidate_id)
            if not (run_dir / "cv_run_manifest.json").exists():
                raise FileNotFoundError(f"CV run not found: {candidate_id}")
            self._send_json({"job": mask_job_status(candidate_id)})
        except (FileNotFoundError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)

    def _handle_start_sam2_job(self, path: str) -> None:
        candidate_id = unquote(path.removeprefix("/api/cv-runs/").removesuffix("/sam2")).strip("/")
        try:
            self._send_json({"job": start_sam2_job(self.config, candidate_id)})
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_get_mask_job(self, path: str) -> None:
        candidate_id = unquote(path.removeprefix("/api/cv-runs/").removesuffix("/mask")).strip("/")
        try:
            run_dir = _cv_run_dir(self.config, candidate_id)
            if not (run_dir / "cv_run_manifest.json").exists():
                raise FileNotFoundError(f"CV run not found: {candidate_id}")
            self._send_json(
                {
                    "job": mask_job_status(candidate_id),
                    "backends": MASK_BACKENDS,
                    "quality_modes": MASK_QUALITY_MODES,
                }
            )
        except (FileNotFoundError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)

    def _handle_start_mask_job(self, path: str) -> None:
        candidate_id = unquote(path.removeprefix("/api/cv-runs/").removesuffix("/mask")).strip("/")
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 64_000)
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            backend = str(data.get("backend") or "sam2")
            options = _sanitize_mask_options(data) if backend == "sam31_mlx" else {}
            self._send_json(
                {
                    "job": start_mask_job(
                        self.config,
                        candidate_id,
                        backend=backend,
                        options=options,
                    ),
                    "backends": MASK_BACKENDS,
                    "quality_modes": MASK_QUALITY_MODES,
                }
            )
        except json.JSONDecodeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _serve_static(self, path: str, *, head_only: bool = False) -> None:
        if path in ("", "/"):
            path = "/index.html"
        relative_path = Path(unquote(path).lstrip("/"))
        static_path = (self.config.static_dir / relative_path).resolve()
        static_root = self.config.static_dir.resolve()
        if static_root not in static_path.parents and static_path != static_root:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not static_path.exists() or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = b"" if head_only else static_path.read_bytes()
        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(static_path.stat().st_size))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _serve_video(self, path: str, *, head_only: bool = False) -> None:
        candidate_id = Path(unquote(path)).stem
        clips = load_review_clips(self.config)
        clip_lookup = {str(clip["candidate_id"]): clip for clip in clips}
        clip = clip_lookup.get(candidate_id)
        if not clip:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        video_path = Path(str(clip["video_path"]))
        if not video_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self._serve_file_path(video_path, head_only=head_only)

    def _serve_cv_artifact(self, path: str, *, head_only: bool = False) -> None:
        parts = [unquote(part) for part in path.removeprefix("/cv-artifacts/").split("/") if part]
        if len(parts) != 2:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        candidate_id, filename = parts
        try:
            run_dir = _cv_run_dir(self.config, candidate_id)
            artifact_path = (run_dir / filename).resolve()
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not _is_relative_to(artifact_path, run_dir) or not artifact_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._serve_file_path(artifact_path, head_only=head_only)

    def _serve_file_path(self, file_path: Path, *, head_only: bool = False) -> None:
        file_size = file_path.stat().st_size
        start, end, partial = self._range_bounds(file_size)
        if start >= file_size:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return

        content_length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if head_only:
            return

        with file_path.open("rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(remaining, 256 * 1024))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, socket.timeout):
                    break
                remaining -= len(chunk)

    def _range_bounds(self, file_size: int) -> tuple[int, int, bool]:
        header = self.headers.get("Range")
        if not header:
            return 0, file_size - 1, False

        match = re.match(r"bytes=(\d*)-(\d*)$", header.strip())
        if not match:
            return 0, file_size - 1, False

        start_raw, end_raw = match.groups()
        if start_raw == "" and end_raw:
            suffix = min(int(end_raw), file_size)
            return file_size - suffix, file_size - 1, True

        start = int(start_raw or 0)
        end = int(end_raw) if end_raw else file_size - 1
        return start, min(end, file_size - 1), True


def make_handler(config: ReviewAppConfig) -> type[ReviewRequestHandler]:
    class ConfiguredReviewRequestHandler(ReviewRequestHandler):
        pass

    ConfiguredReviewRequestHandler.config = config
    return ConfiguredReviewRequestHandler


def run_review_server(host: str, port: int, config: ReviewAppConfig) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(config))
    return server


def parse_query_limit(default: int, query: str) -> int:
    params = parse_qs(query)
    if "limit" not in params:
        return default
    return max(1, min(100, _as_int(params["limit"][0], default)))
