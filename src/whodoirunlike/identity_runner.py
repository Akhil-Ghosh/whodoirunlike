from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from whodoirunlike.cv_flow import read_json, utc_now_iso, write_json


DEFAULT_IDENTITY_BACKEND = "prompt_template_tracker_v1"
IdentityProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class VideoFrames:
    frames: list[np.ndarray]
    fps: float
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def build_identity_progress(
    *,
    phase: str,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    total_frames = max(0, int(total_frames))
    processed_frames = max(0, min(int(processed_frames), total_frames or int(processed_frames)))
    percent = processed_frames / total_frames if total_frames else 0.0
    eta_seconds: float | None = None
    if processed_frames and total_frames > processed_frames and elapsed_seconds > 0:
        eta_seconds = (elapsed_seconds / processed_frames) * (total_frames - processed_frames)
    elif total_frames and processed_frames >= total_frames:
        eta_seconds = 0.0
    return {
        "phase": phase,
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "percent": round(percent, 4),
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 1),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
    }


def load_video_frames(video_path: Path) -> VideoFrames:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    return VideoFrames(frames=frames, fps=fps, width=width, height=height)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _selection(prompt: dict[str, Any]) -> dict[str, Any]:
    selection = prompt.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("person_prompt.json selection must be an object")
    if selection.get("type") in (None, "", "unset"):
        raise ValueError("Select and save the target runner before running identity tracking")
    return selection


def _normalized_box_to_pixels(box: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    x = _clamp(float(box.get("x") or 0.0), 0.0, 1.0) * width
    y = _clamp(float(box.get("y") or 0.0), 0.0, 1.0) * height
    w = _clamp(float(box.get("width") or 0.0), 0.0, 1.0) * width
    h = _clamp(float(box.get("height") or 0.0), 0.0, 1.0) * height
    if w < 2 or h < 2:
        return None
    return _clip_box((x, y, w, h), width, height)


def prompt_initial_box(prompt: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    selection = _selection(prompt)
    box = selection.get("box")
    if isinstance(box, dict):
        pixel_box = _normalized_box_to_pixels(box, width, height)
        if pixel_box:
            return pixel_box

    subject_candidate = selection.get("subject_candidate")
    if isinstance(subject_candidate, dict) and isinstance(subject_candidate.get("box"), dict):
        pixel_box = _normalized_box_to_pixels(subject_candidate["box"], width, height)
        if pixel_box:
            return pixel_box

    points = selection.get("positive_points") or []
    if not points:
        raise ValueError("Identity tracking needs a positive point or box prompt")
    point = points[0]
    anchor_x = _clamp(float(point.get("x") or 0.5), 0.0, 1.0) * width
    anchor_y = _clamp(float(point.get("y") or 0.5), 0.0, 1.0) * height
    box_width = max(16.0, width * 0.16)
    box_height = max(32.0, height * 0.42)
    return _clip_box(
        (
            anchor_x - box_width / 2.0,
            anchor_y - box_height * 0.45,
            box_width,
            box_height,
        ),
        width,
        height,
    )


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    w = max(2.0, min(float(w), max(2.0, width - 1.0)))
    h = max(2.0, min(float(h), max(2.0, height - 1.0)))
    x = _clamp(float(x), 0.0, max(0.0, width - w))
    y = _clamp(float(y), 0.0, max(0.0, height - h))
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _hist_embedding(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(64, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    vector = hist.flatten().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _match_next_box(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], float]:
    height, width = current_gray.shape[:2]
    x, y, w, h = previous_box
    template = previous_gray[y : y + h, x : x + w]
    if template.size == 0:
        return previous_box, 0.0

    pad_x = max(8, int(round(w * 0.85)))
    pad_y = max(8, int(round(h * 0.85)))
    sx = max(0, x - pad_x)
    sy = max(0, y - pad_y)
    ex = min(width, x + w + pad_x)
    ey = min(height, y + h + pad_y)
    search = current_gray[sy:ey, sx:ex]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return previous_box, 0.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    nx = sx + max_location[0]
    ny = sy + max_location[1]
    return _clip_box((nx, ny, w, h), width, height), float(max_value)


def _box_payload(box: tuple[int, int, int, int], width: int, height: int) -> dict[str, float]:
    x, y, w, h = box
    return {
        "x": round(x / max(width, 1), 6),
        "y": round(y / max(height, 1), 6),
        "width": round(w / max(width, 1), 6),
        "height": round(h / max(height, 1), 6),
    }


def _track_row(
    *,
    frame_index: int,
    fps: float,
    width: int,
    height: int,
    box: tuple[int, int, int, int] | None,
    template_score: float,
    reid_similarity: float,
    state: str,
    reasons: list[str],
) -> dict[str, Any]:
    if box is None:
        x = y = w = h = area = None
    else:
        x, y, w, h = box
        area = (w * h) / float(max(width * height, 1))
    return {
        "frame_index": int(frame_index),
        "time_seconds": round(frame_index / fps, 6) if fps > 0 else None,
        "track_id": 1,
        "is_target": True,
        "bbox_x": round(x / width, 6) if x is not None and width else None,
        "bbox_y": round(y / height, 6) if y is not None and height else None,
        "bbox_width": round(w / width, 6) if w is not None and width else None,
        "bbox_height": round(h / height, 6) if h is not None and height else None,
        "bbox_area_ratio": round(area, 8) if area is not None else None,
        "template_score": round(float(template_score), 6),
        "reid_similarity": round(float(reid_similarity), 6),
        "identity_state": state,
        "identity_risk": state == "identity_risk",
        "tracker_ok": state != "missing",
        "reasons": reasons,
    }


def _reid_row(
    *,
    frame_index: int,
    fps: float,
    embedding: np.ndarray,
    similarity: float,
    memory_updated: bool,
) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "time_seconds": round(frame_index / fps, 6) if fps > 0 else None,
        "track_id": 1,
        "is_target": True,
        "embedding_model": "hsv_histogram_8x8",
        "embedding_dim": int(embedding.shape[0]),
        "embedding": [round(float(value), 8) for value in embedding.tolist()],
        "similarity_to_target_memory": round(float(similarity), 6),
        "memory_updated": bool(memory_updated),
    }


def _state_for_scores(
    *,
    template_score: float,
    reid_similarity: float,
    reid_accept: float,
    reid_recover: float,
) -> tuple[str, list[str], bool]:
    reasons: list[str] = []
    if template_score < 0.12:
        reasons.append("template_match_missing")
        return "missing", reasons, False
    if template_score < 0.28:
        reasons.append("low_template_score")
    if reid_similarity < reid_recover:
        reasons.append("low_reid_similarity")
    state = "usable" if not reasons else "identity_risk"
    memory_updated = state == "usable" and reid_similarity >= reid_accept
    return state, reasons, memory_updated


def _track_direction(
    *,
    frames: list[np.ndarray],
    fps: float,
    start_index: int,
    initial_box: tuple[int, int, int, int],
    initial_embedding: np.ndarray,
    step: int,
    reid_accept: float,
    reid_recover: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    height, width = frames[0].shape[:2]
    gray_frames = [_gray(frame) for frame in frames]
    rows: dict[int, dict[str, Any]] = {}
    reid_rows: dict[int, dict[str, Any]] = {}
    memory = initial_embedding.copy()
    current_box = initial_box
    previous_gray = gray_frames[start_index]

    frame_index = start_index + step
    while 0 <= frame_index < len(frames):
        matched_box, template_score = _match_next_box(previous_gray, gray_frames[frame_index], current_box)
        embedding = _hist_embedding(_crop(frames[frame_index], matched_box))
        similarity = _cosine_similarity(embedding, memory)
        state, reasons, memory_updated = _state_for_scores(
            template_score=template_score,
            reid_similarity=similarity,
            reid_accept=reid_accept,
            reid_recover=reid_recover,
        )
        if memory_updated:
            blended = (memory * 0.9) + (embedding * 0.1)
            norm = float(np.linalg.norm(blended))
            memory = blended / norm if norm > 0 else blended

        rows[frame_index] = _track_row(
            frame_index=frame_index,
            fps=fps,
            width=width,
            height=height,
            box=matched_box,
            template_score=template_score,
            reid_similarity=similarity,
            state=state,
            reasons=reasons,
        )
        reid_rows[frame_index] = _reid_row(
            frame_index=frame_index,
            fps=fps,
            embedding=embedding,
            similarity=similarity,
            memory_updated=memory_updated,
        )
        current_box = matched_box
        previous_gray = gray_frames[frame_index]
        frame_index += step

    return rows, reid_rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Identity tracking needs pyarrow to write Parquet. "
            "Install with: python -m pip install -e '.[identity]'"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def identity_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for row in rows:
        state = str(row.get("identity_state") or "missing")
        if state == "usable":
            if active:
                active["end_frame_index"] = int(row["frame_index"]) - 1
                active["end_time_seconds"] = row.get("time_seconds")
                segments.append(active)
                active = None
            continue
        if active and active["state"] == state:
            active["end_frame_index"] = int(row["frame_index"])
            active["end_time_seconds"] = row.get("time_seconds")
            continue
        if active:
            segments.append(active)
        active = {
            "state": state,
            "start_frame_index": int(row["frame_index"]),
            "end_frame_index": int(row["frame_index"]),
            "start_time_seconds": row.get("time_seconds"),
            "end_time_seconds": row.get("time_seconds"),
        }
    if active:
        segments.append(active)
    return segments


def summarize_identity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    usable = sum(1 for row in rows if row.get("identity_state") == "usable")
    risk = sum(1 for row in rows if row.get("identity_state") == "identity_risk")
    missing = sum(1 for row in rows if row.get("identity_state") == "missing")
    similarities = [
        float(row["reid_similarity"])
        for row in rows
        if row.get("reid_similarity") not in (None, "")
    ]
    return {
        "frame_count": total,
        "usable_frames": usable,
        "identity_risk_frames": risk,
        "missing_frames": missing,
        "target_identity_stability_rate": round(usable / total, 6) if total else 0.0,
        "identity_risk_rate": round(risk / total, 6) if total else 0.0,
        "missing_rate": round(missing / total, 6) if total else 0.0,
        "mean_reid_similarity": round(float(np.mean(similarities)), 6) if similarities else 0.0,
        "min_reid_similarity": round(float(np.min(similarities)), 6) if similarities else 0.0,
        "identity_risk_segments": identity_segments(rows),
    }


def _read_reid_threshold(seed: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(seed.get("reid", {}).get(key, default))
    except (TypeError, ValueError):
        return default


def update_manifest_after_identity_tracking(
    manifest_path: Path,
    *,
    tracklets_path: Path,
    reid_path: Path,
    tracklets_jsonl_path: Path,
    reid_jsonl_path: Path,
    qc_metrics_path: Path,
    result: dict[str, Any],
) -> None:
    manifest = read_json(manifest_path)
    stages = manifest.setdefault("stages", {})
    detector_tracker = stages.setdefault("detector_tracker", {})
    detector_tracker.update(
        {
            "status": "complete",
            "backend": result["backend"],
            "track_seed": str(result["track_seed_path"]),
            "tracklets": str(tracklets_path),
            "tracklets_jsonl": str(tracklets_jsonl_path),
            "reid": str(reid_path),
            "reid_jsonl": str(reid_jsonl_path),
            "qc_metrics": str(qc_metrics_path),
            "metrics": result["metrics"],
            "completed_at": utc_now_iso(),
        }
    )
    whole_runner_mask = stages.setdefault("whole_runner_mask", {})
    if whole_runner_mask.get("status") in (None, "", "pending_prompt", "pending_tracker"):
        whole_runner_mask["status"] = "pending_run"
    whole_runner_mask["identity_gate"] = "detector_tracker"
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def update_manifest_after_identity_failure(manifest_path: Path, *, error: str) -> None:
    manifest = read_json(manifest_path)
    stage = manifest.setdefault("stages", {}).setdefault("detector_tracker", {})
    stage["status"] = "failed"
    stage["backend"] = DEFAULT_IDENTITY_BACKEND
    stage["error"] = error
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def run_identity_tracking(
    *,
    run_dir: Path,
    progress_callback: IdentityProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = read_json(manifest_path)
    paths = manifest["paths"]
    source_segment = Path(str(paths["source_segment"]))
    prompt_path = Path(str(paths.get("person_prompt") or run_dir / "person_prompt.json"))
    track_seed_path = Path(str(paths.get("track_seed") or run_dir / "track_seed.json"))
    tracklets_path = Path(str(paths.get("tracklets") or run_dir / "tracklets.parquet"))
    reid_path = Path(str(paths.get("reid") or run_dir / "reid.parquet"))
    tracklets_jsonl_path = run_dir / "tracklets.jsonl"
    reid_jsonl_path = run_dir / "reid.jsonl"
    qc_metrics_path = Path(str(paths.get("qc_metrics") or run_dir / "qc_metrics.json"))

    try:
        frames = load_video_frames(source_segment)
        prompt = read_json(prompt_path)
        seed = read_json(track_seed_path) if track_seed_path.exists() else {}
        prompt_frame = prompt.get("frame", {}) if isinstance(prompt.get("frame"), dict) else {}
        start_index = prompt_frame.get("frame_index")
        if start_index in (None, ""):
            start_index = frames.frame_count // 2
        start_index = max(0, min(int(start_index), frames.frame_count - 1))
        initial_box = prompt_initial_box(prompt, frames.width, frames.height)
        initial_embedding = _hist_embedding(_crop(frames.frames[start_index], initial_box))
        reid_accept = _read_reid_threshold(seed, "cosine_accept", 0.65)
        reid_recover = _read_reid_threshold(seed, "cosine_recover", 0.58)

        if progress_callback:
            progress_callback(
                build_identity_progress(
                    phase="tracking",
                    processed_frames=1,
                    total_frames=frames.frame_count,
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )

        rows: dict[int, dict[str, Any]] = {
            start_index: _track_row(
                frame_index=start_index,
                fps=frames.fps,
                width=frames.width,
                height=frames.height,
                box=initial_box,
                template_score=1.0,
                reid_similarity=1.0,
                state="usable",
                reasons=[],
            )
        }
        reid_rows: dict[int, dict[str, Any]] = {
            start_index: _reid_row(
                frame_index=start_index,
                fps=frames.fps,
                embedding=initial_embedding,
                similarity=1.0,
                memory_updated=True,
            )
        }

        for step in (1, -1):
            direction_rows, direction_reid = _track_direction(
                frames=frames.frames,
                fps=frames.fps,
                start_index=start_index,
                initial_box=initial_box,
                initial_embedding=initial_embedding,
                step=step,
                reid_accept=reid_accept,
                reid_recover=reid_recover,
            )
            rows.update(direction_rows)
            reid_rows.update(direction_reid)
            if progress_callback:
                progress_callback(
                    build_identity_progress(
                        phase="tracking",
                        processed_frames=len(rows),
                        total_frames=frames.frame_count,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )

        ordered_rows = [rows[index] for index in sorted(rows)]
        ordered_reid_rows = [reid_rows[index] for index in sorted(reid_rows)]
        metrics = summarize_identity_rows(ordered_rows)
        prompt_box = _box_payload(initial_box, frames.width, frames.height)
        metrics["prompt_frame_index"] = start_index
        metrics["initial_prompt_box"] = prompt_box
        metrics["backend"] = DEFAULT_IDENTITY_BACKEND

        _write_parquet(tracklets_path, ordered_rows)
        _write_parquet(reid_path, ordered_reid_rows)
        _write_jsonl(tracklets_jsonl_path, ordered_rows)
        _write_jsonl(reid_jsonl_path, ordered_reid_rows)

        qc_metrics = {
            "version": 1,
            "candidate_id": manifest.get("candidate_id"),
            "updated_at": utc_now_iso(),
            "identity": metrics,
        }
        if qc_metrics_path.exists():
            existing_qc = read_json(qc_metrics_path)
            if isinstance(existing_qc, dict):
                qc_metrics = {**existing_qc, "identity": metrics, "updated_at": utc_now_iso()}
        write_json(qc_metrics_path, qc_metrics)

        seed.update(
            {
                "version": seed.get("version", 1),
                "candidate_id": manifest.get("candidate_id"),
                "status": "complete",
                "backend": DEFAULT_IDENTITY_BACKEND,
                "target_track_id": 1,
                "prompt_box": prompt_box,
                "outputs": {
                    "tracklets": str(tracklets_path),
                    "tracklets_jsonl": str(tracklets_jsonl_path),
                    "reid": str(reid_path),
                    "reid_jsonl": str(reid_jsonl_path),
                    "qc_metrics": str(qc_metrics_path),
                },
                "metrics": metrics,
                "updated_at": utc_now_iso(),
            }
        )
        write_json(track_seed_path, seed)

        result = {
            "candidate_id": manifest.get("candidate_id"),
            "backend": DEFAULT_IDENTITY_BACKEND,
            "status": "complete",
            "frame_count": len(ordered_rows),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "track_seed_path": str(track_seed_path),
            "tracklets_path": str(tracklets_path),
            "tracklets_jsonl_path": str(tracklets_jsonl_path),
            "reid_path": str(reid_path),
            "reid_jsonl_path": str(reid_jsonl_path),
            "qc_metrics_path": str(qc_metrics_path),
            "metrics": metrics,
        }
        update_manifest_after_identity_tracking(
            manifest_path,
            tracklets_path=tracklets_path,
            reid_path=reid_path,
            tracklets_jsonl_path=tracklets_jsonl_path,
            reid_jsonl_path=reid_jsonl_path,
            qc_metrics_path=qc_metrics_path,
            result=result,
        )
        if progress_callback:
            progress_callback(
                build_identity_progress(
                    phase="completed",
                    processed_frames=len(ordered_rows),
                    total_frames=frames.frame_count,
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        return result
    except Exception as exc:
        update_manifest_after_identity_failure(manifest_path, error=str(exc))
        raise
