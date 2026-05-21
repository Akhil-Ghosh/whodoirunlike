from __future__ import annotations

import json
import time
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from whodoirunlike.cv_flow import read_json, utc_now_iso, write_json


DEFAULT_IDENTITY_BACKEND = "boxmot_botsort"
TEMPLATE_IDENTITY_BACKEND = "prompt_template_tracker_v1"
BOXMOT_BACKENDS = {
    "boxmot_botsort": "BotSort",
    "boxmot_deepocsort": "DeepOcSort",
    "boxmot_bytetrack": "ByteTrack",
}
BOXMOT_TRACKER_TYPES = {
    "boxmot_botsort": "botsort",
    "boxmot_deepocsort": "deepocsort",
    "boxmot_bytetrack": "bytetrack",
}
BOXMOT_BACKEND_ALIASES = {
    "botsort": "boxmot_botsort",
    "bot-sort": "boxmot_botsort",
    "deepocsort": "boxmot_deepocsort",
    "deep-oc-sort": "boxmot_deepocsort",
    "bytetrack": "boxmot_bytetrack",
    "byte-track": "boxmot_bytetrack",
    "template": TEMPLATE_IDENTITY_BACKEND,
    "prompt_template": TEMPLATE_IDENTITY_BACKEND,
}
DEFAULT_DETECTOR_MODEL = "yolo11n.pt"
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"
IdentityProgressCallback = Callable[[dict[str, Any]], None]


def canonical_identity_backend(value: str | None) -> str:
    backend = str(value or DEFAULT_IDENTITY_BACKEND).strip().lower().replace("_", "-")
    normalized = backend.replace("-", "_")
    if normalized in BOXMOT_BACKENDS or normalized == TEMPLATE_IDENTITY_BACKEND:
        return normalized
    if backend in BOXMOT_BACKEND_ALIASES:
        return BOXMOT_BACKEND_ALIASES[backend]
    if normalized in BOXMOT_BACKEND_ALIASES:
        return BOXMOT_BACKEND_ALIASES[normalized]
    valid = sorted([*BOXMOT_BACKENDS, TEMPLATE_IDENTITY_BACKEND, *BOXMOT_BACKEND_ALIASES])
    raise ValueError(f"Unsupported identity backend: {value}. Expected one of: {', '.join(valid)}")


def identity_setup_status(backend: str | None = None) -> dict[str, Any]:
    backend_name = canonical_identity_backend(backend)
    if backend_name == TEMPLATE_IDENTITY_BACKEND:
        return {
            "backend": backend_name,
            "ready": True,
            "reasons": [],
            "install_command": None,
        }

    missing = [
        package
        for package in ("ultralytics", "boxmot")
        if importlib.util.find_spec(package) is None
    ]
    return {
        "backend": backend_name,
        "ready": not missing,
        "reasons": [f"Missing Python package: {package}" for package in missing],
        "install_command": 'python -m pip install -e ".[mot]"',
    }


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


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def result_to_person_detections(result: Any, person_class_id: int = 0) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = _as_numpy(getattr(boxes, "xyxy", None)).astype(np.float32, copy=False)
    if xyxy.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = xyxy.reshape((-1, 4))
    confidence = _as_numpy(getattr(boxes, "conf", None)).astype(np.float32, copy=False).reshape((-1,))
    classes = _as_numpy(getattr(boxes, "cls", None)).astype(np.float32, copy=False).reshape((-1,))
    if confidence.shape[0] != xyxy.shape[0]:
        confidence = np.ones((xyxy.shape[0],), dtype=np.float32)
    if classes.shape[0] != xyxy.shape[0]:
        classes = np.zeros((xyxy.shape[0],), dtype=np.float32)
    detections = np.concatenate([xyxy, confidence[:, None], classes[:, None]], axis=1)
    return detections[detections[:, 5].astype(int) == int(person_class_id)].astype(np.float32, copy=False)


def _xywh_to_xyxy(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


def _xyxy_to_xywh(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if len(box) < 4:
        return None
    x1 = _clamp(float(box[0]), 0.0, max(0.0, width - 1.0))
    y1 = _clamp(float(box[1]), 0.0, max(0.0, height - 1.0))
    x2 = _clamp(float(box[2]), 0.0, max(0.0, width - 1.0))
    y2 = _clamp(float(box[3]), 0.0, max(0.0, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return _clip_box((x1, y1, x2 - x1, y2 - y1), width, height)


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _prompt_points(prompt: dict[str, Any]) -> list[dict[str, float]]:
    selection = _selection(prompt)
    points = selection.get("positive_points") or []
    if not isinstance(points, list):
        return []
    parsed: list[dict[str, float]] = []
    for point in points:
        if isinstance(point, dict):
            parsed.append(
                {
                    "x": _clamp(float(point.get("x") or 0.0), 0.0, 1.0),
                    "y": _clamp(float(point.get("y") or 0.0), 0.0, 1.0),
                }
            )
    return parsed


def _prompt_point_score(
    points: Sequence[dict[str, float]],
    box_xyxy: Sequence[float],
    width: int,
    height: int,
) -> float:
    if not points:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in box_xyxy[:4]]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    best = 0.0
    for point in points:
        px = float(point["x"]) * width
        py = float(point["y"]) * height
        if x1 <= px <= x2 and y1 <= py <= y2:
            best = max(best, 1.0)
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        distance = np.hypot((px - cx) / box_width, (py - cy) / box_height)
        best = max(best, max(0.0, 1.0 - float(distance)))
    return best


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


def _boxmot_track_row(
    *,
    frame_index: int,
    fps: float,
    width: int,
    height: int,
    box: tuple[int, int, int, int] | None,
    track_id: int | None,
    is_target: bool,
    detection_confidence: float | None,
    reid_similarity: float | None,
    state: str,
    reasons: list[str],
    backend: str,
) -> dict[str, Any]:
    if box is None:
        x = y = w = h = area = None
    else:
        x, y, w, h = box
        area = (w * h) / float(max(width * height, 1))
    return {
        "frame_index": int(frame_index),
        "time_seconds": round(frame_index / fps, 6) if fps > 0 else None,
        "track_id": int(track_id) if track_id is not None else None,
        "is_target": bool(is_target),
        "bbox_x": round(x / width, 6) if x is not None and width else None,
        "bbox_y": round(y / height, 6) if y is not None and height else None,
        "bbox_width": round(w / width, 6) if w is not None and width else None,
        "bbox_height": round(h / height, 6) if h is not None and height else None,
        "bbox_area_ratio": round(area, 8) if area is not None else None,
        "detection_confidence": round(float(detection_confidence), 6)
        if detection_confidence is not None
        else None,
        "template_score": None,
        "reid_similarity": round(float(reid_similarity), 6)
        if reid_similarity is not None
        else None,
        "identity_state": state,
        "identity_risk": state == "identity_risk",
        "tracker_ok": state != "missing",
        "reasons": reasons,
        "source_backend": backend,
    }


def _boxmot_reid_row(
    *,
    frame_index: int,
    fps: float,
    track_id: int | None,
    embedding: np.ndarray,
    similarity: float,
    memory_updated: bool,
    embedding_model: str,
) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "time_seconds": round(frame_index / fps, 6) if fps > 0 else None,
        "track_id": int(track_id) if track_id is not None else None,
        "is_target": True,
        "embedding_model": embedding_model,
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


def _select_identity_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_yolo_model(detector_model: str) -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "YOLO person detection needs ultralytics. Install with: "
            'python -m pip install -e ".[mot]"'
        ) from exc
    return YOLO(detector_model)


def _create_boxmot_tracker(
    backend: str,
    *,
    reid_weights: str,
    device: str,
    half: bool,
) -> Any:
    try:
        from boxmot.trackers.tracker_zoo import create_tracker
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BoT-SORT/Deep OC-SORT tracking needs BoxMOT. Install with: "
            'python -m pip install -e ".[mot]"'
        ) from exc

    tracker_type = BOXMOT_TRACKER_TYPES[backend]
    return create_tracker(
        tracker_type,
        reid_weights=Path(reid_weights),
        device=device,
        half=half,
        per_class=False,
    )


def _run_yolo_person_detector(
    model: Any,
    frame: np.ndarray,
    *,
    device: str,
    confidence: float,
    iou: float,
    imgsz: int,
) -> np.ndarray:
    kwargs: dict[str, Any] = {
        "classes": [0],
        "conf": float(confidence),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    results = model.predict(frame, **kwargs)
    result = results[0] if isinstance(results, (list, tuple)) else results
    return result_to_person_detections(result)


def _parse_boxmot_tracks(
    tracks: Any,
    *,
    frame_index: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    array = _as_numpy(tracks)
    if array.size == 0:
        return []
    array = array.reshape((1, -1)) if array.ndim == 1 else array
    parsed: list[dict[str, Any]] = []
    for row in array:
        if row.shape[0] < 5:
            continue
        box = _xyxy_to_xywh(row[:4], width, height)
        if box is None:
            continue
        cls = int(round(float(row[6]))) if row.shape[0] > 6 else 0
        if cls != 0:
            continue
        confidence = float(row[5]) if row.shape[0] > 5 else None
        track_id = int(round(float(row[4])))
        parsed.append(
            {
                "frame_index": int(frame_index),
                "track_id": track_id,
                "box": box,
                "box_xyxy": _xywh_to_xyxy(box),
                "confidence": confidence,
            }
        )
    return parsed


def _score_target_candidate(
    candidate: dict[str, Any],
    *,
    start_index: int,
    prompt_box_xyxy: Sequence[float],
    prompt_points: Sequence[dict[str, float]],
    width: int,
    height: int,
    search_radius: int,
) -> float:
    distance = abs(int(candidate["frame_index"]) - int(start_index))
    temporal = max(0.0, 1.0 - (distance / max(1, search_radius)))
    confidence = float(candidate.get("confidence") or 0.0)
    iou = _bbox_iou_xyxy(candidate["box_xyxy"], prompt_box_xyxy)
    point = _prompt_point_score(prompt_points, candidate["box_xyxy"], width, height)
    return (0.7 * iou) + (0.2 * point) + (0.1 * confidence) + (0.1 * temporal)


def _select_target_candidate(
    candidates_by_frame: dict[int, list[dict[str, Any]]],
    *,
    start_index: int,
    prompt_box: tuple[int, int, int, int],
    prompt_points: Sequence[dict[str, float]],
    width: int,
    height: int,
    search_radius: int = 15,
) -> dict[str, Any] | None:
    prompt_box_xyxy = _xywh_to_xyxy(prompt_box)
    best: dict[str, Any] | None = None
    best_score = -1.0
    frame_indexes = [
        index
        for index in sorted(candidates_by_frame)
        if abs(index - start_index) <= search_radius
    ]
    if not frame_indexes:
        frame_indexes = sorted(candidates_by_frame)
        search_radius = max(search_radius, 1)
    for frame_index in frame_indexes:
        for candidate in candidates_by_frame[frame_index]:
            score = _score_target_candidate(
                candidate,
                start_index=start_index,
                prompt_box_xyxy=prompt_box_xyxy,
                prompt_points=prompt_points,
                width=width,
                height=height,
                search_radius=search_radius,
            )
            if score > best_score:
                best = candidate
                best_score = score
    return best


def _target_candidate_state(
    *,
    candidate: dict[str, Any],
    embedding: np.ndarray,
    memory: np.ndarray,
    previous_box: tuple[int, int, int, int] | None,
    previous_frame_index: int | None,
    width: int,
    height: int,
    reid_accept: float,
    reid_recover: float,
) -> tuple[str, list[str], float, bool]:
    similarity = _cosine_similarity(embedding, memory)
    reasons: list[str] = []
    if similarity < reid_recover:
        reasons.append("low_reid_similarity")
    if previous_box is not None and previous_frame_index is not None:
        box = candidate["box"]
        previous_area = max(1, previous_box[2] * previous_box[3])
        current_area = max(1, box[2] * box[3])
        area_ratio = current_area / previous_area
        if area_ratio > 2.2 or area_ratio < 0.45:
            reasons.append("sudden_area_jump")
        previous_center = (previous_box[0] + previous_box[2] / 2.0, previous_box[1] + previous_box[3] / 2.0)
        current_center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        frame_gap = max(1, int(candidate["frame_index"]) - int(previous_frame_index))
        max_jump = max(width, height) * 0.18 * frame_gap
        if float(np.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1])) > max_jump:
            reasons.append("sudden_motion_spike")
    state = "usable" if not reasons else "identity_risk"
    return state, reasons, similarity, state == "usable" and similarity >= reid_accept


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


def update_manifest_after_identity_failure(
    manifest_path: Path,
    *,
    error: str,
    backend: str = DEFAULT_IDENTITY_BACKEND,
) -> None:
    manifest = read_json(manifest_path)
    stage = manifest.setdefault("stages", {}).setdefault("detector_tracker", {})
    stage["status"] = "failed"
    stage["backend"] = backend
    stage["error"] = error
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def run_boxmot_identity_tracking(
    *,
    run_dir: Path,
    backend: str = DEFAULT_IDENTITY_BACKEND,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    reid_weights: str = DEFAULT_REID_WEIGHTS,
    device: str | None = None,
    half: bool = False,
    detector_confidence: float = 0.25,
    detector_iou: float = 0.7,
    detector_imgsz: int = 960,
    progress_callback: IdentityProgressCallback | None = None,
) -> dict[str, Any]:
    backend = canonical_identity_backend(backend)
    if backend not in BOXMOT_BACKENDS:
        raise ValueError(f"{backend} is not a BoxMOT identity backend")

    started_at = time.monotonic()
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = read_json(manifest_path)
    paths = manifest["paths"]
    source_segment = Path(str(paths["source_segment"]))
    prompt_path = Path(str(paths.get("person_prompt") or run_dir / "person_prompt.json"))
    track_seed_path = Path(str(paths.get("track_seed") or run_dir / "track_seed.json"))
    tracklets_path = Path(str(paths.get("tracklets") or run_dir / "tracklets.parquet"))
    reid_path = Path(str(paths.get("reid") or run_dir / "reid.parquet"))
    tracklets_jsonl_path = Path(str(paths.get("tracklets_jsonl") or run_dir / "tracklets.jsonl"))
    reid_jsonl_path = Path(str(paths.get("reid_jsonl") or run_dir / "reid.jsonl"))
    qc_metrics_path = Path(str(paths.get("qc_metrics") or run_dir / "qc_metrics.json"))

    try:
        setup = identity_setup_status(backend)
        if not setup["ready"]:
            raise RuntimeError(
                "Identity tracking backend is not configured: "
                + "; ".join(setup["reasons"])
                + f". Install with: {setup['install_command']}"
            )

        frames = load_video_frames(source_segment)
        prompt = read_json(prompt_path)
        seed = read_json(track_seed_path) if track_seed_path.exists() else {}
        prompt_frame = prompt.get("frame", {}) if isinstance(prompt.get("frame"), dict) else {}
        start_index = prompt_frame.get("frame_index")
        if start_index in (None, ""):
            start_index = frames.frame_count // 2
        start_index = max(0, min(int(start_index), frames.frame_count - 1))
        initial_box = prompt_initial_box(prompt, frames.width, frames.height)
        positive_points = _prompt_points(prompt)
        reid_accept = _read_reid_threshold(seed, "cosine_accept", 0.65)
        reid_recover = _read_reid_threshold(seed, "cosine_recover", 0.58)
        selected_device = _select_identity_device(device)

        detector = _load_yolo_model(detector_model)
        tracker = _create_boxmot_tracker(
            backend,
            reid_weights=reid_weights,
            device=selected_device,
            half=half,
        )

        candidates_by_frame: dict[int, list[dict[str, Any]]] = {}
        for frame_index, frame in enumerate(frames.frames):
            detections = _run_yolo_person_detector(
                detector,
                frame,
                device=selected_device,
                confidence=detector_confidence,
                iou=detector_iou,
                imgsz=detector_imgsz,
            )
            tracked = tracker.update(detections, frame)
            candidates_by_frame[frame_index] = _parse_boxmot_tracks(
                tracked,
                frame_index=frame_index,
                width=frames.width,
                height=frames.height,
            )
            if progress_callback:
                progress_callback(
                    build_identity_progress(
                        phase="detect_track",
                        processed_frames=frame_index + 1,
                        total_frames=frames.frame_count,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )

        target_candidate = _select_target_candidate(
            candidates_by_frame,
            start_index=start_index,
            prompt_box=initial_box,
            prompt_points=positive_points,
            width=frames.width,
            height=frames.height,
        )
        if target_candidate is None:
            raise RuntimeError(
                "BoxMOT did not produce a person track near the saved prompt. "
                "Retry with a clearer prompt, lower --detector-confidence, or "
                "--backend prompt_template_tracker_v1."
            )
        target_track_id = int(target_candidate["track_id"])

        target_candidates: dict[int, dict[str, Any]] = {}
        for frame_index, candidates in candidates_by_frame.items():
            matching = [candidate for candidate in candidates if candidate["track_id"] == target_track_id]
            if matching:
                target_candidates[frame_index] = max(
                    matching,
                    key=lambda candidate: float(candidate.get("confidence") or 0.0),
                )

        memory = _hist_embedding(_crop(frames.frames[target_candidate["frame_index"]], target_candidate["box"]))
        if not float(np.linalg.norm(memory)):
            memory = _hist_embedding(_crop(frames.frames[start_index], initial_box))

        track_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        reid_rows: list[dict[str, Any]] = []
        zero_embedding = np.zeros(64, dtype=np.float32)
        previous_usable_box: tuple[int, int, int, int] | None = None
        previous_usable_frame: int | None = None
        embedding_model = f"boxmot_internal:{reid_weights};artifact:hsv_histogram_8x8"

        for frame_index, frame in enumerate(frames.frames):
            candidates = candidates_by_frame.get(frame_index, [])
            target = target_candidates.get(frame_index)
            if target is None:
                target_row = _boxmot_track_row(
                    frame_index=frame_index,
                    fps=frames.fps,
                    width=frames.width,
                    height=frames.height,
                    box=None,
                    track_id=target_track_id,
                    is_target=True,
                    detection_confidence=None,
                    reid_similarity=0.0,
                    state="missing",
                    reasons=["target_track_missing"],
                    backend=backend,
                )
                target_rows.append(target_row)
                track_rows.append(target_row)
                reid_rows.append(
                    _boxmot_reid_row(
                        frame_index=frame_index,
                        fps=frames.fps,
                        track_id=target_track_id,
                        embedding=zero_embedding,
                        similarity=0.0,
                        memory_updated=False,
                        embedding_model=embedding_model,
                    )
                )
            else:
                embedding = _hist_embedding(_crop(frame, target["box"]))
                if frame_index == int(target_candidate["frame_index"]):
                    similarity = 1.0
                    state = "usable"
                    reasons: list[str] = []
                    memory_updated = True
                else:
                    state, reasons, similarity, memory_updated = _target_candidate_state(
                        candidate=target,
                        embedding=embedding,
                        memory=memory,
                        previous_box=previous_usable_box,
                        previous_frame_index=previous_usable_frame,
                        width=frames.width,
                        height=frames.height,
                        reid_accept=reid_accept,
                        reid_recover=reid_recover,
                    )
                if memory_updated:
                    blended = (memory * 0.9) + (embedding * 0.1)
                    norm = float(np.linalg.norm(blended))
                    memory = blended / norm if norm > 0 else blended
                if state == "usable":
                    previous_usable_box = target["box"]
                    previous_usable_frame = frame_index

                target_row = _boxmot_track_row(
                    frame_index=frame_index,
                    fps=frames.fps,
                    width=frames.width,
                    height=frames.height,
                    box=target["box"],
                    track_id=target_track_id,
                    is_target=True,
                    detection_confidence=target.get("confidence"),
                    reid_similarity=similarity,
                    state=state,
                    reasons=reasons,
                    backend=backend,
                )
                target_rows.append(target_row)
                track_rows.append(target_row)
                reid_rows.append(
                    _boxmot_reid_row(
                        frame_index=frame_index,
                        fps=frames.fps,
                        track_id=target_track_id,
                        embedding=embedding,
                        similarity=similarity,
                        memory_updated=memory_updated,
                        embedding_model=embedding_model,
                    )
                )

            for candidate in candidates:
                if int(candidate["track_id"]) == target_track_id:
                    continue
                track_rows.append(
                    _boxmot_track_row(
                        frame_index=frame_index,
                        fps=frames.fps,
                        width=frames.width,
                        height=frames.height,
                        box=candidate["box"],
                        track_id=int(candidate["track_id"]),
                        is_target=False,
                        detection_confidence=candidate.get("confidence"),
                        reid_similarity=None,
                        state="distractor",
                        reasons=[],
                        backend=backend,
                    )
                )

        track_rows.sort(
            key=lambda row: (
                int(row["frame_index"]),
                0 if row.get("is_target") else 1,
                int(row.get("track_id") or -1),
            )
        )
        metrics = summarize_identity_rows(target_rows)
        prompt_box = _box_payload(initial_box, frames.width, frames.height)
        metrics.update(
            {
                "prompt_frame_index": start_index,
                "initial_prompt_box": prompt_box,
                "target_track_id": target_track_id,
                "backend": backend,
                "detector_model": detector_model,
                "reid_weights": reid_weights,
                "boxmot_tracker": BOXMOT_BACKENDS[backend],
                "tracklet_rows": len(track_rows),
                "distractor_tracklet_rows": len(track_rows) - len(target_rows),
            }
        )

        _write_parquet(tracklets_path, track_rows)
        _write_parquet(reid_path, reid_rows)
        _write_jsonl(tracklets_jsonl_path, track_rows)
        _write_jsonl(reid_jsonl_path, reid_rows)

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
                "backend": backend,
                "target_track_id": target_track_id,
                "prompt_box": prompt_box,
                "detector": {
                    "model": detector_model,
                    "confidence": float(detector_confidence),
                    "iou": float(detector_iou),
                    "imgsz": int(detector_imgsz),
                },
                "tracker": {
                    "library": "boxmot",
                    "name": BOXMOT_BACKENDS[backend],
                    "backend": backend,
                    "reid_weights": reid_weights,
                    "device": selected_device,
                    "half": bool(half),
                },
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
            "backend": backend,
            "status": "complete",
            "frame_count": len(target_rows),
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
                    processed_frames=len(target_rows),
                    total_frames=frames.frame_count,
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        return result
    except Exception as exc:
        update_manifest_after_identity_failure(manifest_path, error=str(exc), backend=backend)
        raise


def run_identity_tracking(
    *,
    run_dir: Path,
    backend: str = DEFAULT_IDENTITY_BACKEND,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    reid_weights: str = DEFAULT_REID_WEIGHTS,
    device: str | None = None,
    half: bool = False,
    detector_confidence: float = 0.25,
    detector_iou: float = 0.7,
    detector_imgsz: int = 960,
    progress_callback: IdentityProgressCallback | None = None,
) -> dict[str, Any]:
    backend = canonical_identity_backend(backend)
    if backend == TEMPLATE_IDENTITY_BACKEND:
        return run_template_identity_tracking(
            run_dir=run_dir,
            progress_callback=progress_callback,
        )
    return run_boxmot_identity_tracking(
        run_dir=run_dir,
        backend=backend,
        detector_model=detector_model,
        reid_weights=reid_weights,
        device=device,
        half=half,
        detector_confidence=detector_confidence,
        detector_iou=detector_iou,
        detector_imgsz=detector_imgsz,
        progress_callback=progress_callback,
    )


def run_template_identity_tracking(
    *,
    run_dir: Path,
    progress_callback: IdentityProgressCallback | None = None,
) -> dict[str, Any]:
    backend = TEMPLATE_IDENTITY_BACKEND
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
        metrics["backend"] = backend

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
                "backend": backend,
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
            "backend": backend,
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
        update_manifest_after_identity_failure(manifest_path, error=str(exc), backend=backend)
        raise
