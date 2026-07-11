from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.pose_runner import (
    LANDMARK_NAMES,
    build_pose_progress,
    draw_skeleton,
    hard_mask_frame,
)
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam2_runner import inspect_video, write_json
from whodoirunlike.video_io import make_browser_playable_mp4s


MMPoseProgressCallback = Callable[[dict[str, Any]], None]
MMPOSE_DEVICE_ENV = "MMPOSE_DEVICE"
RTMW_RUNTIME_BACKEND_ENV = "RTMW_RUNTIME_BACKEND"
MMPOSE_USE_DETECTOR_ENV = "MMPOSE_USE_DETECTOR"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTMW_RUNTIME_BACKEND = "onnxruntime"
YOLOX_M_HUMANART_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)
YOLOX_TINY_HUMANART_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
)


@dataclass(frozen=True)
class MMPoseModelSpec:
    id: str
    label: str
    family: str
    detector_url: str
    pose_url: str
    detector_input_size: tuple[int, int]
    pose_input_size: tuple[int, int]
    input_size: str
    whole_ap: float | None = None
    notes: str | None = None


MMPOSE_MODEL_SPECS: dict[str, MMPoseModelSpec] = {
    "mmpose_rtmw_l_384": MMPoseModelSpec(
        id="mmpose_rtmw_l_384",
        label="RTMW-L 384x288",
        family="rtmw",
        detector_url=YOLOX_M_HUMANART_ONNX,
        pose_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
        ),
        detector_input_size=(640, 640),
        pose_input_size=(288, 384),
        input_size="384x288",
        whole_ap=70.1,
        notes="Best RTMW ONNX path available locally.",
    ),
    "mmpose_rtmw_l_256": MMPoseModelSpec(
        id="mmpose_rtmw_l_256",
        label="RTMW-L 256x192",
        family="rtmw",
        detector_url=YOLOX_M_HUMANART_ONNX,
        pose_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"
        ),
        detector_input_size=(640, 640),
        pose_input_size=(192, 256),
        input_size="256x192",
        whole_ap=66.0,
        notes="Faster RTMW-L whole-body model.",
    ),
    "mmpose_rtmw_m_256": MMPoseModelSpec(
        id="mmpose_rtmw_m_256",
        label="RTMW-M 256x192",
        family="rtmw",
        detector_url=YOLOX_TINY_HUMANART_ONNX,
        pose_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"
        ),
        detector_input_size=(416, 416),
        pose_input_size=(192, 256),
        input_size="256x192",
        whole_ap=58.2,
        notes="Lightweight RTMW whole-body model for quick checks.",
    ),
    "mmpose_rtmpose_l_384": MMPoseModelSpec(
        id="mmpose_rtmpose_l_384",
        label="RTMPose-L WholeBody 384x288",
        family="rtmpose",
        detector_url=YOLOX_M_HUMANART_ONNX,
        pose_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip"
        ),
        detector_input_size=(640, 640),
        pose_input_size=(288, 384),
        input_size="384x288",
        whole_ap=66.5,
        notes="RTMPose-L COCO+UBody ONNX comparison baseline.",
    ),
}

DEFAULT_MMPOSE_BACKEND = "mmpose_rtmw_l_384"
_MMPOSE_IMPORT_CHECKED = False
_MMPOSE_IMPORT_ERROR: str | None = None
_RTMLIB_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_RTMLIB_MODEL_CACHE_LOCK = threading.RLock()


@dataclass
class _LockedRTMLibModel:
    model: Any
    inference_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.inference_lock:
            return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

COCO_WHOLEBODY_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]
COCO_WHOLEBODY_NAMES.extend(f"face_{index}" for index in range(68))
COCO_WHOLEBODY_NAMES.extend(f"left_hand_{index}" for index in range(21))
COCO_WHOLEBODY_NAMES.extend(f"right_hand_{index}" for index in range(21))

MMPOSE_TO_MEDIAPIPE = {
    0: 0,
    1: 2,
    2: 5,
    3: 7,
    4: 8,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
    17: 31,
    19: 29,
    20: 32,
    22: 30,
    95: 21,
    99: 19,
    111: 17,
    116: 22,
    120: 20,
    132: 18,
}

MMPOSE_FALLBACKS_TO_MEDIAPIPE = {
    31: [17, 18],
    32: [20, 21],
}

MEDIAPIPE_SYNTHETIC_FROM_MMPOSE = {
    1: 1,
    3: 1,
    4: 2,
    6: 2,
    9: 0,
    10: 0,
    17: 9,
    19: 9,
    21: 9,
    18: 10,
    20: 10,
    22: 10,
}

MEDIAPIPE_CORE_INDICES = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}


def mmpose_backend_ids() -> set[str]:
    return set(MMPOSE_MODEL_SPECS)


def mmpose_model_spec(model_id: str | None = None) -> MMPoseModelSpec:
    model_id = str(model_id or DEFAULT_MMPOSE_BACKEND).strip().lower()
    if model_id not in MMPOSE_MODEL_SPECS:
        valid = ", ".join(sorted(MMPOSE_MODEL_SPECS))
        raise ValueError(f"RTMW/RTMPose backend must be one of: {valid}")
    return MMPOSE_MODEL_SPECS[model_id]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mmpose_setup_status(model_id: str | None = None) -> dict[str, Any]:
    global _MMPOSE_IMPORT_CHECKED, _MMPOSE_IMPORT_ERROR

    spec = mmpose_model_spec(model_id)
    dependencies = {
        "rtmlib": importlib.util.find_spec("rtmlib") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
    }
    missing = [name for name, available in dependencies.items() if not available]
    reasons: list[str] = []
    if missing:
        reasons.append(
            "Install optional RTMW dependencies in this venv: " + ", ".join(missing)
        )
    import_error: str | None = None
    if not missing:
        if not _MMPOSE_IMPORT_CHECKED:
            try:
                from rtmlib import Custom as _RTMLibCustom  # noqa: F401
            except Exception as exc:  # noqa: BLE001 - dependency health is surfaced to the UI.
                _MMPOSE_IMPORT_ERROR = str(exc)
            _MMPOSE_IMPORT_CHECKED = True
        import_error = _MMPOSE_IMPORT_ERROR
        if import_error:
            reasons.append(f"RTMLib import failed: {import_error}")

    device = os.environ.get(MMPOSE_DEVICE_ENV, "cpu").strip() or "cpu"
    runtime_backend = os.environ.get(RTMW_RUNTIME_BACKEND_ENV, DEFAULT_RTMW_RUNTIME_BACKEND).strip()
    runtime_backend = runtime_backend or DEFAULT_RTMW_RUNTIME_BACKEND
    use_detector = _env_bool(MMPOSE_USE_DETECTOR_ENV, True)
    return {
        "ready": not reasons,
        "reasons": reasons,
        "backend": spec.id,
        "label": spec.label,
        "family": spec.family,
        "runtime": "rtmlib",
        "runtime_backend": runtime_backend,
        "use_detector": use_detector,
        "detector_url": spec.detector_url,
        "pose_url": spec.pose_url,
        "detector_input_size": list(spec.detector_input_size),
        "pose_input_size": list(spec.pose_input_size),
        "input_size": spec.input_size,
        "whole_ap": spec.whole_ap,
        "notes": spec.notes,
        "device": device,
        "import_error": import_error,
        "env": {
            "device": MMPOSE_DEVICE_ENV,
            "runtime_backend": RTMW_RUNTIME_BACKEND_ENV,
            "use_detector": MMPOSE_USE_DETECTOR_ENV,
        },
        "dependencies": dependencies,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mask_frame(capture: cv2.VideoCapture | None, width: int, height: int) -> np.ndarray | None:
    if capture is None:
        return None
    ok, frame = capture.read()
    if not ok:
        return None
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (gray > 20).astype("uint8") * 255


def _mask_bbox(mask: np.ndarray | None) -> dict[str, float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    height, width = mask.shape[:2]
    x1 = float(xs.min()) / max(width, 1)
    y1 = float(ys.min()) / max(height, 1)
    x2 = float(xs.max() + 1) / max(width, 1)
    y2 = float(ys.max() + 1) / max(height, 1)
    return {
        "x": round(x1, 6),
        "y": round(y1, 6),
        "width": round(max(0.0, x2 - x1), 6),
        "height": round(max(0.0, y2 - y1), 6),
    }


def _bbox_iou(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if not a or not b:
        return 0.0
    ax1 = float(a["x"])
    ay1 = float(a["y"])
    ax2 = ax1 + float(a["width"])
    ay2 = ay1 + float(a["height"])
    bx1 = float(b["x"])
    by1 = float(b["y"])
    bx2 = bx1 + float(b["width"])
    by2 = by1 + float(b["height"])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _crop_bounds(mask: np.ndarray | None, width: int, height: int) -> tuple[int, int, int, int]:
    if mask is None:
        return 0, 0, width, height
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, width, height
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    pad = max(24, int(max(x2 - x1, y2 - y1) * 0.16))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def _crop_target_frame(
    frame: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = frame.shape[:2]
    masked = hard_mask_frame(frame, mask) if mask is not None else frame
    x1, y1, x2, y2 = _crop_bounds(mask, width, height)
    if x2 - x1 < 32 or y2 - y1 < 32:
        x1, y1, x2, y2 = 0, 0, width, height
    return masked[y1:y2, x1:x2], {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def _prediction_list(result: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = result.get("predictions") or []
    if not predictions:
        return []
    if isinstance(predictions[0], list):
        return [prediction for prediction in predictions[0] if isinstance(prediction, dict)]
    return [prediction for prediction in predictions if isinstance(prediction, dict)]


def rtmlib_arrays_to_predictions(keypoints: Any, scores: Any) -> list[dict[str, Any]]:
    keypoint_array = np.asarray(keypoints, dtype=np.float32)
    if keypoint_array.ndim == 2:
        keypoint_array = keypoint_array[None, :, :]
    if keypoint_array.ndim != 3 or keypoint_array.shape[2] < 2:
        return []

    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.ndim == 1:
        score_array = score_array[None, :]
    if score_array.ndim == 3 and score_array.shape[2] == 1:
        score_array = score_array[:, :, 0]
    if score_array.ndim != 2 or score_array.shape[0] < keypoint_array.shape[0]:
        score_array = np.ones(keypoint_array.shape[:2], dtype=np.float32)

    predictions: list[dict[str, Any]] = []
    for index in range(keypoint_array.shape[0]):
        source_scores = score_array[index]
        if source_scores.shape[0] < keypoint_array.shape[1]:
            padded = np.ones((keypoint_array.shape[1],), dtype=np.float32)
            padded[: source_scores.shape[0]] = source_scores
            source_scores = padded
        source_scores = np.nan_to_num(source_scores, nan=0.0, posinf=1.0, neginf=0.0)
        predictions.append(
            {
                "keypoints": keypoint_array[index, :, :2],
                "keypoint_scores": np.clip(source_scores[: keypoint_array.shape[1]], 0.0, 1.0),
            }
        )
    return predictions


def _prediction_arrays(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    keypoint_value = prediction.get("keypoints")
    keypoints = np.asarray([] if keypoint_value is None else keypoint_value, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    score_value = prediction.get("keypoint_scores")
    scores = np.asarray([] if score_value is None else score_value, dtype=np.float32)
    if scores.ndim != 1 or scores.shape[0] < keypoints.shape[0]:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)
    return keypoints[:, :2], scores[: keypoints.shape[0]]


def _frame_point(
    point: Sequence[float],
    *,
    crop: dict[str, int],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    x = float(point[0])
    y = float(point[1])
    if 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5:
        x *= max(crop["width"], 1)
        y *= max(crop["height"], 1)
    return (
        float(np.clip(crop["x"] + x, 0, max(frame_width - 1, 1))),
        float(np.clip(crop["y"] + y, 0, max(frame_height - 1, 1))),
    )


def _bbox_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    crop: dict[str, int],
    frame_width: int,
    frame_height: int,
    min_score: float = 0.05,
) -> dict[str, float] | None:
    valid = scores >= min_score
    if not valid.any():
        return None
    points = np.asarray(
        [
            _frame_point(point, crop=crop, frame_width=frame_width, frame_height=frame_height)
            for point in keypoints[valid]
        ],
        dtype=np.float32,
    )
    xs = points[:, 0] / max(frame_width, 1)
    ys = points[:, 1] / max(frame_height, 1)
    x1 = float(np.clip(xs.min(), 0.0, 1.0))
    y1 = float(np.clip(ys.min(), 0.0, 1.0))
    x2 = float(np.clip(xs.max(), 0.0, 1.0))
    y2 = float(np.clip(ys.max(), 0.0, 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x": round(x1, 6),
        "y": round(y1, 6),
        "width": round(x2 - x1, 6),
        "height": round(y2 - y1, 6),
    }


def select_mmpose_prediction(
    predictions: Sequence[dict[str, Any]],
    *,
    crop: dict[str, int],
    frame_width: int,
    frame_height: int,
    mask_bbox: dict[str, float] | None,
) -> tuple[int | None, dict[str, Any] | None, dict[str, float] | None, float]:
    best_index: int | None = None
    best_prediction: dict[str, Any] | None = None
    best_bbox: dict[str, float] | None = None
    best_iou = 0.0
    best_score = -1.0
    fallback: tuple[int, dict[str, Any], dict[str, float] | None, float] | None = None
    fallback_score = -1.0

    for index, prediction in enumerate(predictions):
        keypoints, scores = _prediction_arrays(prediction)
        if len(keypoints) == 0:
            continue
        bbox = _bbox_from_keypoints(
            keypoints,
            scores,
            crop=crop,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        visible = scores >= 0.05
        confidence = float(scores[visible].mean()) if visible.any() else 0.0
        mask_iou = _bbox_iou(bbox, mask_bbox)
        score = confidence + mask_iou * 4.0
        if score > fallback_score:
            fallback = (index, prediction, bbox, mask_iou)
            fallback_score = score
        if mask_bbox and mask_iou < 0.01:
            continue
        if score > best_score:
            best_index = index
            best_prediction = prediction
            best_bbox = bbox
            best_iou = mask_iou
            best_score = score

    if best_prediction is None and mask_bbox is None and fallback:
        best_index, best_prediction, best_bbox, best_iou = fallback
    return best_index, best_prediction, best_bbox, best_iou


def _source_name(index: int) -> str:
    if 0 <= index < len(COCO_WHOLEBODY_NAMES):
        return COCO_WHOLEBODY_NAMES[index]
    return f"mmpose_{index}"


def _raw_landmarks_from_prediction(
    prediction: dict[str, Any] | None,
    *,
    crop: dict[str, int],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    if prediction is None:
        return []
    keypoints, scores = _prediction_arrays(prediction)
    landmarks: list[dict[str, Any]] = []
    for index, point in enumerate(keypoints):
        x, y = _frame_point(point, crop=crop, frame_width=frame_width, frame_height=frame_height)
        score = float(scores[index]) if index < len(scores) else 0.0
        landmarks.append(
            {
                "index": index,
                "name": _source_name(index),
                "x": round(x / max(frame_width, 1), 6),
                "y": round(y / max(frame_height, 1), 6),
                "score": round(score, 6),
            }
        )
    return landmarks


def _empty_canonical_landmark(index: int, backend: str) -> dict[str, Any]:
    return {
        "index": index,
        "name": LANDMARK_NAMES[index],
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "visibility": 0.0,
        "presence": 0.0,
        "source": backend,
        "source_index": None,
        "source_name": None,
        "synthetic": False,
        "missing": True,
    }


def _copy_mmpose_landmark(
    target: list[dict[str, Any]],
    *,
    media_index: int,
    source: dict[str, Any],
    backend: str,
    synthetic: bool = False,
) -> None:
    score = round(float(source.get("score") or 0.0) * (0.35 if synthetic else 1.0), 6)
    target[media_index] = {
        "index": media_index,
        "name": LANDMARK_NAMES[media_index],
        "x": round(float(source.get("x") or 0.0), 6),
        "y": round(float(source.get("y") or 0.0), 6),
        "z": 0.0,
        "visibility": score,
        "presence": score,
        "source": backend,
        "source_index": int(source.get("index") or 0),
        "source_name": str(source.get("name") or ""),
        "synthetic": synthetic,
        "missing": False,
    }


def mmpose_row_to_pose_row(row: dict[str, Any], *, backend: str) -> dict[str, Any]:
    raw_landmarks = row.get("landmarks") or []
    canonical = [_empty_canonical_landmark(index, backend) for index in range(len(LANDMARK_NAMES))]

    for source_index, media_index in MMPOSE_TO_MEDIAPIPE.items():
        if source_index < len(raw_landmarks):
            _copy_mmpose_landmark(
                canonical,
                media_index=media_index,
                source=raw_landmarks[source_index],
                backend=backend,
            )

    for media_index, source_indices in MMPOSE_FALLBACKS_TO_MEDIAPIPE.items():
        if not canonical[media_index]["missing"]:
            continue
        for source_index in source_indices:
            if source_index < len(raw_landmarks):
                _copy_mmpose_landmark(
                    canonical,
                    media_index=media_index,
                    source=raw_landmarks[source_index],
                    backend=backend,
                )
                break

    for media_index, source_index in MEDIAPIPE_SYNTHETIC_FROM_MMPOSE.items():
        if source_index < len(raw_landmarks) and canonical[media_index]["missing"]:
            _copy_mmpose_landmark(
                canonical,
                media_index=media_index,
                source=raw_landmarks[source_index],
                backend=backend,
                synthetic=True,
            )

    core_scores = [
        float(landmark.get("visibility") or 0.0)
        for landmark in canonical
        if int(landmark["index"]) in MEDIAPIPE_CORE_INDICES and not landmark.get("missing")
    ]
    visibility_mean = float(np.mean(core_scores)) if core_scores else 0.0
    usable = bool(row.get("usable")) and visibility_mean >= 0.05
    return {
        "frame_index": int(row.get("frame_index") or 0),
        "time_seconds": row.get("time_seconds"),
        "frame_width": row.get("frame_width"),
        "frame_height": row.get("frame_height"),
        "detected": bool(row.get("detected")),
        "usable": usable,
        "drop_reason": None if usable else row.get("drop_reason") or "rtmw_missing_or_off_mask",
        "visibility_mean": round(visibility_mean, 6),
        "presence_mean": round(visibility_mean, 6),
        "bbox": row.get("bbox"),
        "selected_pose_index": row.get("selected_pose_index"),
        "candidate_count": row.get("candidate_count"),
        "mask_iou": row.get("mask_iou"),
        "source_pose_backend": backend,
        "landmarks": canonical,
        "world_landmarks": [],
    }


def _overlay_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    overlay = frame.copy()
    fill = np.zeros_like(frame)
    fill[:, :, 1] = 185
    mask_bool = mask > 0
    overlay = np.where(mask_bool[:, :, None], (frame * 0.75 + fill * 0.25).astype("uint8"), overlay)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (245, 242, 233), 2, lineType=cv2.LINE_AA)
    return overlay


def summarize_mmpose_pose(
    rows: Sequence[dict[str, Any]],
    *,
    input_video: Path,
    spec: MMPoseModelSpec,
    fps: float,
) -> dict[str, Any]:
    detected = [row for row in rows if row.get("detected")]
    usable = [row for row in rows if row.get("usable")]
    visibility_values = [float(row.get("visibility_mean") or 0.0) for row in detected]
    drop_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("drop_reason") or "usable")
        drop_counts[reason] = drop_counts.get(reason, 0) + 1
    frame_count = len(rows)
    return {
        "version": 1,
        "created_at": utc_now_iso(),
        "input_video": str(input_video),
        "backend": spec.id,
        "model": {
            "provider": "rtmlib",
            "runtime_backend": os.environ.get(
                RTMW_RUNTIME_BACKEND_ENV, DEFAULT_RTMW_RUNTIME_BACKEND
            ).strip()
            or DEFAULT_RTMW_RUNTIME_BACKEND,
            "use_detector": _env_bool(MMPOSE_USE_DETECTOR_ENV, True),
            "family": spec.family,
            "label": spec.label,
            "detector_url": spec.detector_url,
            "pose_url": spec.pose_url,
            "detector_input_size": list(spec.detector_input_size),
            "pose_input_size": list(spec.pose_input_size),
            "input_size": spec.input_size,
            "source_landmark_count": 133,
            "canonical_landmark_count": len(LANDMARK_NAMES),
        },
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "duration_seconds": round(frame_count / fps, 3) if fps else None,
        "quality": {
            "detected_frames": len(detected),
            "usable_frames": len(usable),
            "pose_hit_rate": round(len(detected) / frame_count, 4) if frame_count else 0.0,
            "usable_rate": round(len(usable) / frame_count, 4) if frame_count else 0.0,
            "visibility_mean": round(mean(visibility_values), 4) if visibility_values else 0.0,
            "drop_counts": drop_counts,
        },
    }


def update_manifest_after_mmpose_pose(
    manifest_path: Path,
    *,
    spec: MMPoseModelSpec,
    pose_landmarks_path: Path,
    raw_mmpose_landmarks_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    qa_overlay_key: str,
    features_path: Path,
    result: dict[str, Any],
) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    paths = manifest.setdefault("paths", {})
    paths["pose_landmarks"] = str(pose_landmarks_path)
    paths["mmpose_landmarks"] = str(raw_mmpose_landmarks_path)
    paths["skeleton_render"] = str(skeleton_render_path)
    paths[qa_overlay_key] = str(qa_overlay_path)
    paths["features"] = str(features_path)
    quality = result.get("quality", {})
    manifest.setdefault("stages", {}).setdefault("pose", {}).pop("error", None)
    run.update_stages(
        {
            "pose": {
                "status": "complete",
                "backend": spec.id,
                "recommended_tool": f"RTMLib {spec.label}",
                "output": str(pose_landmarks_path),
                "raw_output": str(raw_mmpose_landmarks_path),
                "summary": {
                    "pose_hit_rate": quality.get("pose_hit_rate"),
                    "usable_rate": quality.get("usable_rate"),
                    "visibility_mean": quality.get("visibility_mean"),
                },
            },
            "renders": {
                "status": "partial_complete",
                "skeleton_render": str(skeleton_render_path),
                qa_overlay_key: str(qa_overlay_path),
            },
            "features": {"status": "complete", "output": str(features_path)},
        },
        manifest,
    )


def update_manifest_after_mmpose_failure(manifest_path: Path, *, backend: str, error: str) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    run.update_stage(
        "pose",
        {"status": "failed", "backend": backend, "error": error},
        manifest,
    )


def build_rtmlib_model(
    spec: MMPoseModelSpec,
    *,
    device: str,
    runtime_backend: str,
    use_detector: bool | None = None,
    cache_enabled: bool = True,
) -> Any:
    detector_enabled = (
        _env_bool(MMPOSE_USE_DETECTOR_ENV, True)
        if use_detector is None
        else bool(use_detector)
    )
    key = (
        spec.id,
        spec.detector_url if detector_enabled else None,
        spec.pose_url,
        spec.detector_input_size,
        spec.pose_input_size,
        str(runtime_backend),
        str(device),
        detector_enabled,
    )
    with _RTMLIB_MODEL_CACHE_LOCK:
        if cache_enabled and key in _RTMLIB_MODEL_CACHE:
            return _RTMLIB_MODEL_CACHE[key]

        from rtmlib import Custom

        model = _LockedRTMLibModel(
            Custom(
                det_class="YOLOX" if detector_enabled else None,
                det=spec.detector_url if detector_enabled else None,
                det_input_size=spec.detector_input_size,
                pose_class="RTMPose",
                pose=spec.pose_url,
                pose_input_size=spec.pose_input_size,
                to_openpose=False,
                backend=runtime_backend,
                device=device,
            )
        )
        if cache_enabled:
            _RTMLIB_MODEL_CACHE[key] = model
        return model


def clear_rtmlib_model_cache() -> None:
    with _RTMLIB_MODEL_CACHE_LOCK:
        _RTMLIB_MODEL_CACHE.clear()


def process_mmpose_video(
    *,
    source_video: Path,
    mask_video: Path | None,
    pose_landmarks_path: Path,
    raw_mmpose_landmarks_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    features_path: Path,
    spec: MMPoseModelSpec,
    device: str,
    runtime_backend: str = DEFAULT_RTMW_RUNTIME_BACKEND,
    normalize_qa_overlay: bool = True,
    progress_callback: MMPoseProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="decoding",
                processed_frames=0,
                total_frames=0,
                elapsed_seconds=0.0,
            )
        )
    meta = inspect_video(source_video)
    fps = float(meta.get("fps") or 30.0)
    frame_count = int(meta.get("frame_count") or 0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not inspect RTMW input video: {source_video}")

    source_capture = cv2.VideoCapture(str(source_video))
    mask_capture = cv2.VideoCapture(str(mask_video)) if mask_video and mask_video.exists() else None
    if not source_capture.isOpened():
        raise ValueError(f"Could not open RTMW source video: {source_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_render_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_writer = cv2.VideoWriter(str(skeleton_render_path), fourcc, fps, (width, height), True)
    qa_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    qa_writer = cv2.VideoWriter(str(qa_overlay_path), fourcc, fps, (width, height), True)
    if not skeleton_writer.isOpened() or not qa_writer.isOpened():
        source_capture.release()
        if mask_capture:
            mask_capture.release()
        raise ValueError("Could not open RTMW output video writers")

    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="loading_rtmw_model",
                processed_frames=0,
                total_frames=frame_count,
                elapsed_seconds=0.0,
            )
        )

    model = build_rtmlib_model(spec, device=device, runtime_backend=runtime_backend)
    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="running_rtmw",
                processed_frames=0,
                total_frames=frame_count,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )

    raw_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    try:
        for frame_index in range(frame_count):
            ok, source_frame = source_capture.read()
            if not ok:
                break
            if source_frame.shape[:2] != (height, width):
                source_frame = cv2.resize(source_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = _mask_frame(mask_capture, width, height)
            crop_frame, crop = _crop_target_frame(source_frame, mask)
            mask_bbox = _mask_bbox(mask)
            if mask_bbox is None:
                predictions: list[dict[str, Any]] = []
            else:
                keypoints, scores = model(crop_frame)
                predictions = rtmlib_arrays_to_predictions(keypoints, scores)
            selected_index, selected, bbox, mask_iou = select_mmpose_prediction(
                predictions,
                crop=crop,
                frame_width=width,
                frame_height=height,
                mask_bbox=mask_bbox,
            )
            raw_row = {
                "frame_index": frame_index,
                "time_seconds": round(frame_index / fps, 3) if fps else None,
                "frame_width": width,
                "frame_height": height,
                "detected": bool(predictions),
                "usable": selected is not None,
                "drop_reason": (
                    None
                    if selected is not None
                    else "runner_mask_missing_or_empty"
                    if mask_bbox is None
                    else "rtmw_missing_or_off_mask"
                ),
                "selected_pose_index": selected_index,
                "candidate_count": len(predictions),
                "mask_iou": round(mask_iou, 4),
                "bbox": bbox,
                "crop": crop,
                "backend": spec.id,
                "landmarks": _raw_landmarks_from_prediction(
                    selected,
                    crop=crop,
                    frame_width=width,
                    frame_height=height,
                ),
            }
            raw_rows.append(raw_row)
            pose_row = mmpose_row_to_pose_row(raw_row, backend=spec.id)
            pose_rows.append(pose_row)
            skeleton_base = np.full((height, width, 3), (239, 235, 227), dtype=np.uint8)
            skeleton_writer.write(draw_skeleton(skeleton_base, pose_row))
            qa_writer.write(draw_skeleton(_overlay_mask(source_frame, mask), pose_row))

            if progress_callback and (frame_index == 0 or (frame_index + 1) % 5 == 0):
                progress_callback(
                    build_pose_progress(
                        phase="running_rtmw",
                        processed_frames=frame_index + 1,
                        total_frames=frame_count,
                        elapsed_seconds=time.monotonic() - started_at,
                        frame_index=frame_index,
                        detected=bool(raw_row["detected"]),
                        usable=bool(pose_row["usable"]),
                    )
                )
    finally:
        source_capture.release()
        if mask_capture:
            mask_capture.release()
        skeleton_writer.release()
        qa_writer.release()

    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="writing_outputs",
                processed_frames=len(pose_rows),
                total_frames=frame_count,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    _write_jsonl(raw_mmpose_landmarks_path, raw_rows)
    _write_jsonl(pose_landmarks_path, pose_rows)
    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="encoding",
                processed_frames=len(pose_rows),
                total_frames=frame_count,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    browser_outputs = [skeleton_render_path]
    if normalize_qa_overlay:
        browser_outputs.append(qa_overlay_path)
    make_browser_playable_mp4s(browser_outputs)
    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="postprocessing",
                processed_frames=len(pose_rows),
                total_frames=frame_count,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    summary = summarize_mmpose_pose(pose_rows, input_video=source_video, spec=spec, fps=fps)
    write_json(features_path, summary)
    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="completed",
                processed_frames=len(pose_rows),
                total_frames=frame_count,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    return summary


def run_mmpose_pose(
    *,
    run_dir: Path,
    model_id: str = DEFAULT_MMPOSE_BACKEND,
    device: str | None = None,
    isolate_qa_overlay: bool = False,
    progress_callback: MMPoseProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    spec = mmpose_model_spec(model_id)
    setup = mmpose_setup_status(spec.id)
    run = RunningClipRun(run_dir)
    manifest_path = run.manifest_path
    manifest = run.read_manifest()
    source_segment = run.artifact_path("source_segment", manifest)
    runner_mask = run.artifact_path("runner_mask", manifest)
    raw_mmpose_landmarks_path = run.artifact_path("mmpose_landmarks", manifest)
    pose_landmarks_path = run.artifact_path("pose_landmarks", manifest)
    skeleton_render_path = run.artifact_path("skeleton_render", manifest)
    qa_overlay_key = "pose_qa_overlay" if isolate_qa_overlay else "qa_overlay"
    qa_overlay_path = run.artifact_path(qa_overlay_key, manifest)
    features_path = run.artifact_path("features", manifest)

    if not setup["ready"]:
        error = "; ".join(setup["reasons"])
        update_manifest_after_mmpose_failure(manifest_path, backend=spec.id, error=error)
        return {
            "candidate_id": manifest.get("candidate_id"),
            "backend": spec.id,
            "status": "unavailable",
            "error": error,
            "setup": setup,
            "frame_count": 0,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }
    if not runner_mask.exists():
        raise FileNotFoundError("Runner mask not found; run SAM 3.1 before RTMW/RTMPose")

    try:
        result = process_mmpose_video(
            source_video=source_segment,
            mask_video=runner_mask,
            pose_landmarks_path=pose_landmarks_path,
            raw_mmpose_landmarks_path=raw_mmpose_landmarks_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            features_path=features_path,
            spec=spec,
            device=device or str(setup["device"]),
            runtime_backend=str(setup.get("runtime_backend") or DEFAULT_RTMW_RUNTIME_BACKEND),
            normalize_qa_overlay=not isolate_qa_overlay,
            progress_callback=progress_callback,
        )
        update_manifest_after_mmpose_pose(
            manifest_path,
            spec=spec,
            pose_landmarks_path=pose_landmarks_path,
            raw_mmpose_landmarks_path=raw_mmpose_landmarks_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            qa_overlay_key=qa_overlay_key,
            features_path=features_path,
            result=result,
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "backend": spec.id,
            "status": "complete",
            "frame_count": result.get("frame_count", 0),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "pose_landmarks_path": str(pose_landmarks_path),
            "mmpose_landmarks_path": str(raw_mmpose_landmarks_path),
            "skeleton_render_path": str(skeleton_render_path),
            "qa_overlay_path": str(qa_overlay_path),
            "features_path": str(features_path),
            "quality": result.get("quality", {}),
            "model": result.get("model", {}),
        }
    except Exception as exc:
        update_manifest_after_mmpose_failure(manifest_path, backend=spec.id, error=str(exc))
        raise
