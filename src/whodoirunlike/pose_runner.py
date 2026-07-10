from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam2_runner import inspect_video, read_json, write_json
from whodoirunlike.video_io import make_browser_playable_mp4s
from whodoirunlike.video_eval import FOOT_LANDMARKS, POSE_MODEL_URLS, ensure_pose_model


PoseProgressCallback = Callable[[dict[str, Any]], None]

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]

CORE_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28]


@dataclass(frozen=True)
class PoseCandidate:
    index: int
    landmarks: Sequence[Any]
    world_landmarks: Sequence[Any] | None
    visibility_mean: float
    bbox: dict[str, float] | None
    score: float


def build_pose_progress(
    *,
    phase: str,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
    frame_index: int | None = None,
    detected: bool | None = None,
    usable: bool | None = None,
) -> dict[str, Any]:
    total_frames = max(0, int(total_frames))
    processed_frames = max(0, min(int(processed_frames), total_frames or int(processed_frames)))
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    percent = float(processed_frames / total_frames) if total_frames else 0.0
    if processed_frames > 0 and total_frames > processed_frames and elapsed_seconds > 0:
        eta_seconds: float | None = (elapsed_seconds / processed_frames) * (
            total_frames - processed_frames
        )
    elif total_frames and processed_frames >= total_frames:
        eta_seconds = 0.0
    else:
        eta_seconds = None

    payload: dict[str, Any] = {
        "phase": phase,
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "percent": round(percent, 4),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
    }
    if frame_index is not None:
        payload["frame_index"] = int(frame_index)
    if detected is not None:
        payload["detected"] = bool(detected)
    if usable is not None:
        payload["usable"] = bool(usable)
    return payload


def _bbox_iou(box_a: dict[str, float] | None, box_b: dict[str, float] | None) -> float:
    if not box_a or not box_b:
        return 0.0
    ax1 = float(box_a["x"])
    ay1 = float(box_a["y"])
    ax2 = ax1 + float(box_a["width"])
    ay2 = ay1 + float(box_a["height"])
    bx1 = float(box_b["x"])
    by1 = float(box_b["y"])
    bx2 = bx1 + float(box_b["width"])
    by2 = by1 + float(box_b["height"])
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center(box: dict[str, float] | None) -> tuple[float, float] | None:
    if not box:
        return None
    return (float(box["x"]) + float(box["width"]) / 2, float(box["y"]) + float(box["height"]) / 2)


def _point_score(point: tuple[float, float] | None, box: dict[str, float] | None) -> float:
    if point is None or box is None:
        return 0.0
    center = _bbox_center(box)
    if center is None:
        return 0.0
    distance = math.hypot(center[0] - point[0], center[1] - point[1])
    return max(0.0, 1.0 - distance / math.sqrt(2))


def _normalized_prompt_box(prompt: dict[str, Any]) -> dict[str, float] | None:
    box = prompt.get("selection", {}).get("box")
    if not isinstance(box, dict):
        return None
    return {
        "x": max(0.0, min(float(box.get("x", 0.0)), 1.0)),
        "y": max(0.0, min(float(box.get("y", 0.0)), 1.0)),
        "width": max(0.0, min(float(box.get("width", 0.0)), 1.0)),
        "height": max(0.0, min(float(box.get("height", 0.0)), 1.0)),
    }


def _normalized_prompt_anchor(prompt: dict[str, Any]) -> tuple[float, float] | None:
    positive = prompt.get("selection", {}).get("positive_points") or []
    if not positive:
        return _bbox_center(_normalized_prompt_box(prompt))
    xs = [max(0.0, min(float(point.get("x", 0.0)), 1.0)) for point in positive]
    ys = [max(0.0, min(float(point.get("y", 0.0)), 1.0)) for point in positive]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _landmark_visibility(landmark: Any) -> float:
    return float(getattr(landmark, "visibility", 1.0) or 0.0)


def _landmark_presence(landmark: Any) -> float:
    return float(getattr(landmark, "presence", 1.0) or 0.0)


def _landmark_bbox(landmarks: Sequence[Any], min_visibility: float = 0.2) -> dict[str, float] | None:
    visible = [lm for lm in landmarks if _landmark_visibility(lm) >= min_visibility]
    if not visible:
        return None
    xs = [max(0.0, min(float(lm.x), 1.0)) for lm in visible]
    ys = [max(0.0, min(float(lm.y), 1.0)) for lm in visible]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x": round(x1, 6),
        "y": round(y1, 6),
        "width": round(x2 - x1, 6),
        "height": round(y2 - y1, 6),
    }


def _pose_candidates(result: Any) -> list[PoseCandidate]:
    candidates: list[PoseCandidate] = []
    pose_landmarks = result.pose_landmarks or []
    world_landmarks = result.pose_world_landmarks or []
    for index, landmarks in enumerate(pose_landmarks):
        visibility_values = [_landmark_visibility(lm) for lm in landmarks]
        visibility_mean = mean(visibility_values) if visibility_values else 0.0
        bbox = _landmark_bbox(landmarks)
        core_visibility = mean(_landmark_visibility(landmarks[i]) for i in CORE_LANDMARKS)
        bbox_area = (bbox["width"] * bbox["height"]) if bbox else 0.0
        score = (visibility_mean * 0.45) + (core_visibility * 0.45) + min(1.0, bbox_area * 4.0) * 0.1
        candidates.append(
            PoseCandidate(
                index=index,
                landmarks=landmarks,
                world_landmarks=world_landmarks[index] if index < len(world_landmarks) else None,
                visibility_mean=visibility_mean,
                bbox=bbox,
                score=score,
            )
        )
    return candidates


def choose_pose_candidate(
    candidates: Sequence[PoseCandidate],
    *,
    prompt_box: dict[str, float] | None = None,
    prompt_anchor: tuple[float, float] | None = None,
    previous_bbox: dict[str, float] | None = None,
    mask_box: dict[str, float] | None = None,
    min_mask_iou: float = 0.02,
) -> PoseCandidate | None:
    if not candidates:
        return None
    best: PoseCandidate | None = None
    best_score = -1.0
    for candidate in candidates:
        mask_overlap = _bbox_iou(candidate.bbox, mask_box)
        if mask_box and mask_overlap < min_mask_iou:
            continue
        continuity = _bbox_iou(candidate.bbox, previous_bbox)
        prompt_overlap = _bbox_iou(candidate.bbox, prompt_box)
        anchor = _point_score(prompt_anchor, candidate.bbox)
        total = (
            candidate.score
            + continuity * 1.1
            + prompt_overlap * (1.15 if previous_bbox is None else 0.25)
            + anchor * (0.55 if previous_bbox is None else 0.15)
            + mask_overlap * 4.0
        )
        if total > best_score:
            best = candidate
            best_score = total
    return best


def _landmarks_payload(landmarks: Sequence[Any]) -> list[dict[str, float | str | int]]:
    return [
        {
            "index": index,
            "name": LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else str(index),
            "x": round(float(lm.x), 6),
            "y": round(float(lm.y), 6),
            "z": round(float(getattr(lm, "z", 0.0)), 6),
            "visibility": round(_landmark_visibility(lm), 6),
            "presence": round(_landmark_presence(lm), 6),
        }
        for index, lm in enumerate(landmarks)
    ]


def _world_landmarks_payload(landmarks: Sequence[Any] | None) -> list[dict[str, float | str | int]]:
    if not landmarks:
        return []
    return [
        {
            "index": index,
            "name": LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else str(index),
            "x": round(float(lm.x), 6),
            "y": round(float(lm.y), 6),
            "z": round(float(getattr(lm, "z", 0.0)), 6),
            "visibility": round(_landmark_visibility(lm), 6),
        }
        for index, lm in enumerate(landmarks)
    ]


def _drop_reason(candidate: PoseCandidate | None) -> str | None:
    if candidate is None:
        return "pose_missing"
    if candidate.visibility_mean < 0.35:
        return "low_visibility"
    landmarks = candidate.landmarks
    core_visibility = mean(_landmark_visibility(landmarks[i]) for i in CORE_LANDMARKS)
    if core_visibility < 0.35:
        return "core_landmarks_low_visibility"
    foot_visibility = mean(_landmark_visibility(landmarks[i]) for i in FOOT_LANDMARKS)
    if foot_visibility < 0.25:
        return "feet_low_visibility"
    if candidate.bbox:
        height = candidate.bbox["height"]
        width = candidate.bbox["width"]
        if height < 0.12:
            return "runner_too_small"
        if width > 0.9 or height > 0.96:
            return "runner_cropped_or_too_large"
    return None


def pose_row(
    *,
    frame_index: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    candidate: PoseCandidate | None,
    candidate_count: int,
) -> dict[str, Any]:
    reason = _drop_reason(candidate)
    return {
        "frame_index": frame_index,
        "time_seconds": round(frame_index / fps, 3) if fps else None,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "detected": candidate is not None,
        "usable": reason is None,
        "drop_reason": reason,
        "selected_pose_index": candidate.index if candidate else None,
        "candidate_count": candidate_count,
        "visibility_mean": round(candidate.visibility_mean, 6) if candidate else 0.0,
        "bbox": candidate.bbox if candidate else None,
        "landmarks": _landmarks_payload(candidate.landmarks) if candidate else [],
        "world_landmarks": _world_landmarks_payload(candidate.world_landmarks) if candidate else [],
    }


def _point_from_landmarks(landmarks: Sequence[dict[str, Any]], index: int, width: int, height: int) -> tuple[int, int] | None:
    if index >= len(landmarks):
        return None
    landmark = landmarks[index]
    if float(landmark.get("visibility", 0.0)) < 0.2:
        return None
    x = int(round(float(landmark["x"]) * max(width - 1, 1)))
    y = int(round(float(landmark["y"]) * max(height - 1, 1)))
    return x, y


def draw_skeleton(frame: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    landmarks = row.get("landmarks") or []
    if not landmarks:
        return output
    for start, end in POSE_CONNECTIONS:
        p1 = _point_from_landmarks(landmarks, start, width, height)
        p2 = _point_from_landmarks(landmarks, end, width, height)
        if p1 and p2:
            cv2.line(output, p1, p2, (28, 37, 49), 3, lineType=cv2.LINE_AA)
            cv2.line(output, p1, p2, (244, 173, 119), 1, lineType=cv2.LINE_AA)
    for index, landmark in enumerate(landmarks):
        point = _point_from_landmarks(landmarks, index, width, height)
        if not point:
            continue
        radius = 5 if index in CORE_LANDMARKS else 3
        cv2.circle(output, point, radius, (247, 247, 241), -1, lineType=cv2.LINE_AA)
        cv2.circle(output, point, radius, (28, 37, 49), 1, lineType=cv2.LINE_AA)
    return output


def _mask_frame(mask_capture: cv2.VideoCapture | None, width: int, height: int) -> np.ndarray | None:
    if mask_capture is None:
        return None
    ok, mask_frame = mask_capture.read()
    if not ok:
        return None
    if mask_frame.shape[:2] != (height, width):
        mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    return (gray > 20).astype("uint8") * 255


def _mask_bbox_normalized(mask: np.ndarray | None) -> dict[str, float] | None:
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


def hard_mask_frame(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    mask_bool = mask > 0
    output = np.zeros_like(frame)
    output[mask_bool] = frame[mask_bool]
    return output


def _overlay_mask(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    overlay = frame.copy()
    fill = np.zeros_like(frame)
    fill[:, :, 1] = 200
    mask_bool = mask > 0
    overlay = np.where(mask_bool[:, :, None], (frame * 0.72 + fill * 0.28).astype("uint8"), overlay)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (247, 247, 241), 2, lineType=cv2.LINE_AA)
    return overlay


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _joint_point(row: dict[str, Any], index: int) -> tuple[float, float] | None:
    landmarks = row.get("landmarks") or []
    if index >= len(landmarks):
        return None
    landmark = landmarks[index]
    if float(landmark.get("visibility", 0.0)) < 0.25:
        return None
    return float(landmark["x"]), float(landmark["y"])


def _midpoint(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _torso_lean(row: dict[str, Any]) -> float | None:
    shoulders = _midpoint(_joint_point(row, 11), _joint_point(row, 12))
    hips = _midpoint(_joint_point(row, 23), _joint_point(row, 24))
    if shoulders is None or hips is None:
        return None
    dx = shoulders[0] - hips[0]
    dy = hips[1] - shoulders[1]
    if abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dx, dy))


def _range(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def summarize_pose(rows: Sequence[dict[str, Any]], *, input_video: Path, model_variant: str, fps: float) -> dict[str, Any]:
    detected = [row for row in rows if row.get("detected")]
    usable = [row for row in rows if row.get("usable")]
    visibility_values = [float(row.get("visibility_mean") or 0.0) for row in detected]
    left_ankle_x = [p[0] for row in usable if (p := _joint_point(row, 27))]
    right_ankle_x = [p[0] for row in usable if (p := _joint_point(row, 28))]
    ankle_y = [
        p[1]
        for row in usable
        for index in (27, 28)
        if (p := _joint_point(row, index)) is not None
    ]
    torso_lean_values = [value for row in usable if (value := _torso_lean(row)) is not None]

    drop_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("drop_reason") or "usable")
        drop_counts[reason] = drop_counts.get(reason, 0) + 1

    frame_count = len(rows)
    return {
        "version": 1,
        "created_at": utc_now_iso(),
        "input_video": str(input_video),
        "model": {
            "provider": "mediapipe",
            "task": "pose_landmarker",
            "variant": model_variant,
            "landmark_count": len(LANDMARK_NAMES),
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
        "explainability_metrics": {
            "left_ankle_x_range": round(_range(left_ankle_x), 6) if _range(left_ankle_x) else None,
            "right_ankle_x_range": round(_range(right_ankle_x), 6) if _range(right_ankle_x) else None,
            "ankle_y_range": round(_range(ankle_y), 6) if _range(ankle_y) else None,
            "torso_lean_mean_deg": round(mean(torso_lean_values), 3) if torso_lean_values else None,
            "torso_lean_range_deg": round(_range(torso_lean_values), 3)
            if _range(torso_lean_values)
            else None,
        },
    }


def _resolve_pose_input(paths: dict[str, Any], input_mode: str) -> Path:
    source_segment = Path(str(paths["source_segment"]))
    masked_runner = Path(str(paths.get("masked_runner") or ""))
    if input_mode == "source":
        return source_segment
    if input_mode == "masked":
        if not masked_runner.exists():
            raise FileNotFoundError(f"Masked runner video not found: {masked_runner}")
        return masked_runner
    if input_mode != "auto":
        raise ValueError("input_mode must be one of: auto, source, masked")
    return masked_runner if masked_runner.exists() else source_segment


def process_pose_video(
    *,
    input_video: Path,
    source_video: Path,
    mask_video: Path | None,
    prompt: dict[str, Any],
    pose_landmarks_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    features_path: Path,
    model_path: Path,
    model_variant: str,
    progress_callback: PoseProgressCallback | None = None,
) -> dict[str, Any]:
    meta = inspect_video(input_video)
    fps = float(meta.get("fps") or 30.0)
    frame_count = int(meta.get("frame_count") or 0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not inspect pose input video: {input_video}")

    input_capture = cv2.VideoCapture(str(input_video))
    source_capture = cv2.VideoCapture(str(source_video))
    mask_capture = cv2.VideoCapture(str(mask_video)) if mask_video and mask_video.exists() else None
    if not input_capture.isOpened():
        raise ValueError(f"Could not open pose input video: {input_video}")
    if not source_capture.isOpened():
        input_capture.release()
        raise ValueError(f"Could not open source video for pose QA: {source_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_render_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_writer = cv2.VideoWriter(str(skeleton_render_path), fourcc, fps, (width, height), True)
    qa_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    qa_writer = cv2.VideoWriter(str(qa_overlay_path), fourcc, fps, (width, height), True)
    if not skeleton_writer.isOpened() or not qa_writer.isOpened():
        input_capture.release()
        source_capture.release()
        if mask_capture:
            mask_capture.release()
        raise ValueError("Could not open pose output video writers")

    mp_image = mp.Image
    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )

    start = time.monotonic()
    prompt_box = _normalized_prompt_box(prompt)
    prompt_anchor = _normalized_prompt_anchor(prompt)
    previous_bbox: dict[str, float] | None = None
    rows: list[dict[str, Any]] = []

    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="loading_model",
                processed_frames=0,
                total_frames=frame_count,
                elapsed_seconds=0.0,
            )
        )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_index in range(frame_count):
            ok_input, input_frame = input_capture.read()
            ok_source, source_frame = source_capture.read()
            if not ok_input or not ok_source:
                break
            if input_frame.shape[:2] != (height, width):
                input_frame = cv2.resize(input_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if source_frame.shape[:2] != (height, width):
                source_frame = cv2.resize(source_frame, (width, height), interpolation=cv2.INTER_LINEAR)

            mask = _mask_frame(mask_capture, width, height)
            mask_box = _mask_bbox_normalized(mask)
            pose_input_frame = hard_mask_frame(source_frame, mask) if mask is not None else input_frame
            frame_rgb = cv2.cvtColor(pose_input_frame, cv2.COLOR_BGR2RGB)
            image = mp_image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = math.floor(frame_index / fps * 1000)
            result = landmarker.detect_for_video(image, timestamp_ms)
            candidates = _pose_candidates(result)
            selected = choose_pose_candidate(
                candidates,
                prompt_box=prompt_box,
                prompt_anchor=prompt_anchor,
                previous_bbox=previous_bbox,
                mask_box=mask_box,
            )
            if selected and selected.bbox:
                previous_bbox = selected.bbox

            row = pose_row(
                frame_index=frame_index,
                fps=fps,
                frame_width=width,
                frame_height=height,
                candidate=selected,
                candidate_count=len(candidates),
            )
            rows.append(row)

            skeleton_base = np.full((height, width, 3), (239, 235, 227), dtype=np.uint8)
            skeleton_writer.write(draw_skeleton(skeleton_base, row))
            qa_frame = draw_skeleton(_overlay_mask(source_frame, mask), row)
            qa_writer.write(qa_frame)

            if progress_callback and (frame_index == 0 or (frame_index + 1) % 10 == 0):
                progress_callback(
                    build_pose_progress(
                        phase="detecting_pose",
                        processed_frames=frame_index + 1,
                        total_frames=frame_count,
                        elapsed_seconds=time.monotonic() - start,
                        frame_index=frame_index,
                        detected=bool(row["detected"]),
                        usable=bool(row["usable"]),
                    )
                )

    input_capture.release()
    source_capture.release()
    if mask_capture:
        mask_capture.release()
    skeleton_writer.release()
    qa_writer.release()
    make_browser_playable_mp4s([skeleton_render_path, qa_overlay_path])

    summary = summarize_pose(rows, input_video=input_video, model_variant=model_variant, fps=fps)
    _write_jsonl(pose_landmarks_path, rows)
    write_json(features_path, summary)

    if progress_callback:
        progress_callback(
            build_pose_progress(
                phase="writing_outputs",
                processed_frames=len(rows),
                total_frames=frame_count,
                elapsed_seconds=time.monotonic() - start,
            )
        )
    return summary


def update_manifest_after_pose(
    manifest_path: Path,
    *,
    pose_landmarks_path: Path,
    skeleton_render_path: Path,
    features_path: Path,
    result: dict[str, Any],
) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    run.update_stages(
        {
            "pose": {
                "status": "complete",
                "output": str(pose_landmarks_path),
                "summary": {
                    "pose_hit_rate": result.get("quality", {}).get("pose_hit_rate"),
                    "usable_rate": result.get("quality", {}).get("usable_rate"),
                    "visibility_mean": result.get("quality", {}).get("visibility_mean"),
                },
            },
            "renders": {
                "status": "partial_complete",
                "skeleton_render": str(skeleton_render_path),
            },
            "features": {"status": "complete", "output": str(features_path)},
        },
        manifest,
    )


def update_manifest_after_pose_failure(manifest_path: Path, error: str) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    run.update_stage("pose", {"status": "failed", "error": error}, manifest)


def run_pose_landmarks(
    *,
    run_dir: Path,
    model_dir: Path | None = None,
    model_variant: str = "heavy",
    input_mode: str = "auto",
    progress_callback: PoseProgressCallback | None = None,
) -> dict[str, Any]:
    if model_variant not in POSE_MODEL_URLS:
        valid = ", ".join(sorted(POSE_MODEL_URLS))
        raise ValueError(f"model_variant must be one of: {valid}")

    run = RunningClipRun(run_dir)
    manifest_path = run.manifest_path
    manifest = run.read_manifest()
    resolved_manifest = run.ensure_paths(
        manifest,
        keys=(
            "source_segment",
            "masked_runner",
            "runner_mask",
            "person_prompt",
            "pose_landmarks",
            "skeleton_render",
            "qa_overlay",
            "features",
        ),
    )
    paths = resolved_manifest["paths"]
    source_segment = Path(str(paths["source_segment"]))
    input_video = _resolve_pose_input(paths, input_mode)
    mask_video = Path(str(paths.get("runner_mask") or "")) if paths.get("runner_mask") else None
    prompt_path = Path(str(paths["person_prompt"]))
    prompt = read_json(prompt_path)
    pose_landmarks_path = Path(str(paths["pose_landmarks"]))
    skeleton_render_path = Path(str(paths["skeleton_render"]))
    qa_overlay_path = Path(str(paths["qa_overlay"]))
    features_path = Path(str(paths["features"]))
    model_path = ensure_pose_model(model_dir or Path("models/mediapipe"), model_variant)

    started_at = time.monotonic()
    try:
        summary = process_pose_video(
            input_video=input_video,
            source_video=source_segment,
            mask_video=mask_video,
            prompt=prompt,
            pose_landmarks_path=pose_landmarks_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            features_path=features_path,
            model_path=model_path,
            model_variant=model_variant,
            progress_callback=progress_callback,
        )
        update_manifest_after_pose(
            manifest_path,
            pose_landmarks_path=pose_landmarks_path,
            skeleton_render_path=skeleton_render_path,
            features_path=features_path,
            result=summary,
        )
        return {
            "backend": "mediapipe_pose",
            "model_variant": model_variant,
            "input_mode": input_mode,
            "input_video": str(input_video),
            "frame_count": summary.get("frame_count", 0),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "pose_landmarks_path": str(pose_landmarks_path),
            "skeleton_render_path": str(skeleton_render_path),
            "features_path": str(features_path),
            "qa_overlay_path": str(qa_overlay_path),
            "quality": summary.get("quality", {}),
        }
    except Exception as exc:
        update_manifest_after_pose_failure(manifest_path, str(exc))
        raise
