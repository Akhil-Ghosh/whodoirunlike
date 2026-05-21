from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.pose_runner import LANDMARK_NAMES, hard_mask_frame
from whodoirunlike.sam2_runner import inspect_video, read_json, write_json
from whodoirunlike.video_io import make_browser_playable_mp4s


OPENPOSE_BIN_ENV = "OPENPOSE_BIN"
OPENPOSE_MODEL_FOLDER_ENV = "OPENPOSE_MODEL_FOLDER"
OpenPoseProgressCallback = Callable[[dict[str, Any]], None]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENPOSE_ROOT = REPO_ROOT / "models/openpose/openpose-with-caffe-for-MacM1"
DEFAULT_OPENPOSE_BIN = DEFAULT_OPENPOSE_ROOT / "build/examples/openpose/openpose.bin"
DEFAULT_OPENPOSE_MODEL_FOLDER = DEFAULT_OPENPOSE_ROOT / "models"
DEFAULT_BODY25_MODEL = DEFAULT_OPENPOSE_MODEL_FOLDER / "pose/body_25/pose_iter_584000.caffemodel"

BODY25_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "mid_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

BODY25_CONNECTIONS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (11, 22),
    (22, 23),
    (11, 24),
    (14, 19),
    (19, 20),
    (14, 21),
]

BODY25_TO_MEDIAPIPE = {
    0: 0,
    15: 5,
    16: 2,
    17: 8,
    18: 7,
    2: 12,
    3: 14,
    4: 16,
    5: 11,
    6: 13,
    7: 15,
    9: 24,
    10: 26,
    11: 28,
    12: 23,
    13: 25,
    14: 27,
    19: 31,
    21: 29,
    22: 32,
    24: 30,
}

MEDIAPIPE_SYNTHETIC_FROM_BODY25 = {
    1: 16,  # left_eye_inner
    3: 16,  # left_eye_outer
    4: 15,  # right_eye_inner
    6: 15,  # right_eye_outer
    9: 0,  # mouth_left
    10: 0,  # mouth_right
    17: 7,  # left_pinky
    19: 7,  # left_index
    21: 7,  # left_thumb
    18: 4,  # right_pinky
    20: 4,  # right_index
    22: 4,  # right_thumb
}

MEDIAPIPE_CORE_INDICES = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}


def build_openpose_progress(
    *,
    phase: str,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
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
    return {
        "phase": phase,
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "percent": round(percent, 4),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
    }


def openpose_setup_status(
    *,
    binary_path: str | Path | None = None,
    model_folder: str | Path | None = None,
) -> dict[str, Any]:
    binary_value = str(binary_path or os.environ.get(OPENPOSE_BIN_ENV, "")).strip()
    local_binary = str(DEFAULT_OPENPOSE_BIN) if DEFAULT_OPENPOSE_BIN.exists() else None
    resolved_binary = (
        binary_value
        or local_binary
        or shutil.which("openpose")
        or shutil.which("openpose.bin")
    )
    reasons: list[str] = []
    if not resolved_binary:
        reasons.append(
            f"OpenPose binary not found. Set {OPENPOSE_BIN_ENV} to an openpose/openpose.bin executable."
        )
    elif not Path(resolved_binary).expanduser().exists() and not shutil.which(resolved_binary):
        reasons.append(f"{OPENPOSE_BIN_ENV} does not exist: {resolved_binary}")

    model_value = str(model_folder or os.environ.get(OPENPOSE_MODEL_FOLDER_ENV, "")).strip()
    local_model_folder = (
        str(DEFAULT_OPENPOSE_MODEL_FOLDER) if DEFAULT_OPENPOSE_MODEL_FOLDER.exists() else ""
    )
    resolved_model_folder = model_value or local_model_folder
    if resolved_model_folder and not Path(resolved_model_folder).expanduser().exists():
        reasons.append(f"{OPENPOSE_MODEL_FOLDER_ENV} does not exist: {resolved_model_folder}")
    elif resolved_model_folder:
        body25_model = Path(resolved_model_folder).expanduser() / "pose/body_25/pose_iter_584000.caffemodel"
        if not body25_model.exists() or body25_model.stat().st_size == 0:
            reasons.append(f"OpenPose BODY_25 model not found or empty: {body25_model}")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "binary": resolved_binary,
        "model_folder": resolved_model_folder or None,
        "env": {
            "binary": OPENPOSE_BIN_ENV,
            "model_folder": OPENPOSE_MODEL_FOLDER_ENV,
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _bbox_from_points(points: np.ndarray, width: int, height: int, min_score: float = 0.05) -> dict[str, float] | None:
    valid = points[:, 2] >= min_score
    if not valid.any():
        return None
    xs = points[valid, 0] / max(width, 1)
    ys = points[valid, 1] / max(height, 1)
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


def _mask_bbox(mask: np.ndarray | None) -> dict[str, float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    height, width = mask.shape[:2]
    return {
        "x": round(float(xs.min()) / max(width, 1), 6),
        "y": round(float(ys.min()) / max(height, 1), 6),
        "width": round(float(xs.max() - xs.min() + 1) / max(width, 1), 6),
        "height": round(float(ys.max() - ys.min() + 1) / max(height, 1), 6),
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


def _landmarks_from_body25(points: np.ndarray, width: int, height: int) -> list[dict[str, Any]]:
    landmarks = []
    for index, name in enumerate(BODY25_NAMES):
        x, y, score = points[index]
        landmarks.append(
            {
                "index": index,
                "name": name,
                "x": round(float(x) / max(width, 1), 6),
                "y": round(float(y) / max(height, 1), 6),
                "score": round(float(score), 6),
            }
        )
    return landmarks


def _empty_canonical_landmark(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "name": LANDMARK_NAMES[index],
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "visibility": 0.0,
        "presence": 0.0,
        "source": "openpose_body25",
        "source_index": None,
        "source_name": None,
        "synthetic": False,
        "missing": True,
    }


def _copy_body25_landmark(
    target: list[dict[str, Any]],
    *,
    media_index: int,
    source: dict[str, Any],
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
        "source": "openpose_body25",
        "source_index": int(source.get("index") or 0),
        "source_name": str(source.get("name") or ""),
        "synthetic": synthetic,
        "missing": False,
    }


def body25_row_to_pose_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map OpenPose BODY_25 output into the canonical 33-landmark pose contract."""
    body25_landmarks = row.get("landmarks") or []
    canonical = [_empty_canonical_landmark(index) for index in range(len(LANDMARK_NAMES))]

    for body25_index, media_index in BODY25_TO_MEDIAPIPE.items():
        if body25_index < len(body25_landmarks):
            _copy_body25_landmark(
                canonical,
                media_index=media_index,
                source=body25_landmarks[body25_index],
            )

    for media_index, body25_index in MEDIAPIPE_SYNTHETIC_FROM_BODY25.items():
        if body25_index < len(body25_landmarks) and canonical[media_index]["missing"]:
            _copy_body25_landmark(
                canonical,
                media_index=media_index,
                source=body25_landmarks[body25_index],
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
        "usable": usable,
        "drop_reason": None if usable else row.get("drop_reason") or "openpose_missing_or_off_mask",
        "visibility_mean": round(visibility_mean, 6),
        "presence_mean": round(visibility_mean, 6),
        "bbox": row.get("bbox"),
        "selected_pose_index": row.get("selected_pose_index"),
        "candidate_count": row.get("candidate_count"),
        "mask_iou": row.get("mask_iou"),
        "source_pose_backend": "openpose_body25",
        "landmarks": canonical,
        "world_landmarks": [],
    }


def _summarize_canonical_pose(rows: Sequence[dict[str, Any]], raw_summary: dict[str, Any]) -> dict[str, Any]:
    visibility_values = [float(row.get("visibility_mean") or 0.0) for row in rows]
    usable_frames = sum(1 for row in rows if row.get("usable"))
    frame_count = len(rows)
    return {
        "quality": {
            "pose_hit_rate": raw_summary.get("pose_hit_rate", 0.0),
            "usable_rate": round(usable_frames / frame_count, 4) if frame_count else 0.0,
            "visibility_mean": round(float(np.mean(visibility_values)), 6) if visibility_values else 0.0,
            "landmark_count": len(LANDMARK_NAMES),
            "source_landmark_count": len(BODY25_NAMES),
        },
        "raw_openpose": raw_summary,
    }


def select_openpose_person(
    people: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    mask_bbox: dict[str, float] | None,
    min_mask_iou: float = 0.02,
) -> tuple[int | None, np.ndarray | None, dict[str, float] | None, float]:
    best_index: int | None = None
    best_points: np.ndarray | None = None
    best_bbox: dict[str, float] | None = None
    best_iou = 0.0
    best_score = -1.0
    for index, person in enumerate(people):
        raw_points = person.get("pose_keypoints_2d") or []
        if len(raw_points) < len(BODY25_NAMES) * 3:
            continue
        points = np.asarray(raw_points, dtype=np.float32).reshape(-1, 3)[: len(BODY25_NAMES)]
        bbox = _bbox_from_points(points, width, height)
        mask_iou = _bbox_iou(bbox, mask_bbox)
        if mask_bbox and mask_iou < min_mask_iou:
            continue
        visible = points[:, 2] >= 0.05
        confidence = float(points[visible, 2].mean()) if visible.any() else 0.0
        score = confidence + mask_iou * 4.0
        if score > best_score:
            best_index = index
            best_points = points
            best_bbox = bbox
            best_iou = mask_iou
            best_score = score
    return best_index, best_points, best_bbox, best_iou


def draw_openpose_skeleton(frame: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    landmarks = row.get("landmarks") or []
    if not landmarks:
        return output
    for start, end in BODY25_CONNECTIONS:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        a = landmarks[start]
        b = landmarks[end]
        if float(a.get("score") or 0.0) < 0.05 or float(b.get("score") or 0.0) < 0.05:
            continue
        p1 = (int(float(a["x"]) * width), int(float(a["y"]) * height))
        p2 = (int(float(b["x"]) * width), int(float(b["y"]) * height))
        cv2.line(output, p1, p2, (42, 36, 30), 3, lineType=cv2.LINE_AA)
        cv2.line(output, p1, p2, (118, 202, 255), 1, lineType=cv2.LINE_AA)
    for landmark in landmarks:
        if float(landmark.get("score") or 0.0) < 0.05:
            continue
        point = (int(float(landmark["x"]) * width), int(float(landmark["y"]) * height))
        cv2.circle(output, point, 4, (255, 250, 243), -1, lineType=cv2.LINE_AA)
        cv2.circle(output, point, 4, (42, 36, 30), 1, lineType=cv2.LINE_AA)
    return output


def _prepare_openpose_input_frames(
    *,
    source_segment: Path,
    runner_mask: Path | None,
    input_dir: Path,
    progress_callback: OpenPoseProgressCallback | None,
    started_at: float,
) -> dict[str, Any]:
    meta = inspect_video(source_segment)
    width = int(meta["width"])
    height = int(meta["height"])
    frame_count = int(meta.get("frame_count") or 0)
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("*.png"):
        old.unlink()

    source_capture = cv2.VideoCapture(str(source_segment))
    mask_capture = cv2.VideoCapture(str(runner_mask)) if runner_mask and runner_mask.exists() else None
    frame_index = 0
    try:
        while True:
            ok, frame = source_capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = _mask_frame(mask_capture, width, height)
            cv2.imwrite(str(input_dir / f"frame_{frame_index:012d}.png"), hard_mask_frame(frame, mask))
            frame_index += 1
            if progress_callback and (frame_index == 1 or frame_index % 10 == 0):
                progress_callback(
                    build_openpose_progress(
                        phase="preparing_openpose_frames",
                        processed_frames=frame_index,
                        total_frames=frame_count,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )
    finally:
        source_capture.release()
        if mask_capture:
            mask_capture.release()
    return {**meta, "frame_count": frame_index}


def _run_openpose_binary(
    *,
    binary: str,
    input_dir: Path,
    output_dir: Path,
    model_folder: str | None,
    total_frames: int = 0,
    progress_callback: OpenPoseProgressCallback | None = None,
    started_at: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.json"):
        old.unlink()
    command = [
        binary,
        "--image_dir",
        str(input_dir),
        "--write_json",
        str(output_dir),
        "--display",
        "0",
        "--render_pose",
        "0",
        "--model_pose",
        "BODY_25",
    ]
    if model_folder:
        command.extend(["--model_folder", model_folder])
    stdout_path = output_dir.parent / "openpose_stdout.log"
    stderr_path = output_dir.parent / "openpose_stderr.log"
    started = started_at if started_at is not None else time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        while True:
            return_code = process.poll()
            if progress_callback and total_frames:
                processed_frames = len(list(output_dir.glob("*_keypoints.json")))
                progress_callback(
                    build_openpose_progress(
                        phase="running_openpose",
                        processed_frames=processed_frames,
                        total_frames=total_frames,
                        elapsed_seconds=time.monotonic() - started,
                    )
                )
            if return_code is not None:
                break
            time.sleep(1.0)
    if return_code != 0:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"OpenPose failed with exit code {return_code}: {stderr_tail}")


def _openpose_json_for_frame(output_dir: Path, frame_index: int) -> Path:
    return output_dir / f"frame_{frame_index:012d}_keypoints.json"


def _rows_from_openpose_json(
    *,
    output_dir: Path,
    source_segment: Path,
    runner_mask: Path | None,
    landmarks_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    progress_callback: OpenPoseProgressCallback | None,
    started_at: float,
) -> dict[str, Any]:
    meta = inspect_video(source_segment)
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])
    frame_count = int(meta.get("frame_count") or 0)
    source_capture = cv2.VideoCapture(str(source_segment))
    mask_capture = cv2.VideoCapture(str(runner_mask)) if runner_mask and runner_mask.exists() else None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    skeleton_writer = cv2.VideoWriter(str(skeleton_render_path), fourcc, fps, (width, height), True)
    qa_writer = cv2.VideoWriter(str(qa_overlay_path), fourcc, fps, (width, height), True)
    if not skeleton_writer.isOpened() or not qa_writer.isOpened():
        source_capture.release()
        if mask_capture:
            mask_capture.release()
        raise ValueError("Could not open OpenPose output video writers")

    rows: list[dict[str, Any]] = []
    try:
        for frame_index in range(frame_count):
            ok, source_frame = source_capture.read()
            if not ok:
                break
            if source_frame.shape[:2] != (height, width):
                source_frame = cv2.resize(source_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = _mask_frame(mask_capture, width, height)
            mask_bbox = _mask_bbox(mask)
            json_path = _openpose_json_for_frame(output_dir, frame_index)
            people = []
            if json_path.exists():
                people = json.loads(json_path.read_text(encoding="utf-8")).get("people") or []
            selected_index, points, bbox, mask_iou = select_openpose_person(
                people,
                width=width,
                height=height,
                mask_bbox=mask_bbox,
            )
            usable = points is not None
            row = {
                "frame_index": frame_index,
                "time_seconds": round(frame_index / fps, 3) if fps else None,
                "frame_width": width,
                "frame_height": height,
                "detected": bool(people),
                "usable": usable,
                "drop_reason": None if usable else "openpose_missing_or_off_mask",
                "selected_pose_index": selected_index,
                "candidate_count": len(people),
                "mask_iou": round(mask_iou, 4),
                "bbox": bbox,
                "landmarks": _landmarks_from_body25(points, width, height) if points is not None else [],
            }
            rows.append(row)
            skeleton_base = np.full((height, width, 3), (239, 235, 227), dtype=np.uint8)
            skeleton_writer.write(draw_openpose_skeleton(skeleton_base, row))
            qa_writer.write(draw_openpose_skeleton(source_frame, row))
            if progress_callback and (frame_index == 0 or (frame_index + 1) % 10 == 0):
                progress_callback(
                    build_openpose_progress(
                        phase="reading_openpose_results",
                        processed_frames=frame_index + 1,
                        total_frames=frame_count,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )
    finally:
        source_capture.release()
        if mask_capture:
            mask_capture.release()
        skeleton_writer.release()
        qa_writer.release()

    write_jsonl(landmarks_path, rows)
    make_browser_playable_mp4s([skeleton_render_path, qa_overlay_path])
    detected = sum(1 for row in rows if row["detected"])
    usable = sum(1 for row in rows if row["usable"])
    return {
        "frame_count": len(rows),
        "detected_frames": detected,
        "usable_frames": usable,
        "pose_hit_rate": round(detected / len(rows), 4) if rows else 0.0,
        "usable_rate": round(usable / len(rows), 4) if rows else 0.0,
    }


def compare_openpose_to_mediapipe(
    *,
    openpose_landmarks_path: Path,
    mediapipe_landmarks_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    openpose_rows = read_jsonl(openpose_landmarks_path)
    mediapipe_rows = read_jsonl(mediapipe_landmarks_path)
    media_by_frame = {int(row.get("frame_index") or index): row for index, row in enumerate(mediapipe_rows)}
    bbox_ious: list[float] = []
    keypoint_distances: list[float] = []
    both_usable = 0
    for openpose_row in openpose_rows:
        frame_index = int(openpose_row.get("frame_index") or 0)
        media_row = media_by_frame.get(frame_index)
        if not media_row or not openpose_row.get("usable") or not media_row.get("usable"):
            continue
        both_usable += 1
        bbox_ious.append(_bbox_iou(openpose_row.get("bbox"), media_row.get("bbox")))
        openpose_landmarks = openpose_row.get("landmarks") or []
        media_landmarks = media_row.get("landmarks") or []
        for openpose_index, media_index in BODY25_TO_MEDIAPIPE.items():
            if openpose_index >= len(openpose_landmarks) or media_index >= len(media_landmarks):
                continue
            openpose_point = openpose_landmarks[openpose_index]
            media_point = media_landmarks[media_index]
            if float(openpose_point.get("score") or 0.0) < 0.05:
                continue
            if float(media_point.get("visibility") or 0.0) < 0.2:
                continue
            dx = float(openpose_point["x"]) - float(media_point["x"])
            dy = float(openpose_point["y"]) - float(media_point["y"])
            keypoint_distances.append(float((dx * dx + dy * dy) ** 0.5))
    payload = {
        "version": 1,
        "created_at": utc_now_iso(),
        "frame_count": len(openpose_rows),
        "both_usable_frames": both_usable,
        "bbox_iou_mean": round(float(np.mean(bbox_ious)), 6) if bbox_ious else 0.0,
        "keypoint_distance_mean": round(float(np.mean(keypoint_distances)), 6)
        if keypoint_distances
        else 0.0,
        "keypoint_pairs": len(keypoint_distances),
    }
    write_json(output_path, payload)
    return payload


def update_manifest_openpose(
    manifest_path: Path,
    *,
    status: str,
    landmarks_path: Path,
    comparison_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    manifest = read_json(manifest_path)
    paths = manifest.setdefault("paths", {})
    paths["openpose_landmarks"] = str(landmarks_path)
    paths["openpose_skeleton_render"] = str(skeleton_render_path)
    paths["openpose_qa_overlay"] = str(qa_overlay_path)
    paths["pose_comparison"] = str(comparison_path)
    stage = manifest.setdefault("stages", {}).setdefault("openpose", {})
    stage["status"] = status
    stage["recommended_tool"] = "OpenPose BODY_25 optional benchmark"
    stage["output"] = str(landmarks_path)
    stage["comparison"] = str(comparison_path)
    if summary:
        stage["summary"] = summary
    if error:
        stage["error"] = error
    else:
        stage.pop("error", None)
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def update_manifest_openpose_pose(
    manifest_path: Path,
    *,
    pose_landmarks_path: Path,
    raw_openpose_landmarks_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    features_path: Path,
    result: dict[str, Any],
) -> None:
    manifest = read_json(manifest_path)
    paths = manifest.setdefault("paths", {})
    paths["pose_landmarks"] = str(pose_landmarks_path)
    paths["openpose_landmarks"] = str(raw_openpose_landmarks_path)
    paths["skeleton_render"] = str(skeleton_render_path)
    paths["qa_overlay"] = str(qa_overlay_path)
    paths["openpose_skeleton_render"] = str(skeleton_render_path)
    paths["openpose_qa_overlay"] = str(qa_overlay_path)
    paths["features"] = str(features_path)
    stages = manifest.setdefault("stages", {})
    quality = result.get("quality", {})
    stages["pose"] = {
        "status": "complete",
        "backend": "openpose_body25",
        "recommended_tool": "OpenPose BODY_25 canonical pose",
        "output": str(pose_landmarks_path),
        "raw_output": str(raw_openpose_landmarks_path),
        "summary": {
            "pose_hit_rate": quality.get("pose_hit_rate"),
            "usable_rate": quality.get("usable_rate"),
            "visibility_mean": quality.get("visibility_mean"),
        },
    }
    stages.setdefault("renders", {})["status"] = "partial_complete"
    stages["renders"]["skeleton_render"] = str(skeleton_render_path)
    stages.setdefault("features", {})["status"] = "complete"
    stages["features"]["output"] = str(features_path)
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def run_openpose_pose(
    *,
    run_dir: Path,
    binary_path: str | Path | None = None,
    model_folder: str | Path | None = None,
    progress_callback: OpenPoseProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = read_json(manifest_path)
    paths = manifest["paths"]
    source_segment = Path(str(paths["source_segment"]))
    runner_mask = Path(str(paths.get("runner_mask") or ""))
    raw_landmarks_path = Path(str(paths.get("openpose_landmarks") or run_dir / "openpose_landmarks.jsonl"))
    pose_landmarks_path = Path(str(paths.get("pose_landmarks") or run_dir / "pose_landmarks.jsonl"))
    skeleton_render_path = Path(str(paths.get("skeleton_render") or run_dir / "skeleton_render.mp4"))
    qa_overlay_path = Path(str(paths.get("qa_overlay") or run_dir / "qa_overlay.mp4"))
    features_path = Path(str(paths.get("features") or run_dir / "features.json"))

    setup = openpose_setup_status(binary_path=binary_path, model_folder=model_folder)
    if not setup["ready"]:
        error = "; ".join(setup["reasons"])
        update_manifest_openpose(
            manifest_path,
            status="unavailable",
            landmarks_path=raw_landmarks_path,
            comparison_path=Path(str(paths.get("pose_comparison") or run_dir / "pose_comparison.json")),
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            error=error,
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "backend": "openpose_body25",
            "status": "unavailable",
            "error": error,
            "setup": setup,
            "frame_count": 0,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }

    if progress_callback:
        progress_callback(
            build_openpose_progress(
                phase="preparing_openpose_frames",
                processed_frames=0,
                total_frames=0,
                elapsed_seconds=0.0,
            )
        )
    input_dir = run_dir / "openpose_input_frames"
    output_dir = run_dir / "openpose_json"
    meta = _prepare_openpose_input_frames(
        source_segment=source_segment,
        runner_mask=runner_mask if runner_mask.exists() else None,
        input_dir=input_dir,
        progress_callback=progress_callback,
        started_at=started_at,
    )
    if progress_callback:
        progress_callback(
            build_openpose_progress(
                phase="running_openpose",
                processed_frames=0,
                total_frames=int(meta.get("frame_count") or 0),
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    try:
        _run_openpose_binary(
            binary=str(setup["binary"]),
            input_dir=input_dir,
            output_dir=output_dir,
            model_folder=str(setup["model_folder"]) if setup.get("model_folder") else None,
            total_frames=int(meta.get("frame_count") or 0),
            progress_callback=progress_callback,
            started_at=started_at,
        )
        raw_summary = _rows_from_openpose_json(
            output_dir=output_dir,
            source_segment=source_segment,
            runner_mask=runner_mask if runner_mask.exists() else None,
            landmarks_path=raw_landmarks_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            progress_callback=progress_callback,
            started_at=started_at,
        )
        raw_rows = read_jsonl(raw_landmarks_path)
        pose_rows = [body25_row_to_pose_row(row) for row in raw_rows]
        write_jsonl(pose_landmarks_path, pose_rows)
        summary = _summarize_canonical_pose(pose_rows, raw_summary)
        features = {
            "version": 1,
            "created_at": utc_now_iso(),
            "backend": "openpose_body25",
            "model_pose": "BODY_25",
            "input_video": str(source_segment),
            "frame_count": len(pose_rows),
            **summary,
        }
        write_json(features_path, features)
        update_manifest_openpose_pose(
            manifest_path,
            pose_landmarks_path=pose_landmarks_path,
            raw_openpose_landmarks_path=raw_landmarks_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            features_path=features_path,
            result=summary,
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "backend": "openpose_body25",
            "status": "complete",
            "frame_count": len(pose_rows),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "pose_landmarks_path": str(pose_landmarks_path),
            "openpose_landmarks": str(raw_landmarks_path),
            "skeleton_render_path": str(skeleton_render_path),
            "qa_overlay_path": str(qa_overlay_path),
            "features_path": str(features_path),
            **summary,
        }
    except Exception as exc:
        update_manifest_openpose(
            manifest_path,
            status="failed",
            landmarks_path=raw_landmarks_path,
            comparison_path=Path(str(paths.get("pose_comparison") or run_dir / "pose_comparison.json")),
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            error=str(exc),
        )
        raise


def run_openpose_comparison(
    *,
    run_dir: Path,
    binary_path: str | Path | None = None,
    model_folder: str | Path | None = None,
    progress_callback: OpenPoseProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = read_json(manifest_path)
    paths = manifest["paths"]
    source_segment = Path(str(paths["source_segment"]))
    runner_mask = Path(str(paths.get("runner_mask") or ""))
    mediapipe_landmarks = Path(str(paths["pose_landmarks"]))
    landmarks_path = Path(str(paths.get("openpose_landmarks") or run_dir / "openpose_landmarks.jsonl"))
    comparison_path = Path(str(paths.get("pose_comparison") or run_dir / "pose_comparison.json"))
    skeleton_render_path = Path(
        str(paths.get("openpose_skeleton_render") or run_dir / "openpose_skeleton_render.mp4")
    )
    qa_overlay_path = Path(str(paths.get("openpose_qa_overlay") or run_dir / "openpose_qa_overlay.mp4"))

    setup = openpose_setup_status(binary_path=binary_path, model_folder=model_folder)
    if not setup["ready"]:
        error = "; ".join(setup["reasons"])
        update_manifest_openpose(
            manifest_path,
            status="unavailable",
            landmarks_path=landmarks_path,
            comparison_path=comparison_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            error=error,
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "backend": "openpose_body25",
            "status": "unavailable",
            "error": error,
            "setup": setup,
            "frame_count": 0,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }

    if progress_callback:
        progress_callback(
            build_openpose_progress(
                phase="preparing_openpose_frames",
                processed_frames=0,
                total_frames=0,
                elapsed_seconds=0.0,
            )
        )
    input_dir = run_dir / "openpose_input_frames"
    output_dir = run_dir / "openpose_json"
    meta = _prepare_openpose_input_frames(
        source_segment=source_segment,
        runner_mask=runner_mask if runner_mask.exists() else None,
        input_dir=input_dir,
        progress_callback=progress_callback,
        started_at=started_at,
    )
    if progress_callback:
        progress_callback(
            build_openpose_progress(
                phase="running_openpose",
                processed_frames=0,
                total_frames=int(meta.get("frame_count") or 0),
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    try:
        _run_openpose_binary(
            binary=str(setup["binary"]),
            input_dir=input_dir,
            output_dir=output_dir,
            model_folder=str(setup["model_folder"]) if setup.get("model_folder") else None,
            total_frames=int(meta.get("frame_count") or 0),
            progress_callback=progress_callback,
            started_at=started_at,
        )
        summary = _rows_from_openpose_json(
            output_dir=output_dir,
            source_segment=source_segment,
            runner_mask=runner_mask if runner_mask.exists() else None,
            landmarks_path=landmarks_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            progress_callback=progress_callback,
            started_at=started_at,
        )
        comparison = compare_openpose_to_mediapipe(
            openpose_landmarks_path=landmarks_path,
            mediapipe_landmarks_path=mediapipe_landmarks,
            output_path=comparison_path,
        )
        update_manifest_openpose(
            manifest_path,
            status="complete",
            landmarks_path=landmarks_path,
            comparison_path=comparison_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            summary={**summary, **comparison},
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "backend": "openpose_body25",
            "status": "complete",
            "frame_count": summary["frame_count"],
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "openpose": summary,
            "comparison": comparison,
            "openpose_landmarks": str(landmarks_path),
            "pose_comparison": str(comparison_path),
        }
    except Exception as exc:
        update_manifest_openpose(
            manifest_path,
            status="failed",
            landmarks_path=landmarks_path,
            comparison_path=comparison_path,
            skeleton_render_path=skeleton_render_path,
            qa_overlay_path=qa_overlay_path,
            error=str(exc),
        )
        raise
