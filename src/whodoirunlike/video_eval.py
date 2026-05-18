from __future__ import annotations

import csv
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import imageio_ffmpeg
import mediapipe as mp
from rich.console import Console
from yt_dlp import YoutubeDL


POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

FOOT_LANDMARKS = [27, 28, 29, 30, 31, 32]
TORSO_LANDMARKS = [11, 12, 23, 24]


@dataclass(frozen=True)
class FramePoseStats:
    detected: bool
    visibility_mean: float
    body_height: float
    body_width: float
    full_body_visible: bool
    size_ok: bool
    view_proxy: float
    ankle_mid_x: float | None
    ankle_mid_y: float | None


def ensure_pose_model(model_dir: Path, variant: str = "lite") -> Path:
    if variant not in POSE_MODEL_URLS:
        raise ValueError(f"Unknown pose model variant: {variant}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"pose_landmarker_{variant}.task"
    if not model_path.exists():
        urllib.request.urlretrieve(POSE_MODEL_URLS[variant], model_path)
    return model_path


def candidate_rows(
    scored_csv: Path,
    limit: int,
    recommendations: set[str],
    max_duration_seconds: float | None,
) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(scored_csv.open(newline="", encoding="utf-8")))
    filtered = [row for row in rows if row.get("recommendation") in recommendations]
    if max_duration_seconds is not None:
        filtered = [
            row
            for row in filtered
            if row.get("duration_seconds") and float(row["duration_seconds"]) <= max_duration_seconds
        ]
    return filtered[:limit]


def youtube_format(max_height: int | None) -> str:
    if max_height is None:
        return "bv*/b/best"
    if max_height <= 0:
        raise ValueError("max_height must be positive or None")
    return (
        f"bv*[height<={max_height}][ext=mp4]/"
        f"b[height<={max_height}][ext=mp4]/"
        f"best[height<={max_height}][ext=mp4]/"
        f"bv*[height<={max_height}]/"
        f"b[height<={max_height}]/"
        f"best[height<={max_height}]/"
        "bv*[ext=mp4]/b[ext=mp4]/best[ext=mp4]/bv*/b/best"
    )


def download_candidate(
    row: dict[str, Any],
    output_dir: Path,
    *,
    force: bool = False,
    max_height: int | None = 360,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{row['candidate_id']}.mp4"
    if not force:
        matches = [
            path
            for path in output_dir.glob(f"{row['candidate_id']}.*")
            if path.is_file()
            and path.stat().st_size > 0
            and not path.name.endswith((".part", ".ytdl", ".temp", ".tmp"))
        ]
        if matches:
            return matches[0]

    opts = {
        "format": youtube_format(max_height),
        "outtmpl": str(video_path.with_suffix(".%(ext)s")),
        "quiet": True,
        "noplaylist": True,
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "merge_output_format": "mp4",
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([row["url"]])

    matches = list(output_dir.glob(f"{row['candidate_id']}.*"))
    if not matches:
        raise FileNotFoundError(f"yt-dlp did not create a file for {row['candidate_id']}")
    return next(
        path
        for path in matches
        if path.is_file() and not path.name.endswith((".part", ".ytdl", ".temp", ".tmp"))
    )


def sample_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= sample_count:
        return list(range(total_frames))
    return sorted({round(i * (total_frames - 1) / (sample_count - 1)) for i in range(sample_count)})


def frame_stats(result: Any) -> FramePoseStats:
    if not result.pose_landmarks:
        return FramePoseStats(False, 0.0, 0.0, 0.0, False, False, 0.0, None, None)

    landmarks = result.pose_landmarks[0]
    visible_landmarks = [
        lm for lm in landmarks if getattr(lm, "visibility", 1.0) is None or lm.visibility >= 0.35
    ]
    if not visible_landmarks:
        return FramePoseStats(False, 0.0, 0.0, 0.0, False, False, 0.0, None, None)

    xs = [max(0.0, min(1.0, lm.x)) for lm in visible_landmarks]
    ys = [max(0.0, min(1.0, lm.y)) for lm in visible_landmarks]
    body_width = max(xs) - min(xs)
    body_height = max(ys) - min(ys)
    visibility_values = [getattr(lm, "visibility", 1.0) or 0.0 for lm in visible_landmarks]
    visibility_mean = mean(visibility_values)

    full_body_visible = min(xs) > 0.015 and max(xs) < 0.985 and min(ys) > 0.015 and max(ys) < 0.995
    foot_visibility = mean(getattr(landmarks[i], "visibility", 0.0) or 0.0 for i in FOOT_LANDMARKS)
    full_body_visible = full_body_visible and foot_visibility >= 0.35
    size_ok = 0.18 <= body_height <= 0.88 and body_width <= 0.82

    shoulder_span = abs(landmarks[11].x - landmarks[12].x)
    hip_span = abs(landmarks[23].x - landmarks[24].x)
    view_proxy = (shoulder_span + hip_span) / max(body_height, 1e-6)

    ankle_mid_x = mean([landmarks[27].x, landmarks[28].x])
    ankle_mid_y = mean([landmarks[27].y, landmarks[28].y])

    return FramePoseStats(
        True,
        visibility_mean,
        body_height,
        body_width,
        full_body_visible,
        size_ok,
        view_proxy,
        ankle_mid_x,
        ankle_mid_y,
    )


def motion_score(stats: list[FramePoseStats]) -> float:
    xs = [item.ankle_mid_x for item in stats if item.detected and item.ankle_mid_x is not None]
    ys = [item.ankle_mid_y for item in stats if item.detected and item.ankle_mid_y is not None]
    if len(xs) < 4 or len(ys) < 4:
        return 0.0
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    return min(1.0, (x_range * 1.8) + (y_range * 2.5))


def classify_view(view_proxy: float) -> str:
    if view_proxy <= 0:
        return "unknown"
    if view_proxy < 0.28:
        return "side-ish"
    if view_proxy < 0.48:
        return "diagonal-ish"
    return "front/rear-ish"


def evaluate_video(video_path: Path, model_path: Path, sample_count: int) -> dict[str, Any]:
    mp_image = mp.Image
    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if fps else 0.0
    indices = sample_frame_indices(total_frames, sample_count)

    stats: list[FramePoseStats] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp_image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = math.floor(frame_idx / fps * 1000)
            result = landmarker.detect_for_video(image, timestamp_ms)
            stats.append(frame_stats(result))

    cap.release()

    sample_total = len(stats)
    detected = [item for item in stats if item.detected]
    detected_count = len(detected)
    pose_hit_rate = detected_count / sample_total if sample_total else 0.0
    visibility_mean = mean(item.visibility_mean for item in detected) if detected else 0.0
    full_body_rate = (
        sum(1 for item in detected if item.full_body_visible) / detected_count if detected_count else 0.0
    )
    size_ok_rate = sum(1 for item in detected if item.size_ok) / detected_count if detected_count else 0.0
    view_proxy = mean(item.view_proxy for item in detected) if detected else 0.0
    movement = motion_score(stats)

    visual_score = round(
        (pose_hit_rate * 35)
        + (visibility_mean * 20)
        + (full_body_rate * 20)
        + (size_ok_rate * 15)
        + (movement * 10)
    )
    if visual_score >= 78:
        recommendation = "cv_good"
    elif visual_score >= 58:
        recommendation = "cv_maybe"
    else:
        recommendation = "cv_poor"

    reasons = []
    if pose_hit_rate >= 0.75:
        reasons.append("pose detected frequently")
    elif pose_hit_rate >= 0.4:
        reasons.append("pose detected intermittently")
    else:
        reasons.append("weak pose detection")
    if full_body_rate >= 0.6:
        reasons.append("mostly full body")
    elif full_body_rate < 0.25:
        reasons.append("limited full-body visibility")
    if size_ok_rate >= 0.7:
        reasons.append("runner size usable")
    elif size_ok_rate < 0.4:
        reasons.append("runner too small/cropped often")
    if movement >= 0.35:
        reasons.append("visible foot/ankle motion")
    else:
        reasons.append("limited gait motion in sampled frames")
    reasons.append(f"view {classify_view(view_proxy)}")

    return {
        "video_path": str(video_path),
        "duration_seconds_local": round(duration_seconds, 2),
        "fps": round(fps, 3),
        "sampled_frames": sample_total,
        "pose_hit_rate": round(pose_hit_rate, 3),
        "visibility_mean": round(visibility_mean, 3),
        "full_body_rate": round(full_body_rate, 3),
        "size_ok_rate": round(size_ok_rate, 3),
        "motion_score": round(movement, 3),
        "view_proxy": round(view_proxy, 3),
        "camera_angle_proxy": classify_view(view_proxy),
        "cv_score": visual_score,
        "cv_recommendation": recommendation,
        "cv_reasons": "; ".join(reasons),
    }


def write_cv_evaluation(
    scored_csv: Path,
    out_csv: Path,
    *,
    limit: int,
    sample_count: int,
    recommendations: set[str],
    download_dir: Path,
    model_dir: Path,
    model_variant: str,
    max_duration_seconds: float | None = 900,
    max_height: int | None = 360,
    force_download: bool = False,
) -> int:
    console = Console()
    rows = candidate_rows(scored_csv, limit, recommendations, max_duration_seconds)
    model_path = ensure_pose_model(model_dir, model_variant)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        console.print(f"[bold]{index}/{len(rows)}[/bold] {row['runner_name']}: {row['title'][:80]}")
        try:
            video_path = download_candidate(
                row,
                download_dir,
                force=force_download,
                max_height=max_height,
            )
            metrics = evaluate_video(video_path, model_path, sample_count)
            output_rows.append({**row, **metrics, "error": ""})
        except Exception as exc:
            output_rows.append({**row, "cv_score": 0, "cv_recommendation": "error", "error": str(exc)})

    if not output_rows:
        return 0

    fieldnames = list(dict.fromkeys(key for row in output_rows for key in row.keys()))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    json_path = out_csv.with_suffix(".json")
    json_path.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    return len(output_rows)
