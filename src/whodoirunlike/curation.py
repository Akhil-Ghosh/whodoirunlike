from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import cv2
import imageio_ffmpeg
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso


DEFAULT_MIN_WINDOW_SECONDS = 2.0
DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_MAX_WINDOW_SECONDS = 6.0
DEFAULT_STEP_SECONDS = 1.0
DEFAULT_SAMPLE_FPS = 2.0
VIEW_BUCKETS = {"side", "diagonal", "front", "rear", "mixed", "unknown"}


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class ShotInterval:
    start_seconds: float
    end_seconds: float
    source: str = "manual"

    @property
    def duration_seconds(self) -> float:
        return round(max(0.0, self.end_seconds - self.start_seconds), 3)


@dataclass(frozen=True)
class WindowInterval:
    start_seconds: float
    end_seconds: float
    shot_index: int

    @property
    def duration_seconds(self) -> float:
        return round(max(0.0, self.end_seconds - self.start_seconds), 3)


@dataclass(frozen=True)
class WindowMetrics:
    visible_person_fraction: float
    pose_visibility: float
    track_continuity: float
    runningness: float
    side_view_prior: float
    bbox_scale: float
    crowd_penalty: float = 0.0
    occlusion_penalty: float = 0.0
    replay_graphic_penalty: float = 0.0
    sampled_frames: int = 0
    motion_mean: float = 0.0
    motion_area_mean: float = 0.0
    edge_density_mean: float = 0.0


@dataclass(frozen=True)
class ClipWindow:
    window_id: str
    video_path: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    shot_index: int
    rank: int
    score: float
    score_components: dict[str, Any]
    thumbnail_path: str | None = None
    preview_path: str | None = None


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def canonical_view_bucket(value: str | None) -> str:
    bucket = str(value or "unknown").strip().lower()
    return bucket if bucket in VIEW_BUCKETS else "unknown"


def side_view_prior_for_bucket(view_bucket: str | None) -> float:
    return {
        "side": 1.0,
        "diagonal": 0.8,
        "mixed": 0.4,
        "front": 0.25,
        "rear": 0.25,
        "unknown": 0.0,
    }[canonical_view_bucket(view_bucket)]


def score_window(metrics: WindowMetrics) -> float:
    pose_visibility = clamp01(
        0.60 * metrics.pose_visibility + 0.40 * metrics.visible_person_fraction
    )
    score = (
        0.30 * pose_visibility
        + 0.25 * clamp01(metrics.track_continuity)
        + 0.20 * clamp01(metrics.runningness)
        + 0.10 * clamp01(metrics.side_view_prior)
        + 0.10 * clamp01(metrics.bbox_scale)
        - 0.05 * clamp01(metrics.crowd_penalty)
        - 0.05 * clamp01(metrics.occlusion_penalty)
        - 0.05 * clamp01(metrics.replay_graphic_penalty)
    )
    return round(clamp01(score), 4)


def probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(video_path),
        width=width,
        height=height,
        fps=round(fps, 4),
        frame_count=frame_count,
        duration_seconds=round(duration_seconds, 3),
    )


def detect_shots(
    video_path: Path,
    *,
    threshold: float = 27.0,
    min_scene_len_seconds: float = 1.0,
) -> list[ShotInterval]:
    info = probe_video(video_path)
    fallback = [ShotInterval(0.0, info.duration_seconds, source="fallback_full_video")]
    if info.duration_seconds <= 0:
        return []

    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ModuleNotFoundError:
        return fallback

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        min_scene_len = max(1, int(round(min_scene_len_seconds * max(info.fps, 1.0))))
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
    except Exception:
        return fallback

    intervals = [
        ShotInterval(
            round(start.get_seconds(), 3),
            round(end.get_seconds(), 3),
            source="pyscenedetect_content",
        )
        for start, end in scenes
        if end.get_seconds() > start.get_seconds()
    ]
    return intervals or fallback


def partition_shots(
    shots: Iterable[ShotInterval],
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    min_window_seconds: float = DEFAULT_MIN_WINDOW_SECONDS,
    max_window_seconds: float = DEFAULT_MAX_WINDOW_SECONDS,
) -> list[WindowInterval]:
    window_seconds = max(min_window_seconds, min(float(window_seconds), max_window_seconds))
    step_seconds = max(0.25, float(step_seconds))
    windows: list[WindowInterval] = []

    for shot_index, shot in enumerate(shots):
        shot_duration = shot.duration_seconds
        if shot_duration < min_window_seconds:
            continue
        if shot_duration <= window_seconds:
            windows.append(
                WindowInterval(
                    start_seconds=round(shot.start_seconds, 3),
                    end_seconds=round(shot.end_seconds, 3),
                    shot_index=shot_index,
                )
            )
            continue

        last_start = shot.end_seconds - window_seconds
        current = shot.start_seconds
        while current <= last_start + 1e-6:
            windows.append(
                WindowInterval(
                    start_seconds=round(current, 3),
                    end_seconds=round(current + window_seconds, 3),
                    shot_index=shot_index,
                )
            )
            current += step_seconds

        if windows and windows[-1].shot_index == shot_index and windows[-1].end_seconds < shot.end_seconds:
            final_start = max(shot.start_seconds, shot.end_seconds - window_seconds)
            if final_start - windows[-1].start_seconds >= 0.25:
                windows.append(
                    WindowInterval(
                        start_seconds=round(final_start, 3),
                        end_seconds=round(shot.end_seconds, 3),
                        shot_index=shot_index,
                    )
                )

    return windows


def _sample_times(start_seconds: float, end_seconds: float, sample_fps: float) -> list[float]:
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        return []
    sample_count = max(2, int(math.ceil(duration * max(sample_fps, 0.25))))
    if sample_count == 1:
        return [round(start_seconds, 3)]
    return [
        round(start_seconds + (duration * index / (sample_count - 1)), 3)
        for index in range(sample_count)
    ]


def _read_gray_frame(cap: cv2.VideoCapture, fps: float, time_seconds: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_seconds * fps))))
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def heuristic_window_metrics(
    video_path: Path,
    window: WindowInterval,
    *,
    view_bucket: str = "unknown",
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> WindowMetrics:
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = [
        frame
        for time_seconds in _sample_times(window.start_seconds, window.end_seconds, sample_fps)
        if (frame := _read_gray_frame(cap, info.fps or 30.0, time_seconds)) is not None
    ]
    cap.release()

    edge_values: list[float] = []
    motion_values: list[float] = []
    motion_areas: list[float] = []
    continuity_hits = 0
    for index, gray in enumerate(frames):
        edges = cv2.Canny(gray, 60, 140)
        edge_values.append(float(np.mean(edges > 0)))
        if index == 0:
            continue
        diff = cv2.absdiff(frames[index - 1], gray)
        motion = float(np.mean(diff) / 255.0)
        motion_area = float(np.mean(diff > 18))
        motion_values.append(motion)
        motion_areas.append(motion_area)
        if 0.001 <= motion_area <= 0.35:
            continuity_hits += 1

    motion_mean = mean(motion_values) if motion_values else 0.0
    motion_std = pstdev(motion_values) if len(motion_values) > 1 else 0.0
    motion_area_mean = mean(motion_areas) if motion_areas else 0.0
    edge_density = mean(edge_values) if edge_values else 0.0
    continuity = continuity_hits / len(motion_areas) if motion_areas else 0.0

    visible_proxy = clamp01(edge_density * 3.0 + motion_area_mean * 2.5)
    runningness = clamp01(motion_mean * 8.0 + motion_std * 6.0)
    bbox_scale = clamp01(motion_area_mean * 10.0 + edge_density * 1.5)
    crowd_penalty = clamp01(max(0.0, motion_area_mean - 0.18) * 4.0)

    return WindowMetrics(
        visible_person_fraction=visible_proxy,
        pose_visibility=visible_proxy,
        track_continuity=continuity,
        runningness=runningness,
        side_view_prior=side_view_prior_for_bucket(view_bucket),
        bbox_scale=bbox_scale,
        crowd_penalty=crowd_penalty,
        sampled_frames=len(frames),
        motion_mean=round(motion_mean, 6),
        motion_area_mean=round(motion_area_mean, 6),
        edge_density_mean=round(edge_density, 6),
    )


def window_id(video_path: Path, start_seconds: float, end_seconds: float) -> str:
    raw = f"{video_path.resolve()}:{start_seconds:.3f}:{end_seconds:.3f}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{video_path.stem}-{digest}"


def write_thumbnail(video_path: Path, output_path: Path, time_seconds: float) -> Path:
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_seconds * (info.fps or 30.0)))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read thumbnail frame at {time_seconds:.2f}s")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise ValueError(f"Could not write thumbnail: {output_path}")
    return output_path


def write_preview_clip(
    video_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    max_height: int = 360,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale=-2:min({max_height}\\,ih)"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-to",
        f"{end_seconds:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def _score_components(metrics: WindowMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["pose_visibility_blended"] = round(
        clamp01(0.60 * metrics.pose_visibility + 0.40 * metrics.visible_person_fraction),
        4,
    )
    return payload


def _window_payload(
    *,
    video_path: Path,
    window: WindowInterval,
    rank: int,
    score: float,
    metrics: WindowMetrics,
    thumbnail_path: Path | None = None,
    preview_path: Path | None = None,
) -> ClipWindow:
    return ClipWindow(
        window_id=window_id(video_path, window.start_seconds, window.end_seconds),
        video_path=str(video_path),
        start_seconds=window.start_seconds,
        end_seconds=window.end_seconds,
        duration_seconds=window.duration_seconds,
        shot_index=window.shot_index,
        rank=rank,
        score=score,
        score_components=_score_components(metrics),
        thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
        preview_path=str(preview_path) if preview_path else None,
    )


def propose_clip_windows(
    video_path: Path,
    *,
    top_k: int = 12,
    view_bucket: str = "unknown",
    use_scenedetect: bool = True,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    output_dir: Path | None = None,
    write_thumbnails: bool = False,
    write_previews: bool = False,
) -> dict[str, Any]:
    video_path = video_path.resolve()
    info = probe_video(video_path)
    shots = (
        detect_shots(video_path)
        if use_scenedetect
        else [ShotInterval(0.0, info.duration_seconds, source="full_video")]
    )
    windows = partition_shots(shots, window_seconds=window_seconds, step_seconds=step_seconds)

    scored: list[tuple[float, WindowInterval, WindowMetrics]] = []
    for window in windows:
        metrics = heuristic_window_metrics(
            video_path,
            window,
            view_bucket=view_bucket,
            sample_fps=sample_fps,
        )
        scored.append((score_window(metrics), window, metrics))

    scored.sort(key=lambda item: (-item[0], item[1].start_seconds, item[1].end_seconds))
    selected = scored[: max(0, top_k)]

    output_dir = output_dir.resolve() if output_dir else None
    clip_windows: list[ClipWindow] = []
    for rank, (score, window, metrics) in enumerate(selected, start=1):
        thumb_path: Path | None = None
        preview_path: Path | None = None
        if output_dir and write_thumbnails:
            thumb_path = (
                output_dir
                / "thumbnails"
                / f"{window_id(video_path, window.start_seconds, window.end_seconds)}.jpg"
            )
            write_thumbnail(
                video_path,
                thumb_path,
                time_seconds=(window.start_seconds + window.end_seconds) / 2.0,
            )
        if output_dir and write_previews:
            preview_path = (
                output_dir
                / "previews"
                / f"{window_id(video_path, window.start_seconds, window.end_seconds)}.mp4"
            )
            write_preview_clip(
                video_path,
                preview_path,
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
            )
        clip_windows.append(
            _window_payload(
                video_path=video_path,
                window=window,
                rank=rank,
                score=score,
                metrics=metrics,
                thumbnail_path=thumb_path,
                preview_path=preview_path,
            )
        )

    return {
        "version": 1,
        "created_at": utc_now_iso(),
        "pipeline_goal": "identity_stable_runner_clip_proposal",
        "source_video": str(video_path),
        "video": asdict(info),
        "settings": {
            "top_k": top_k,
            "view_bucket": canonical_view_bucket(view_bucket),
            "use_scenedetect": use_scenedetect,
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "sample_fps": sample_fps,
            "scoring_model": "motion_proxy_v1",
            "needs_detector_pose_rescore": True,
        },
        "shots": [asdict(shot) for shot in shots],
        "windows_considered": len(windows),
        "windows": [asdict(window) for window in clip_windows],
    }


def write_curation_manifest(output_path: Path, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "version": 1,
        "created_at": utc_now_iso(),
        "pipeline_goal": "identity_stable_runner_clip_proposal",
        "videos": manifests,
        "windows": [
            {**window, "source_video": manifest["source_video"]}
            for manifest in manifests
            for window in manifest.get("windows", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
