from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.fusion_runner import DENSEPOSE_PART_GROUPS
from whodoirunlike.pose_runner import LANDMARK_NAMES, POSE_CONNECTIONS
from whodoirunlike.sam2_runner import inspect_video, read_json, write_json


FEATURE_VERSION = 1
JOINT_COUNT = 33
PART_COUNT = 25
FormFeatureProgressCallback = Callable[[dict[str, Any]], None]
DENSEPOSE_GROUP_NAMES = tuple(DENSEPOSE_PART_GROUPS.keys())
BONE_NAMES = tuple(
    f"{LANDMARK_NAMES[start]}__{LANDMARK_NAMES[end]}" for start, end in POSE_CONNECTIONS
)
ANGLE_TRIPLES = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_ankle": (25, 27, 31),
    "right_ankle": (26, 28, 32),
}
SEGMENT_ANGLE_PAIRS = {
    "torso_lean": (23, 11),
    "left_thigh_angle": (23, 25),
    "right_thigh_angle": (24, 26),
    "left_shin_angle": (25, 27),
    "right_shin_angle": (26, 28),
}
ANGLE_NAMES = tuple([*ANGLE_TRIPLES.keys(), *SEGMENT_ANGLE_PAIRS.keys()])


@dataclass(frozen=True)
class FormFeaturePaths:
    metadata_path: Path
    arrays_path: Path


def build_form_feature_progress(
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL artifact: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    sums = np.where(finite, values, 0.0).sum(axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float32),
        where=counts > 0,
    ).astype(np.float32)


def _landmark_matrix(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(rows)
    xy = np.full((frame_count, JOINT_COUNT, 2), np.nan, dtype=np.float32)
    world = np.full((frame_count, JOINT_COUNT, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((frame_count, JOINT_COUNT), dtype=np.float32)
    presence = np.zeros((frame_count, JOINT_COUNT), dtype=np.float32)
    for frame_index, row in enumerate(rows):
        for landmark in row.get("landmarks") or []:
            index = int(landmark.get("index") or 0)
            if not 0 <= index < JOINT_COUNT:
                continue
            xy[frame_index, index] = [
                _as_float(landmark.get("x"), np.nan),
                _as_float(landmark.get("y"), np.nan),
            ]
            visibility[frame_index, index] = _as_float(landmark.get("visibility"))
            presence[frame_index, index] = _as_float(landmark.get("presence"), visibility[frame_index, index])
        for landmark in row.get("world_landmarks") or []:
            index = int(landmark.get("index") or 0)
            if not 0 <= index < JOINT_COUNT:
                continue
            world[frame_index, index] = [
                _as_float(landmark.get("x"), np.nan),
                _as_float(landmark.get("y"), np.nan),
                _as_float(landmark.get("z"), np.nan),
            ]
    return xy, world, visibility, presence


def _safe_midpoint(points: np.ndarray, left: int, right: int) -> np.ndarray:
    pair = points[:, [left, right], :]
    return _nanmean(pair, axis=1)


def normalize_pose_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hip_mid = _safe_midpoint(xy, 23, 24)
    shoulder_mid = _safe_midpoint(xy, 11, 12)
    torso = shoulder_mid - hip_mid
    torso_scale = np.linalg.norm(torso, axis=1)
    hip_width = np.linalg.norm(xy[:, 23] - xy[:, 24], axis=1)
    shoulder_width = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1)
    fallback_scale = _nanmean(np.stack([hip_width, shoulder_width], axis=1), axis=1)
    scale = np.where(np.isfinite(torso_scale) & (torso_scale > 1e-4), torso_scale, fallback_scale)
    scale = np.where(np.isfinite(scale) & (scale > 1e-4), scale, 1.0).astype(np.float32)
    normalized = (xy - hip_mid[:, None, :]) / scale[:, None, None]
    return normalized.astype(np.float32), hip_mid.astype(np.float32), scale.astype(np.float32)


def bone_vectors(normalized_xy: np.ndarray) -> np.ndarray:
    bones = np.full((normalized_xy.shape[0], len(POSE_CONNECTIONS), 2), np.nan, dtype=np.float32)
    for index, (start, end) in enumerate(POSE_CONNECTIONS):
        bones[:, index] = normalized_xy[:, end] - normalized_xy[:, start]
    return bones


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=1)
    norm = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosine = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).astype(np.float32)


def _segment_angle(points: np.ndarray, start: int, end: int) -> np.ndarray:
    vector = points[:, end] - points[:, start]
    return np.degrees(np.arctan2(vector[:, 1], vector[:, 0])).astype(np.float32)


def joint_angles(normalized_xy: np.ndarray) -> np.ndarray:
    values = np.full((normalized_xy.shape[0], len(ANGLE_NAMES)), np.nan, dtype=np.float32)
    for index, name in enumerate(ANGLE_NAMES):
        if name in ANGLE_TRIPLES:
            a, b, c = ANGLE_TRIPLES[name]
            values[:, index] = _angle_between(normalized_xy[:, a], normalized_xy[:, b], normalized_xy[:, c])
        else:
            start, end = SEGMENT_ANGLE_PAIRS[name]
            values[:, index] = _segment_angle(normalized_xy, start, end)
    return values


def angular_velocity(angles: np.ndarray, fps: float) -> np.ndarray:
    if len(angles) == 0:
        return angles.copy()
    velocity = np.zeros_like(angles, dtype=np.float32)
    if len(angles) > 1:
        velocity[1:] = np.diff(angles, axis=0) * float(fps or 0.0)
    return velocity


def _rows_by_frame(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row.get("frame_index") or index): row for index, row in enumerate(rows)}


def frame_state_weight(row: dict[str, Any]) -> float:
    confidence = _as_float(row.get("frame_confidence"))
    state = str(row.get("frame_state") or "usable")
    if state == "usable":
        return confidence
    if state == "short_occlusion":
        return confidence * 0.45
    if state == "densepose_missing":
        return _as_float(row.get("pose_confidence")) * 0.75
    if state in {"pose_rejected", "identity_risk", "cutaway"}:
        return 0.0
    return confidence * 0.25


def fused_weights(
    pose_rows: Sequence[dict[str, Any]], fused_rows: Sequence[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[str]], list[str]]:
    fused_by_frame = _rows_by_frame(fused_rows)
    frame_weights = np.zeros(len(pose_rows), dtype=np.float32)
    joint_weights = np.zeros((len(pose_rows), JOINT_COUNT), dtype=np.float32)
    valid_frames = np.zeros(len(pose_rows), dtype=bool)
    questionable_joints: list[list[str]] = []
    frame_states: list[str] = []
    for index, pose_row in enumerate(pose_rows):
        frame_index = int(pose_row.get("frame_index") or index)
        fused = fused_by_frame.get(frame_index, {})
        if fused:
            frame_weights[index] = frame_state_weight(fused)
            frame_states.append(str(fused.get("frame_state") or "unknown"))
            questionable_joints.append([str(name) for name in fused.get("questionable_joints") or []])
            for joint in fused.get("joint_weights") or []:
                joint_index = int(joint.get("index") or 0)
                if 0 <= joint_index < JOINT_COUNT:
                    joint_weights[index, joint_index] = _as_float(joint.get("weight"))
        else:
            frame_states.append("fused_missing")
            questionable_joints.append([])
            if pose_row.get("usable"):
                frame_weights[index] = _as_float(pose_row.get("visibility_mean")) * 0.65
                for landmark in pose_row.get("landmarks") or []:
                    joint_index = int(landmark.get("index") or 0)
                    if 0 <= joint_index < JOINT_COUNT:
                        joint_weights[index, joint_index] = _as_float(landmark.get("visibility")) * 0.65
        valid_frames[index] = frame_weights[index] > 0
    return frame_weights, joint_weights, valid_frames, questionable_joints, frame_states


def densepose_arrays(
    pose_rows: Sequence[dict[str, Any]],
    densepose_rows: Sequence[dict[str, Any]],
    fused_rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    densepose_by_frame = _rows_by_frame(densepose_rows)
    fused_by_frame = _rows_by_frame(fused_rows)
    group_values = np.zeros((len(pose_rows), len(DENSEPOSE_GROUP_NAMES)), dtype=np.float32)
    group_visibility = np.zeros_like(group_values, dtype=np.float32)
    part_centroids = np.full((len(pose_rows), PART_COUNT, 2), np.nan, dtype=np.float32)
    group_centroids = np.full((len(pose_rows), len(DENSEPOSE_GROUP_NAMES), 2), np.nan, dtype=np.float32)
    for index, pose_row in enumerate(pose_rows):
        frame_index = int(pose_row.get("frame_index") or index)
        densepose = densepose_by_frame.get(frame_index, {})
        fused = fused_by_frame.get(frame_index, {})
        coverage = fused.get("densepose_group_coverage") or {}
        for group_index, group_name in enumerate(DENSEPOSE_GROUP_NAMES):
            group_values[index, group_index] = _as_float(coverage.get(group_name))
            group_visibility[index, group_index] = float(group_values[index, group_index] > 0.015)
        centroids = densepose.get("part_centroids") or {}
        part_pixels = densepose.get("part_pixels") or {}
        for part_key, centroid in centroids.items():
            part_index = int(part_key)
            if 0 <= part_index < PART_COUNT and isinstance(centroid, dict):
                part_centroids[index, part_index] = [
                    _as_float(centroid.get("x"), np.nan),
                    _as_float(centroid.get("y"), np.nan),
                ]
        for group_index, group_name in enumerate(DENSEPOSE_GROUP_NAMES):
            weighted_points = []
            weights = []
            for part_id in DENSEPOSE_PART_GROUPS[group_name]:
                point = part_centroids[index, part_id]
                if np.isfinite(point).all():
                    weighted_points.append(point)
                    weights.append(_as_float(part_pixels.get(str(part_id)), 1.0))
            if weighted_points:
                group_centroids[index, group_index] = np.average(
                    np.asarray(weighted_points), weights=np.asarray(weights), axis=0
                )
    return group_values, group_visibility, part_centroids, group_centroids


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_range(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    return float(np.nanpercentile(values[valid], 90) - np.nanpercentile(values[valid], 10))


def _trajectory_length(points: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        return 0.0
    path = points[valid]
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _joint_range(normalized_xy: np.ndarray, weights: np.ndarray, joint_index: int, axis: int) -> float:
    return _weighted_range(normalized_xy[:, joint_index, axis], weights)


def _rhythm_proxy(normalized_xy: np.ndarray, weights: np.ndarray, fps: float) -> float:
    left = normalized_xy[:, 31, 1]
    right = normalized_xy[:, 32, 1]
    signal = _nanmean(np.stack([left, right]), axis=0)
    valid = np.isfinite(signal) & (weights > 0)
    if valid.sum() < 8 or not fps:
        return 0.0
    centered = signal[valid] - np.nanmean(signal[valid])
    signs = np.sign(centered)
    crossings = np.count_nonzero(np.diff(signs) != 0)
    duration = valid.sum() / fps
    return float(crossings / max(duration, 1e-6))


def summary_features(
    *,
    normalized_xy: np.ndarray,
    raw_xy: np.ndarray,
    angles: np.ndarray,
    frame_weights: np.ndarray,
    densepose_groups: np.ndarray,
    densepose_group_visibility: np.ndarray,
    fps: float,
) -> dict[str, float]:
    angle_lookup = {name: index for index, name in enumerate(ANGLE_NAMES)}
    group_lookup = {name: index for index, name in enumerate(DENSEPOSE_GROUP_NAMES)}
    ankle_path = mean(
        [
            _trajectory_length(normalized_xy[:, joint], frame_weights)
            for joint in (27, 28, 29, 30, 31, 32)
        ]
    )
    arm_swing = mean(
        [
            _joint_range(normalized_xy, frame_weights, joint, axis=0)
            for joint in (15, 16, 17, 18, 19, 20, 21, 22)
        ]
    )
    knee_lift = mean(
        [
            _weighted_range(-normalized_xy[:, 25, 1], frame_weights),
            _weighted_range(-normalized_xy[:, 26, 1], frame_weights),
        ]
    )
    leg_recovery = mean(
        [
            _trajectory_length(normalized_xy[:, joint], frame_weights)
            for joint in (25, 26, 27, 28, 31, 32)
        ]
    )
    hip_mid_y = _nanmean(raw_xy[:, [23, 24], 1], axis=1)
    densepose_visibility = {
        f"{name}_visibility_rate": _weighted_mean(densepose_group_visibility[:, index], frame_weights)
        for name, index in group_lookup.items()
    }
    densepose_coverage = {
        f"{name}_coverage_mean": _weighted_mean(densepose_groups[:, index], frame_weights)
        for name, index in group_lookup.items()
    }
    return {
        "torso_lean_mean": round(_weighted_mean(angles[:, angle_lookup["torso_lean"]], frame_weights), 6),
        "torso_lean_range": round(_weighted_range(angles[:, angle_lookup["torso_lean"]], frame_weights), 6),
        "arm_swing_amplitude": round(arm_swing, 6),
        "leg_recovery_path": round(leg_recovery, 6),
        "knee_lift_proxy": round(knee_lift, 6),
        "foot_ankle_trajectory": round(ankle_path, 6),
        "hip_vertical_oscillation_proxy": round(_weighted_range(hip_mid_y, frame_weights), 6),
        "stride_rhythm_proxy": round(_rhythm_proxy(normalized_xy, frame_weights, fps), 6),
        "left_knee_angle_range": round(_weighted_range(angles[:, angle_lookup["left_knee"]], frame_weights), 6),
        "right_knee_angle_range": round(_weighted_range(angles[:, angle_lookup["right_knee"]], frame_weights), 6),
        "left_elbow_angle_range": round(_weighted_range(angles[:, angle_lookup["left_elbow"]], frame_weights), 6),
        "right_elbow_angle_range": round(_weighted_range(angles[:, angle_lookup["right_elbow"]], frame_weights), 6),
        **{key: round(value, 6) for key, value in densepose_visibility.items()},
        **{key: round(value, 6) for key, value in densepose_coverage.items()},
    }


def _state_counts(states: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        counts[state] = counts.get(state, 0) + 1
    return counts


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def update_manifest_after_features(
    manifest_path: Path,
    *,
    metadata_path: Path,
    arrays_path: Path,
    summary: dict[str, Any],
) -> None:
    manifest = read_json(manifest_path)
    paths = manifest.setdefault("paths", {})
    paths["form_features"] = str(metadata_path)
    paths["form_feature_arrays"] = str(arrays_path)
    stages = manifest.setdefault("stages", {})
    stages["form_features"] = {
        "status": "complete",
        "recommended_tool": "Pose sequence + fused confidence feature compiler",
        "output": str(metadata_path),
        "arrays": str(arrays_path),
        "summary": summary,
    }
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def update_manifest_after_features_failure(manifest_path: Path, error: str) -> None:
    manifest = read_json(manifest_path)
    stage = manifest.setdefault("stages", {}).setdefault("form_features", {})
    stage["status"] = "failed"
    stage["error"] = error
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def compile_form_features(
    *,
    run_dir: Path,
    metadata_path: Path | None = None,
    arrays_path: Path | None = None,
    progress_callback: FormFeatureProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = read_json(manifest_path)
    paths = manifest.setdefault("paths", {})
    pose_path = Path(str(paths.get("pose_landmarks") or run_dir / "pose_landmarks.jsonl"))
    fused_path = Path(str(paths.get("fused_form") or run_dir / "fused_form.jsonl"))
    densepose_path = Path(str(paths.get("densepose") or run_dir / "densepose.jsonl"))
    source_path = Path(str(paths.get("source_segment") or run_dir / "source_segment.mp4"))
    metadata_path = metadata_path or Path(str(paths.get("form_features") or run_dir / "form_features.json"))
    arrays_path = arrays_path or Path(str(paths.get("form_feature_arrays") or run_dir / "form_features.npz"))

    try:
        if progress_callback:
            progress_callback(
                build_form_feature_progress(
                    phase="reading_inputs",
                    processed_frames=0,
                    total_frames=0,
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        pose_rows = read_jsonl(pose_path)
        fused_rows = read_jsonl(fused_path) if fused_path.exists() else []
        densepose_rows = read_jsonl(densepose_path) if densepose_path.exists() else []
        video_meta = inspect_video(source_path) if source_path.exists() else {}
        fps = float(video_meta.get("fps") or 0.0)
        duration = float(video_meta.get("duration_seconds") or manifest.get("review", {}).get("duration_seconds") or 0.0)

        raw_xy, pose_world, visibility, presence = _landmark_matrix(pose_rows)
        if progress_callback:
            progress_callback(
                build_form_feature_progress(
                    phase="compiling_features",
                    processed_frames=max(1, len(pose_rows) // 3),
                    total_frames=len(pose_rows),
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        normalized_xy, hip_mid, pose_scale = normalize_pose_xy(raw_xy)
        bones = bone_vectors(normalized_xy)
        angles = joint_angles(normalized_xy)
        velocities = angular_velocity(angles, fps)
        frame_weights, joint_weights, valid_frames, questionable_joints, frame_states = fused_weights(
            pose_rows, fused_rows
        )
        densepose_groups, densepose_group_visibility, part_centroids, group_centroids = densepose_arrays(
            pose_rows, densepose_rows, fused_rows
        )
        if progress_callback:
            progress_callback(
                build_form_feature_progress(
                    phase="summarizing_features",
                    processed_frames=max(1, (len(pose_rows) * 2) // 3),
                    total_frames=len(pose_rows),
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        summaries = summary_features(
            normalized_xy=normalized_xy,
            raw_xy=raw_xy,
            angles=angles,
            frame_weights=frame_weights,
            densepose_groups=densepose_groups,
            densepose_group_visibility=densepose_group_visibility,
            fps=fps,
        )
        usable_frame_count = int(valid_frames.sum())
        frame_count = len(pose_rows)
        quality = {
            "usable_rate": round(usable_frame_count / frame_count, 4) if frame_count else 0.0,
            "usable_frame_count": usable_frame_count,
            "frame_count": frame_count,
            "fused_confidence_mean": round(_weighted_mean(frame_weights, np.ones_like(frame_weights)), 6),
            "pose_visibility_mean": round(float(np.nanmean(visibility)), 6) if visibility.size else 0.0,
            "frame_states": _state_counts(frame_states),
        }
        arrays = {
            "pose_xy": raw_xy.astype(np.float32),
            "pose_xy_normalized": normalized_xy.astype(np.float32),
            "pose_world": pose_world.astype(np.float32),
            "pose_visibility": visibility.astype(np.float32),
            "pose_presence": presence.astype(np.float32),
            "joint_weights": joint_weights.astype(np.float32),
            "frame_weights": frame_weights.astype(np.float32),
            "valid_frames": valid_frames.astype(bool),
            "bone_vectors": bones.astype(np.float32),
            "joint_angles": angles.astype(np.float32),
            "angular_velocity": velocities.astype(np.float32),
            "densepose_groups": densepose_groups.astype(np.float32),
            "densepose_group_visibility": densepose_group_visibility.astype(np.float32),
            "densepose_part_centroids": part_centroids.astype(np.float32),
            "densepose_group_centroids": group_centroids.astype(np.float32),
            "hip_midpoint": hip_mid.astype(np.float32),
            "pose_scale": pose_scale.astype(np.float32),
            "time_seconds": np.asarray(
                [_as_float(row.get("time_seconds"), index / fps if fps else 0.0) for index, row in enumerate(pose_rows)],
                dtype=np.float32,
            ),
        }
        _write_npz(arrays_path, arrays)
        if progress_callback:
            progress_callback(
                build_form_feature_progress(
                    phase="writing_outputs",
                    processed_frames=len(pose_rows),
                    total_frames=len(pose_rows),
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        metadata = {
            "version": FEATURE_VERSION,
            "created_at": utc_now_iso(),
            "candidate_id": manifest.get("candidate_id"),
            "runner_name": manifest.get("runner_name"),
            "runner_slug": manifest.get("runner_slug"),
            "camera_angle": manifest.get("review", {}).get("camera_angle", "unknown"),
            "event_bucket": manifest.get("review", {}).get("primary_bucket", "running"),
            "reviewed_quality": manifest.get("review", {}).get("quality", ""),
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 4),
            "frame_count": frame_count,
            "usable_frame_count": usable_frame_count,
            "source_url": manifest.get("source", {}).get("url"),
            "matching_clip_path": str(source_path),
            "feature_files": {"arrays": str(arrays_path)},
            "array_schema": {
                "joint_names": LANDMARK_NAMES,
                "bone_names": list(BONE_NAMES),
                "angle_names": list(ANGLE_NAMES),
                "densepose_group_names": list(DENSEPOSE_GROUP_NAMES),
            },
            "quality": quality,
            "summary_features": summaries,
            "questionable_joints_by_frame": questionable_joints,
        }
        write_json(metadata_path, metadata)
        elapsed_seconds = round(time.monotonic() - started_at, 3)
        update_manifest_after_features(
            manifest_path,
            metadata_path=metadata_path,
            arrays_path=arrays_path,
            summary={
                "usable_rate": quality["usable_rate"],
                "fused_confidence_mean": quality["fused_confidence_mean"],
                "pose_visibility_mean": quality["pose_visibility_mean"],
            },
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "status": "complete",
            "frame_count": frame_count,
            "usable_frame_count": usable_frame_count,
            "elapsed_seconds": elapsed_seconds,
            "metadata": str(metadata_path),
            "arrays": str(arrays_path),
            "quality": quality,
            "summary_features": summaries,
        }
    except Exception as exc:
        update_manifest_after_features_failure(manifest_path, str(exc))
        raise


def load_feature_metadata(path: Path) -> dict[str, Any]:
    return read_json(path)


def iter_array_names(path: Path) -> Iterable[str]:
    with np.load(path) as data:
        yield from data.files
