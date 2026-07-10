from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.pose_runner import POSE_CONNECTIONS
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam2_runner import inspect_video
from whodoirunlike.video_io import make_browser_playable_mp4


FusionProgressCallback = Callable[[dict[str, Any]], None]

DENSEPOSE_PART_GROUPS = {
    "torso": {1, 2},
    "hands": {3, 4},
    "feet": {5, 6},
    "upper_legs": {7, 8, 9, 10},
    "lower_legs": {11, 12, 13, 14},
    "upper_arms": {15, 16, 17, 18},
    "lower_arms": {19, 20, 21, 22},
    "head": {23, 24},
}

EXPECTED_DENSEPOSE_GROUPS = {
    "nose": {"head"},
    "left_eye_inner": {"head"},
    "left_eye": {"head"},
    "left_eye_outer": {"head"},
    "right_eye_inner": {"head"},
    "right_eye": {"head"},
    "right_eye_outer": {"head"},
    "left_ear": {"head"},
    "right_ear": {"head"},
    "mouth_left": {"head"},
    "mouth_right": {"head"},
    "left_shoulder": {"torso", "upper_arms"},
    "right_shoulder": {"torso", "upper_arms"},
    "left_elbow": {"upper_arms", "lower_arms"},
    "right_elbow": {"upper_arms", "lower_arms"},
    "left_wrist": {"lower_arms", "hands"},
    "right_wrist": {"lower_arms", "hands"},
    "left_pinky": {"hands", "lower_arms"},
    "right_pinky": {"hands", "lower_arms"},
    "left_index": {"hands", "lower_arms"},
    "right_index": {"hands", "lower_arms"},
    "left_thumb": {"hands", "lower_arms"},
    "right_thumb": {"hands", "lower_arms"},
    "left_hip": {"torso", "upper_legs"},
    "right_hip": {"torso", "upper_legs"},
    "left_knee": {"upper_legs", "lower_legs"},
    "right_knee": {"upper_legs", "lower_legs"},
    "left_ankle": {"lower_legs", "feet"},
    "right_ankle": {"lower_legs", "feet"},
    "left_heel": {"feet", "lower_legs"},
    "right_heel": {"feet", "lower_legs"},
    "left_foot_index": {"feet", "lower_legs"},
    "right_foot_index": {"feet", "lower_legs"},
}

KEY_JOINTS = {
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
    "left_foot_index",
    "right_foot_index",
}


def build_fusion_progress(
    *,
    phase: str,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
    frame_index: int | None = None,
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
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL artifact: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _clamp_unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _landmark_xy(landmark: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    x = int(round(_clamp_unit(landmark.get("x")) * max(width - 1, 1)))
    y = int(round(_clamp_unit(landmark.get("y")) * max(height - 1, 1)))
    return x, y


def _mask_from_capture(mask_capture: cv2.VideoCapture | None, width: int, height: int) -> np.ndarray | None:
    if mask_capture is None:
        return None
    ok, mask_frame = mask_capture.read()
    if not ok:
        return None
    if mask_frame.shape[:2] != (height, width):
        mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    return (gray > 20).astype("uint8") * 255


def _sample_mask(mask: np.ndarray | None, point: tuple[int, int]) -> bool:
    if mask is None:
        return False
    x, y = point
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x] > 0)


def _box_iou_pixels(a: Sequence[int] | None, b: Sequence[int] | None) -> float:
    if not a or not b:
        return 0.0
    ax, ay, aw, ah = [float(value) for value in a]
    bx, by, bw, bh = [float(value) for value in b]
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def _mask_bbox_pixels(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    x = int(xs.min())
    y = int(ys.min())
    return [x, y, int(xs.max()) + 1 - x, int(ys.max()) + 1 - y]


def _mask_area_ratio(mask: np.ndarray | None, width: int, height: int) -> float:
    if mask is None:
        return 0.0
    return float((mask > 0).sum()) / float(max(width * height, 1))


def _pose_bbox_pixels(pose_row: dict[str, Any], width: int, height: int) -> list[int] | None:
    bbox = pose_row.get("bbox")
    if not isinstance(bbox, dict):
        return None
    x = int(round(_clamp_unit(bbox.get("x")) * width))
    y = int(round(_clamp_unit(bbox.get("y")) * height))
    w = int(round(_clamp_unit(bbox.get("width")) * width))
    h = int(round(_clamp_unit(bbox.get("height")) * height))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def densepose_group_coverage(densepose_row: dict[str, Any]) -> dict[str, float]:
    part_pixels = densepose_row.get("part_pixels") or {}
    if not isinstance(part_pixels, dict):
        return {group: 0.0 for group in DENSEPOSE_PART_GROUPS}
    counts = {int(key): int(value) for key, value in part_pixels.items()}
    total = max(1, sum(counts.values()))
    return {
        group: round(sum(counts.get(part_id, 0) for part_id in part_ids) / total, 4)
        for group, part_ids in DENSEPOSE_PART_GROUPS.items()
    }


def _group_confidence(landmark_name: str, coverage: dict[str, float]) -> float:
    groups = EXPECTED_DENSEPOSE_GROUPS.get(landmark_name, set())
    if not groups:
        return 0.5
    best = max((coverage.get(group, 0.0) for group in groups), default=0.0)
    return _clamp_score(best / 0.035)


def _densepose_confidence(row: dict[str, Any]) -> float:
    if not row or not row.get("usable"):
        return 0.0
    score = float(row.get("score") or 0.0)
    mask_overlap = float(row.get("mask_overlap") or 0.0)
    coverage = float(row.get("densepose_coverage") or 0.0)
    part_count = min(float(row.get("part_count") or 0.0) / 18.0, 1.0)
    return _clamp_score((score * 0.25) + (min(mask_overlap / 0.25, 1.0) * 0.25) + (min(coverage / 0.22, 1.0) * 0.3) + (part_count * 0.2))


def fuse_frame(
    pose_row: dict[str, Any],
    densepose_row: dict[str, Any],
    *,
    mask: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    coverage = densepose_group_coverage(densepose_row)
    landmarks = pose_row.get("landmarks") or []
    joint_weights: list[dict[str, Any]] = []
    for landmark in landmarks:
        name = str(landmark.get("name") or "")
        point = _landmark_xy(landmark, width, height)
        visibility = float(landmark.get("visibility") or 0.0)
        presence = float(landmark.get("presence") or visibility)
        pose_confidence = _clamp_score((visibility + presence) / 2.0)
        inside_mask = _sample_mask(mask, point)
        densepose_confidence = _group_confidence(name, coverage)
        weight = _clamp_score((pose_confidence * 0.58) + ((1.0 if inside_mask else 0.0) * 0.22) + (densepose_confidence * 0.2))
        joint_weights.append(
            {
                "index": int(landmark.get("index") or 0),
                "name": name,
                "weight": weight,
                "pose_confidence": pose_confidence,
                "inside_runner_mask": inside_mask,
                "densepose_group_confidence": densepose_confidence,
                "expected_groups": sorted(EXPECTED_DENSEPOSE_GROUPS.get(name, set())),
                "x": round(point[0] / max(width - 1, 1), 6),
                "y": round(point[1] / max(height - 1, 1), 6),
            }
        )

    key_weights = [joint["weight"] for joint in joint_weights if joint["name"] in KEY_JOINTS]
    key_joint_confidence = mean(key_weights) if key_weights else 0.0
    pose_confidence = float(pose_row.get("visibility_mean") or 0.0)
    pose_reliable = pose_confidence >= 0.18 or key_joint_confidence >= 0.25
    densepose_confidence = _densepose_confidence(densepose_row)
    densepose_usable = bool(densepose_row.get("usable"))
    mask_area_ratio = _mask_area_ratio(mask, width, height)
    mask_bbox = _mask_bbox_pixels(mask)
    pose_bbox = _pose_bbox_pixels(pose_row, width, height)
    identity_bbox = densepose_row.get("runner_bbox") if densepose_usable else mask_bbox
    pose_mask_iou = _box_iou_pixels(
        pose_bbox,
        identity_bbox,
    )
    if densepose_usable:
        mask_confidence = _clamp_score(
            (pose_mask_iou * 0.55)
            + (min(float(densepose_row.get("mask_overlap") or 0.0) / 0.25, 1.0) * 0.45)
        )
        frame_confidence = _clamp_score(
            (pose_confidence * 0.42)
            + (key_joint_confidence * 0.24)
            + (densepose_confidence * 0.22)
            + (mask_confidence * 0.12)
        )
    else:
        plausible_mask_size = 0.006 <= mask_area_ratio <= 0.45
        mask_confidence = _clamp_score((pose_mask_iou * 0.7) + ((0.3 if plausible_mask_size else 0.0)))
        pose_mask_confidence = _clamp_score(
            (pose_confidence * 0.32)
            + (key_joint_confidence * 0.28)
            + (mask_confidence * 0.4)
        )
        mask_only_confidence = _clamp_score(mask_confidence * 0.85)
        frame_confidence = max(pose_mask_confidence, mask_only_confidence) if pose_reliable else mask_only_confidence
    questionable = [
        joint["name"]
        for joint in joint_weights
        if joint["name"] in KEY_JOINTS and (joint["weight"] < 0.52 or not joint["inside_runner_mask"])
    ]
    if not pose_row.get("usable"):
        frame_state = "pose_rejected"
    elif not densepose_usable and mask_bbox is not None and pose_reliable and frame_confidence >= 0.25:
        frame_state = "pose_mask_fallback"
    elif not densepose_usable and mask_bbox is not None and plausible_mask_size and frame_confidence >= 0.3:
        frame_state = "target_mask_fallback"
    elif not densepose_usable:
        frame_state = "densepose_missing"
    elif pose_mask_iou < 0.08:
        frame_state = "identity_risk"
    elif frame_confidence < 0.45:
        frame_state = "short_occlusion"
    else:
        frame_state = "usable"
    return {
        "frame_index": int(pose_row.get("frame_index") or densepose_row.get("frame_index") or 0),
        "time_seconds": pose_row.get("time_seconds"),
        "frame_state": frame_state,
        "usable": frame_state in {"usable", "pose_mask_fallback", "target_mask_fallback"},
        "frame_confidence": frame_confidence,
        "pose_confidence": round(pose_confidence, 4),
        "key_joint_confidence": round(key_joint_confidence, 4),
        "pose_reliable": pose_reliable,
        "mask_confidence": mask_confidence,
        "mask_area_ratio": round(mask_area_ratio, 6),
        "densepose_confidence": densepose_confidence,
        "pose_mask_iou": round(pose_mask_iou, 4),
        "densepose_group_coverage": coverage,
        "questionable_joints": questionable,
        "joint_weights": joint_weights,
    }


def _draw_mask_edge(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    output = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (245, 245, 236), 2, lineType=cv2.LINE_AA)
    return output


def _draw_fused_overlay(frame: np.ndarray, pose_row: dict[str, Any], fused_row: dict[str, Any], mask: np.ndarray | None) -> np.ndarray:
    output = _draw_mask_edge(frame, mask)
    height, width = output.shape[:2]
    landmarks = pose_row.get("landmarks") or []
    weights = {joint["index"]: joint for joint in fused_row.get("joint_weights", [])}
    frame_state = str(fused_row.get("frame_state") or "")

    if frame_state in {"target_mask_fallback", "densepose_missing", "pose_rejected"}:
        confidence = int(round(float(fused_row.get("frame_confidence") or 0.0) * 100))
        badge = f"Fused confidence {confidence}% | {frame_state.replace('_', ' ')}"
        cv2.rectangle(output, (28, 26), (520, 76), (22, 25, 31), -1)
        cv2.putText(output, badge, (44, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 242, 233), 2, cv2.LINE_AA)
        return output

    for start, end in POSE_CONNECTIONS:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        p1 = _landmark_xy(landmarks[start], width, height)
        p2 = _landmark_xy(landmarks[end], width, height)
        start_weight = float(weights.get(start, {}).get("weight", 0.0))
        end_weight = float(weights.get(end, {}).get("weight", 0.0))
        color = (65, 210, 130) if min(start_weight, end_weight) >= 0.58 else (45, 190, 245)
        if min(start_weight, end_weight) < 0.42:
            color = (80, 80, 245)
        cv2.line(output, p1, p2, color, 3, lineType=cv2.LINE_AA)

    for landmark in landmarks:
        index = int(landmark.get("index") or 0)
        point = _landmark_xy(landmark, width, height)
        weight = float(weights.get(index, {}).get("weight", 0.0))
        color = (64, 214, 117) if weight >= 0.58 else (49, 184, 244)
        radius = 5 if str(landmark.get("name")) in KEY_JOINTS else 3
        if weight < 0.42:
            color = (72, 72, 238)
            radius += 2
        cv2.circle(output, point, radius, (20, 24, 31), -1, lineType=cv2.LINE_AA)
        cv2.circle(output, point, radius, color, 2, lineType=cv2.LINE_AA)

    confidence = int(round(float(fused_row.get("frame_confidence") or 0.0) * 100))
    badge = f"Fused confidence {confidence}% | {str(fused_row.get('frame_state') or '').replace('_', ' ')}"
    cv2.rectangle(output, (28, 26), (520, 76), (22, 25, 31), -1)
    cv2.putText(output, badge, (44, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 242, 233), 2, cv2.LINE_AA)
    return output


def _resolve_path(paths: dict[str, Any], key: str, fallback: Path) -> Path:
    value = paths.get(key)
    return Path(str(value)) if value else fallback


def update_manifest_after_fusion(
    manifest_path: Path,
    *,
    fused_form_path: Path,
    fused_overlay_path: Path,
    summary: dict[str, Any],
) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    paths = manifest.setdefault("paths", {})
    paths["fused_form"] = str(fused_form_path)
    paths["fused_overlay"] = str(fused_overlay_path)
    stages = manifest.setdefault("stages", {})
    stages.setdefault("fused_form", {}).pop("error", None)
    outputs = list(manifest.get("stages", {}).get("renders", {}).get("outputs") or [])
    if str(fused_overlay_path) not in outputs:
        outputs.append(str(fused_overlay_path))
    run.update_stages(
        {
            "fused_form": {
                "status": "complete",
                "recommended_tool": "MediaPipe + SAM mask + DensePose fusion",
                "output": str(fused_form_path),
                "overlay": str(fused_overlay_path),
                "summary": summary,
            },
            "renders": {"outputs": outputs},
        },
        manifest,
    )


def update_manifest_after_fusion_failure(manifest_path: Path, error: str) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    run.update_stage("fused_form", {"status": "failed", "error": error}, manifest)


def summarize_fusion(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "frame_count": 0,
            "usable_frames": 0,
            "usable_rate": 0.0,
            "confidence_mean": 0.0,
        }
    usable_frames = sum(1 for row in rows if row.get("usable"))
    confidence_values = [float(row.get("frame_confidence") or 0.0) for row in rows]
    state_counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("frame_state") or "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    return {
        "frame_count": len(rows),
        "usable_frames": usable_frames,
        "usable_rate": round(usable_frames / len(rows), 4),
        "confidence_mean": round(mean(confidence_values), 4),
        "frame_states": state_counts,
    }


def run_fused_form(
    *,
    run_dir: Path,
    progress_callback: FusionProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    run = RunningClipRun(run_dir)
    manifest_path = run.manifest_path
    manifest = run.read_manifest()

    source_segment = run.artifact_path("source_segment", manifest)
    pose_path = run.artifact_path("pose_landmarks", manifest)
    densepose_path = run.artifact_path("densepose", manifest)
    runner_mask = run.artifact_path("runner_mask", manifest)
    base_overlay = run.artifact_path("qa_overlay", manifest)
    if not base_overlay.exists():
        base_overlay = source_segment
    fused_form_path = run.artifact_path("fused_form", manifest)
    fused_overlay_path = run.artifact_path("fused_overlay", manifest)

    pose_rows = read_jsonl(pose_path)
    densepose_rows = read_jsonl(densepose_path)
    densepose_by_frame = {int(row.get("frame_index") or 0): row for row in densepose_rows}
    video_meta = inspect_video(source_segment)
    width = int(video_meta["width"])
    height = int(video_meta["height"])
    fps = float(video_meta["fps"])
    total_frames = len(pose_rows)

    if progress_callback:
        progress_callback(
            build_fusion_progress(
                phase="fusing_form",
                processed_frames=0,
                total_frames=total_frames,
                elapsed_seconds=0.0,
            )
        )

    base_capture = cv2.VideoCapture(str(base_overlay if base_overlay.exists() else source_segment))
    mask_capture = cv2.VideoCapture(str(runner_mask)) if runner_mask.exists() else None
    if not base_capture.isOpened():
        raise ValueError(f"Could not open fusion base video: {base_overlay}")
    if mask_capture is not None and not mask_capture.isOpened():
        base_capture.release()
        raise ValueError(f"Could not open runner mask: {runner_mask}")

    fused_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(fused_overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        base_capture.release()
        if mask_capture is not None:
            mask_capture.release()
        raise ValueError(f"Could not open fused overlay writer: {fused_overlay_path}")

    fused_rows: list[dict[str, Any]] = []
    try:
        for index, pose_row in enumerate(pose_rows):
            ok, frame = base_capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = _mask_from_capture(mask_capture, width, height)
            frame_index = int(pose_row.get("frame_index") or index)
            densepose_row = densepose_by_frame.get(frame_index, {"frame_index": frame_index, "usable": False})
            fused_row = fuse_frame(
                pose_row,
                densepose_row,
                mask=mask,
                width=width,
                height=height,
            )
            fused_rows.append(fused_row)
            writer.write(_draw_fused_overlay(frame, pose_row, fused_row, mask))
            if progress_callback and (index == 0 or (index + 1) % 10 == 0):
                progress_callback(
                    build_fusion_progress(
                        phase="fusing_form",
                        processed_frames=index + 1,
                        total_frames=total_frames,
                        elapsed_seconds=time.monotonic() - started_at,
                        frame_index=frame_index,
                    )
                )
    finally:
        base_capture.release()
        if mask_capture is not None:
            mask_capture.release()
        writer.release()

    make_browser_playable_mp4(fused_overlay_path)
    write_jsonl(fused_form_path, fused_rows)
    summary = summarize_fusion(fused_rows)
    update_manifest_after_fusion(
        manifest_path,
        fused_form_path=fused_form_path,
        fused_overlay_path=fused_overlay_path,
        summary=summary,
    )
    return {
        "candidate_id": manifest.get("candidate_id"),
        "status": "complete",
        "frame_count": len(fused_rows),
        "usable_frames": summary["usable_frames"],
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "summary": summary,
        "fused_form": str(fused_form_path),
        "fused_overlay": str(fused_overlay_path),
    }
