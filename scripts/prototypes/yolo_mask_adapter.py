#!/usr/bin/env python3
"""PROTOTYPE: compare a YOLO person mask adapter with an existing SAM runner mask."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def mask_scores(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    candidate_bool = candidate > 0
    baseline_bool = baseline > 0
    intersection = int(np.logical_and(candidate_bool, baseline_bool).sum())
    candidate_area = int(candidate_bool.sum())
    baseline_area = int(baseline_bool.sum())
    union = candidate_area + baseline_area - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (candidate_area + baseline_area)
        if candidate_area + baseline_area
        else 1.0,
        "precision": intersection / candidate_area if candidate_area else 0.0,
        "recall": intersection / baseline_area if baseline_area else 0.0,
        "area_ratio": candidate_area / max(candidate.size, 1),
    }


def target_boxes(
    tracklets: list[dict[str, Any]], width: int, height: int
) -> dict[int, tuple[np.ndarray, bool]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracklets:
        if row.get("is_target"):
            by_frame[int(row["frame_index"])].append(row)
    boxes: dict[int, tuple[np.ndarray, bool]] = {}
    for frame_index, rows in by_frame.items():
        row = max(rows, key=lambda item: float(item.get("detection_confidence") or 0.0))
        x1 = float(row["bbox_x"]) * width
        y1 = float(row["bbox_y"]) * height
        boxes[frame_index] = (
            np.asarray(
                [
                    x1,
                    y1,
                    x1 + float(row["bbox_width"]) * width,
                    y1 + float(row["bbox_height"]) * height,
                ],
                dtype=np.float32,
            ),
            bool(row.get("identity_risk")),
        )
    return boxes


def visible_pose_points(rows: list[dict[str, Any]]) -> dict[int, list[tuple[float, float]]]:
    points: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        frame_points = []
        for landmark in row.get("landmarks") or []:
            visibility = float(landmark.get("visibility") or landmark.get("score") or 0.0)
            if visibility >= 0.3 and not landmark.get("missing"):
                frame_points.append((float(landmark["x"]), float(landmark["y"])))
        points[int(row["frame_index"])] = frame_points
    return points


def points_inside(mask: np.ndarray, points: list[tuple[float, float]]) -> tuple[int, int]:
    height, width = mask.shape
    inside = 0
    for x, y in points:
        px = int(np.clip(round(x * (width - 1)), 0, width - 1))
        py = int(np.clip(round(y * (height - 1)), 0, height - 1))
        inside += int(mask[py, px] > 0)
    return inside, len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--segment-model", default="yolo11n-seg.pt")
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument(
        "--respect-identity-risk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source_path = run_dir / "source_segment.mp4"
    baseline_path = run_dir / "runner_mask.mp4"
    tracklets_path = run_dir / "tracklets.jsonl"
    pose_path = run_dir / "pose_landmarks.jsonl"
    capture = cv2.VideoCapture(str(source_path))
    baseline_capture = cv2.VideoCapture(str(baseline_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes = target_boxes(read_jsonl(tracklets_path), width, height)
    pose = visible_pose_points(read_jsonl(pose_path)) if pose_path.exists() else {}

    model_load_started = time.perf_counter()
    segmenter = YOLO(args.segment_model)
    detector = YOLO(args.detector_model) if args.detector_model else None
    model_load_seconds = time.perf_counter() - model_load_started
    kernel = None
    if args.dilate > 0:
        size = args.dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    frame_metrics: list[dict[str, float]] = []
    segment_wall_ms: list[float] = []
    detector_wall_ms: list[float] = []
    matched_frames = 0
    missing_frames = 0
    identity_risk_frames = 0
    output_nonempty_frames = 0
    baseline_nonempty_frames = 0
    pose_inside = 0
    pose_total = 0
    baseline_pose_inside = 0
    previous_mask: np.ndarray | None = None
    previous_baseline: np.ndarray | None = None
    temporal_ious: list[float] = []
    baseline_temporal_ious: list[float] = []

    try:
        for frame_index in range(frame_count):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            ok, frame = capture.read()
            baseline_ok, baseline_frame = baseline_capture.read()
            if not ok or not baseline_ok:
                break
            baseline = (cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY) > 20).astype(np.uint8)
            baseline_nonempty = bool(baseline.any())
            baseline_nonempty_frames += int(baseline_nonempty)
            baseline_inside, _ = points_inside(
                baseline, pose.get(frame_index, [])
            )
            baseline_pose_inside += baseline_inside
            if previous_baseline is not None:
                baseline_temporal_ious.append(
                    mask_scores(baseline, previous_baseline)["iou"]
                )
            previous_baseline = baseline
            target = boxes.get(frame_index)
            target_box = target[0] if target else None
            identity_risk = target[1] if target else False
            identity_risk_frames += int(identity_risk)
            started = time.perf_counter()
            result = segmenter.predict(
                frame,
                classes=[0],
                conf=0.25,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            segment_wall_ms.append((time.perf_counter() - started) * 1000)
            if detector is not None:
                started = time.perf_counter()
                detector.predict(
                    frame,
                    classes=[0],
                    conf=0.25,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                detector_wall_ms.append((time.perf_counter() - started) * 1000)

            if target_box is None or result.boxes is None or result.masks is None:
                missing_frames += 1
                continue
            result_boxes = result.boxes.xyxy.detach().cpu().numpy()
            if not len(result_boxes):
                missing_frames += 1
                continue
            overlaps = [box_iou(box, target_box) for box in result_boxes]
            selected_index = int(np.argmax(overlaps))
            if overlaps[selected_index] < 0.05:
                missing_frames += 1
                continue
            mask = result.masks.data[selected_index].detach().cpu().numpy()
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            if kernel is not None:
                mask = cv2.dilate(mask, kernel)
            if args.respect_identity_risk and identity_risk:
                mask.fill(0)
            metrics = mask_scores(mask, baseline)
            metrics["box_iou"] = overlaps[selected_index]
            metrics["baseline_nonempty"] = baseline_nonempty
            frame_metrics.append(metrics)
            matched_frames += 1
            output_nonempty_frames += int(bool(mask.any()))
            inside, total = points_inside(mask, pose.get(frame_index, []))
            pose_inside += inside
            pose_total += total
            if previous_mask is not None:
                temporal_ious.append(mask_scores(mask, previous_mask)["iou"])
            previous_mask = mask
    finally:
        capture.release()
        baseline_capture.release()

    def metric_summary(
        name: str, *, baseline_nonempty_only: bool = False
    ) -> dict[str, float | None]:
        values = [
            row[name]
            for row in frame_metrics
            if not baseline_nonempty_only or row["baseline_nonempty"]
        ]
        return {
            "mean": round(statistics.fmean(values), 4) if values else None,
            "p10": round(percentile(values, 0.10), 4) if values else None,
            "p50": round(percentile(values, 0.50), 4) if values else None,
        }

    processed = matched_frames + missing_frames
    summary = {
        "prototype": True,
        "question": "Can a tracked YOLO person mask preserve the current SAM output contract?",
        "runtime": {
            "opencv": cv2.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "torch": torch.__version__,
            "device": args.device,
        },
        "inputs": {
            "run_dir": str(run_dir),
            "segment_model": args.segment_model,
            "detector_model": args.detector_model or None,
            "frame_size": [width, height],
            "available_frames": frame_count,
            "processed_frames": processed,
            "imgsz": args.imgsz,
            "dilate_pixels": args.dilate,
            "respect_identity_risk": args.respect_identity_risk,
        },
        "identity": {
            "matched_frames": matched_frames,
            "missing_frames": missing_frames,
            "retention": round(matched_frames / processed, 4) if processed else 0.0,
            "identity_risk_frames": identity_risk_frames,
            "output_nonempty_frames": output_nonempty_frames,
            "baseline_nonempty_frames": baseline_nonempty_frames,
            "track_box_iou": metric_summary("box_iou"),
        },
        "mask_vs_sam_baseline": {
            name: metric_summary(name, baseline_nonempty_only=True)
            for name in ("iou", "dice", "precision", "recall", "area_ratio")
        },
        "temporal": {
            "mean_consecutive_iou": round(statistics.fmean(temporal_ious), 4)
            if temporal_ious
            else None,
            "p10_consecutive_iou": round(percentile(temporal_ious, 0.10), 4)
            if temporal_ious
            else None,
            "sam_mean_consecutive_iou": round(
                statistics.fmean(baseline_temporal_ious), 4
            )
            if baseline_temporal_ious
            else None,
        },
        "downstream_proxy": {
            "visible_pose_keypoints_inside_mask": round(pose_inside / pose_total, 4)
            if pose_total
            else None,
            "visible_pose_keypoint_samples": pose_total,
            "sam_visible_pose_keypoints_inside_mask": round(
                baseline_pose_inside / pose_total, 4
            )
            if pose_total
            else None,
            "inside_mask_rate_delta": round(
                (pose_inside - baseline_pose_inside) / pose_total, 4
            )
            if pose_total
            else None,
        },
        "latency": {
            "model_load_seconds": round(model_load_seconds, 3),
            "segment_wall_ms_per_frame_p50": round(percentile(segment_wall_ms, 0.50), 3)
            if segment_wall_ms
            else None,
            "segment_wall_ms_per_frame_p95": round(percentile(segment_wall_ms, 0.95), 3)
            if segment_wall_ms
            else None,
            "detector_wall_ms_per_frame_p50": round(percentile(detector_wall_ms, 0.50), 3)
            if detector_wall_ms
            else None,
            "detector_wall_ms_per_frame_p95": round(percentile(detector_wall_ms, 0.95), 3)
            if detector_wall_ms
            else None,
            "segmentation_incremental_p50_ms": round(
                (percentile(segment_wall_ms, 0.50) or 0.0)
                - (percentile(detector_wall_ms, 0.50) or 0.0),
                3,
            )
            if detector_wall_ms
            else None,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
