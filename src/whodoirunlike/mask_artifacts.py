from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    binary = (mask > 0).astype("uint8")
    height, width = binary.shape[:2]
    pixels = binary.flatten(order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for value in pixels:
        value = int(value)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current = value
    counts.append(run_length)
    return {"size": [height, width], "counts": counts}


def mask_bbox(mask: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    height, width = mask.shape[:2]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return {
        "x": round(x1 / max(width, 1), 6),
        "y": round(y1 / max(height, 1), 6),
        "width": round((x2 - x1) / max(width, 1), 6),
        "height": round((y2 - y1) / max(height, 1), 6),
    }


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union else 1.0


def iter_mask_video(mask_video_path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(mask_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open mask video: {mask_video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((gray > 20).astype("uint8"))
    cap.release()
    return {"fps": fps, "width": width, "height": height, "frame_count": len(frames)}, frames


def mask_rows_from_video(mask_video_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta, frames = iter_mask_video(mask_video_path)
    rows: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    for frame_index, mask in enumerate(frames):
        area = int(mask.sum())
        area_ratio = area / float(max(mask.shape[0] * mask.shape[1], 1))
        churn_iou = mask_iou(previous, mask) if previous is not None else None
        rows.append(
            {
                "frame_index": frame_index,
                "time_seconds": round(frame_index / meta["fps"], 6) if meta["fps"] else None,
                "width": meta["width"],
                "height": meta["height"],
                "area": area,
                "area_ratio": round(area_ratio, 8),
                "bbox": mask_bbox(mask),
                "temporal_iou_prev": round(churn_iou, 6) if churn_iou is not None else None,
                "rle": encode_uncompressed_rle(mask),
            }
        )
        previous = mask
    return meta, rows


def write_masks_jsonl_from_video(mask_video_path: Path, output_path: Path) -> dict[str, Any]:
    meta, rows = mask_rows_from_video(mask_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporal_ious = [
        float(row["temporal_iou_prev"])
        for row in rows
        if row.get("temporal_iou_prev") is not None
    ]
    return {
        **meta,
        "output_path": str(output_path),
        "mean_temporal_iou": round(float(np.mean(temporal_ious)), 6) if temporal_ious else None,
        "mean_mask_churn": round(1.0 - float(np.mean(temporal_ious)), 6) if temporal_ious else None,
        "nonempty_frames": sum(1 for row in rows if int(row["area"]) > 0),
    }
