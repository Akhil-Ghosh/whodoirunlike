from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam2_runner import inspect_video
from whodoirunlike.video_io import make_browser_playable_mp4


DENSEPOSE_SETUP_INSTRUCTIONS = (
    "DensePose is optional and is not installed in the base project. Install Detectron2 "
    "for your Python/PyTorch platform, then install or expose the Detectron2 "
    "projects/DensePose package. Example starting point: "
    "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'. "
    "Then run this adapter with --config path/to/densepose_rcnn_*.yaml and "
    "--weights path-or-url-to-densepose-model.pkl."
)


class DensePoseSetupError(RuntimeError):
    """Raised when optional DensePose runtime pieces are unavailable."""


DensePoseProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class DensePoseBackend:
    predictor: Any


@dataclass(frozen=True)
class DensePoseFrameOutput:
    row: dict[str, Any]
    labels: np.ndarray | None = None


def build_densepose_progress(
    *,
    phase: str,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
    frame_index: int | None = None,
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
    if usable is not None:
        payload["usable"] = bool(usable)
    return payload


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def update_manifest_densepose(
    manifest_path: Path,
    *,
    status: str,
    densepose_path: Path,
    frame_count: int = 0,
    usable_frames: int = 0,
    error: str | None = None,
    setup_instructions: str | None = None,
) -> None:
    run = RunningClipRun(manifest_path.parent)
    manifest = run.read_manifest()
    manifest["updated_at"] = utc_now_iso()
    densepose_stage = manifest.setdefault("stages", {}).setdefault("densepose", {})
    values: dict[str, Any] = {
        "status": status,
        "output": str(densepose_path),
        "frame_count": int(frame_count),
        "usable_frames": int(usable_frames),
    }
    if error:
        values["error"] = error
    else:
        densepose_stage.pop("error", None)
    if setup_instructions:
        values["setup_instructions"] = setup_instructions
    else:
        densepose_stage.pop("setup_instructions", None)
    run.update_stage("densepose", values, manifest)


def load_densepose_backend(
    *,
    config_path: Path | None = None,
    weights_path: str | None = None,
    confidence_threshold: float = 0.5,
    device: str = "cpu",
) -> DensePoseBackend:
    if config_path is None or weights_path in (None, ""):
        raise DensePoseSetupError(
            "DensePose config and weights are required. " + DENSEPOSE_SETUP_INSTRUCTIONS
        )

    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from densepose import add_densepose_config
    except ModuleNotFoundError as exc:
        raise DensePoseSetupError(DENSEPOSE_SETUP_INSTRUCTIONS) from exc

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(confidence_threshold)
    cfg.freeze()
    return DensePoseBackend(predictor=DefaultPredictor(cfg))


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def _box_xyxy_to_xywh(box: Any) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        int(round(x1)),
        int(round(y1)),
        max(1, int(round(x2 - x1))),
        max(1, int(round(y2 - y1))),
    ]


def _box_mask_overlap(box_xywh: list[int], mask: np.ndarray) -> float:
    x, y, width, height = box_xywh
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(mask.shape[1], x + width)
    y2 = min(mask.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((mask[y1:y2, x1:x2] > 0).sum()) / float(max(width * height, 1))


def _instances_to_cpu(instances: Any) -> Any:
    return instances.to("cpu") if hasattr(instances, "to") else instances


def _chart_result_for_instance(instances: Any, index: int) -> Any | None:
    densepose = getattr(instances, "pred_densepose", None)
    if densepose is None or not instances.has("pred_boxes"):
        return None
    try:
        import_module("densepose.converters.builtin")
        from densepose.converters import ToChartResultConverter

        return ToChartResultConverter.convert(densepose[index], instances.pred_boxes[index])
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
        return None


def _part_centroids(
    labels: np.ndarray,
    *,
    bbox: list[int] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, dict[str, float]]:
    dense_mask = labels > 0
    if not dense_mask.any():
        return {}
    label_height, label_width = labels.shape[:2]
    centroids: dict[str, dict[str, float]] = {}
    for part_id in np.unique(labels[dense_mask]):
        ys, xs = np.where(labels == part_id)
        if len(xs) == 0:
            continue
        local_x = float((xs.mean() + 0.5) / max(label_width, 1))
        local_y = float((ys.mean() + 0.5) / max(label_height, 1))
        payload = {
            "bbox_x": round(local_x, 6),
            "bbox_y": round(local_y, 6),
        }
        if bbox and frame_width and frame_height:
            x, y, width, height = [float(value) for value in bbox]
            payload["x"] = round((x + local_x * width) / max(frame_width, 1), 6)
            payload["y"] = round((y + local_y * height) / max(frame_height, 1), 6)
        else:
            payload["x"] = round(local_x, 6)
            payload["y"] = round(local_y, 6)
        centroids[str(int(part_id))] = payload
    return centroids


def _summarize_chart_result(
    chart_result: Any,
    *,
    bbox: list[int] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    labels = chart_result.labels.detach().cpu().numpy().astype("uint8")
    uv = chart_result.uv.detach().cpu().numpy().astype("float32")
    dense_mask = labels > 0
    part_ids, counts = np.unique(labels[dense_mask], return_counts=True)
    summary: dict[str, Any] = {
        "part_count": int(len(part_ids)),
        "part_ids": [int(part_id) for part_id in part_ids.tolist()],
        "part_pixels": {str(int(part_id)): int(count) for part_id, count in zip(part_ids, counts)},
        "part_centroids": _part_centroids(
            labels,
            bbox=bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        ),
        "densepose_shape": [int(labels.shape[1]), int(labels.shape[0])],
        "densepose_coverage": round(float(dense_mask.sum()) / float(max(labels.size, 1)), 4),
    }
    if dense_mask.any():
        summary["uv_mean"] = [
            round(float(uv[0][dense_mask].mean()), 4),
            round(float(uv[1][dense_mask].mean()), 4),
        ]
        summary["uv_std"] = [
            round(float(uv[0][dense_mask].std()), 4),
            round(float(uv[1][dense_mask].std()), 4),
        ]
    return summary


def apply_densepose_to_frame(
    frame_bgr: np.ndarray,
    runner_mask: np.ndarray,
    backend: DensePoseBackend,
    *,
    frame_index: int,
    min_mask_overlap: float = 0.1,
) -> DensePoseFrameOutput:
    masked_frame = frame_bgr.copy()
    masked_frame[runner_mask <= 0] = 0
    outputs = backend.predictor(masked_frame)
    instances = _instances_to_cpu(outputs.get("instances"))
    if instances is None or len(instances) == 0:
        return DensePoseFrameOutput({"usable": False, "drop_reason": "densepose_missing"})

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy() if hasattr(instances, "scores") else np.ones(len(instances))
    densepose = getattr(instances, "pred_densepose", None)

    best_index: int | None = None
    best_overlap = 0.0
    for index, raw_box in enumerate(boxes):
        bbox = _box_xyxy_to_xywh(raw_box)
        overlap = _box_mask_overlap(bbox, runner_mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    if best_index is None or best_overlap < min_mask_overlap:
        return DensePoseFrameOutput({"usable": False, "drop_reason": "no_detection_on_runner_mask"})

    row = {
        "usable": True,
        "score": round(float(scores[best_index]), 4),
        "bbox": _box_xyxy_to_xywh(boxes[best_index]),
        "mask_overlap": round(best_overlap, 4),
        "drop_reason": None,
    }
    chart_result = _chart_result_for_instance(instances, best_index) if densepose is not None else None
    labels = None
    if chart_result is not None:
        row.update(
            _summarize_chart_result(
                chart_result,
                bbox=row["bbox"],
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
            )
        )
        labels = chart_result.labels.detach().cpu().numpy().astype("uint8")
    else:
        row["part_count"] = None
        row["part_ids"] = []
    return DensePoseFrameOutput(row=row, labels=labels)


def _read_mask_frame(mask_capture: cv2.VideoCapture, width: int, height: int) -> np.ndarray | None:
    ok, mask_frame = mask_capture.read()
    if not ok:
        return None
    if mask_frame.shape[:2] != (height, width):
        mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    return (gray > 8).astype("uint8") * 255


def _blend_densepose_labels(
    overlay: np.ndarray,
    *,
    bbox: list[int],
    labels: np.ndarray,
    alpha: float = 0.58,
) -> np.ndarray:
    x, y, width, height = [int(value) for value in bbox]
    if width <= 0 or height <= 0:
        return overlay
    if labels.shape[:2] != (height, width):
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

    frame_height, frame_width = overlay.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + width)
    y2 = min(frame_height, y + height)
    if x2 <= x1 or y2 <= y1:
        return overlay

    label_crop = labels[y1 - y : y2 - y, x1 - x : x2 - x]
    dense_mask = label_crop > 0
    if not dense_mask.any():
        return overlay

    color_source = np.clip(label_crop.astype("float32") * (255.0 / 24.0), 0, 255).astype("uint8")
    color_map = cv2.applyColorMap(color_source, cv2.COLORMAP_JET)
    roi = overlay[y1:y2, x1:x2]
    roi[dense_mask] = (roi[dense_mask] * (1.0 - alpha) + color_map[dense_mask] * alpha).astype(
        "uint8"
    )
    overlay[y1:y2, x1:x2] = roi
    return overlay


def _draw_densepose_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    row: dict[str, Any],
    *,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    overlay = frame.copy()
    bbox = row.get("bbox")
    if row.get("usable") and bbox and labels is not None:
        overlay = _blend_densepose_labels(overlay, bbox=bbox, labels=labels)
    else:
        blue = np.zeros_like(frame)
        blue[:, :, 0] = 230
        overlay = np.where(
            (mask > 0)[:, :, None],
            (frame * 0.64 + blue * 0.36).astype("uint8"),
            overlay,
        )
    if row.get("usable") and bbox:
        x, y, width, height = [int(value) for value in bbox]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 245, 80), 2)
    return overlay


def run_densepose(
    *,
    run_dir: Path,
    config_path: Path | None = None,
    weights_path: str | None = None,
    confidence_threshold: float = 0.5,
    device: str = "cpu",
    write_qa_overlay: bool = True,
    progress_callback: DensePoseProgressCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    run = RunningClipRun(run_dir)
    manifest_path = run.manifest_path
    manifest = run.read_manifest()
    source_segment = run.artifact_path("source_segment", manifest)
    runner_mask = run.artifact_path("runner_mask", manifest)
    densepose_path = run.artifact_path("densepose", manifest)
    qa_overlay_path = run.artifact_path("qa_overlay", manifest)

    if progress_callback:
        progress_callback(
            build_densepose_progress(
                phase="loading_model",
                processed_frames=0,
                total_frames=0,
                elapsed_seconds=0.0,
            )
        )

    try:
        backend = load_densepose_backend(
            config_path=config_path,
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
    except DensePoseSetupError as exc:
        update_manifest_densepose(
            manifest_path,
            status="failed",
            densepose_path=densepose_path,
            error=str(exc),
            setup_instructions=DENSEPOSE_SETUP_INSTRUCTIONS,
        )
        return {
            "candidate_id": manifest.get("candidate_id"),
            "status": "failed",
            "error": str(exc),
            "setup_instructions": DENSEPOSE_SETUP_INSTRUCTIONS,
            "frame_count": 0,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }

    if not source_segment.exists():
        raise FileNotFoundError(f"Missing source_segment: {source_segment}")
    if not runner_mask.exists():
        raise FileNotFoundError(f"Missing runner_mask: {runner_mask}")

    if progress_callback:
        progress_callback(
            build_densepose_progress(
                phase="decoding",
                processed_frames=0,
                total_frames=0,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    video_meta = inspect_video(source_segment)
    width = int(video_meta["width"])
    height = int(video_meta["height"])
    fps = float(video_meta["fps"])
    total_frames = int(video_meta.get("frame_count") or 0)
    source_capture = cv2.VideoCapture(str(source_segment))
    mask_capture = cv2.VideoCapture(str(runner_mask))
    if not source_capture.isOpened():
        raise ValueError(f"Could not open source_segment: {source_segment}")
    if not mask_capture.isOpened():
        source_capture.release()
        raise ValueError(f"Could not open runner_mask: {runner_mask}")

    writer: cv2.VideoWriter | None = None
    if write_qa_overlay:
        qa_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(qa_overlay_path), fourcc, fps, (width, height), True)
        if not writer.isOpened():
            source_capture.release()
            mask_capture.release()
            raise ValueError(f"Could not open qa overlay writer: {qa_overlay_path}")

    if progress_callback:
        progress_callback(
            build_densepose_progress(
                phase="running_densepose",
                processed_frames=0,
                total_frames=total_frames,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    rows: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = source_capture.read()
            if not ok:
                break
            mask = _read_mask_frame(mask_capture, width, height)
            if mask is None:
                row = {"usable": False, "drop_reason": "runner_mask_frame_missing"}
            else:
                bbox = mask_bbox(mask)
                mask_area_ratio = float((mask > 0).sum()) / float(max(width * height, 1))
                if bbox is None:
                    row = {"usable": False, "drop_reason": "runner_mask_empty"}
                    labels = None
                else:
                    densepose_output = apply_densepose_to_frame(
                        frame,
                        mask,
                        backend,
                        frame_index=frame_index,
                    )
                    if isinstance(densepose_output, DensePoseFrameOutput):
                        row = densepose_output.row
                        labels = densepose_output.labels
                    else:
                        row = densepose_output
                        labels = None
                row["mask_area_ratio"] = round(mask_area_ratio, 6)
                row.setdefault("runner_bbox", bbox)
                if writer is not None:
                    writer.write(_draw_densepose_overlay(frame, mask, row, labels=labels))

            row["frame_index"] = frame_index
            row.setdefault("usable", False)
            row.setdefault("drop_reason", None if row["usable"] else "densepose_missing")
            rows.append(row)
            if progress_callback and (frame_index == 0 or (frame_index + 1) % 10 == 0):
                progress_callback(
                    build_densepose_progress(
                        phase="running_densepose",
                        processed_frames=frame_index + 1,
                        total_frames=total_frames,
                        elapsed_seconds=time.monotonic() - started_at,
                        frame_index=frame_index,
                        usable=bool(row.get("usable")),
                    )
                )
            frame_index += 1
    finally:
        source_capture.release()
        mask_capture.release()
        if writer is not None:
            writer.release()

    if writer is not None:
        if progress_callback:
            progress_callback(
                build_densepose_progress(
                    phase="encoding",
                    processed_frames=len(rows),
                    total_frames=total_frames,
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
        make_browser_playable_mp4(qa_overlay_path)

    usable_frames = sum(1 for row in rows if row.get("usable"))
    if progress_callback:
        progress_callback(
            build_densepose_progress(
                phase="writing_outputs",
                processed_frames=len(rows),
                total_frames=total_frames,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    jsonl_write(densepose_path, rows)
    update_manifest_densepose(
        manifest_path,
        status="complete",
        densepose_path=densepose_path,
        frame_count=len(rows),
        usable_frames=usable_frames,
    )
    if progress_callback:
        progress_callback(
            build_densepose_progress(
                phase="completed",
                processed_frames=len(rows),
                total_frames=total_frames,
                elapsed_seconds=time.monotonic() - started_at,
            )
        )
    return {
        "candidate_id": manifest.get("candidate_id"),
        "status": "complete",
        "frame_count": len(rows),
        "usable_frames": usable_frames,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "densepose": str(densepose_path),
        "qa_overlay": str(qa_overlay_path) if write_qa_overlay else None,
    }
