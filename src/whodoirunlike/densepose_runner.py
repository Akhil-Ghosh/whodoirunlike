from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Sequence

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
DENSEPOSE_BATCH_SIZE_ENV = "DENSEPOSE_BATCH_SIZE"
DEFAULT_DENSEPOSE_BATCH_SIZE = 1
MAX_DENSEPOSE_BATCH_SIZE = 8


class DensePoseSetupError(RuntimeError):
    """Raised when optional DensePose runtime pieces are unavailable."""


DensePoseProgressCallback = Callable[[dict[str, Any]], None]
DensePoseBenchmarkEvidenceCallback = Callable[
    [int, dict[str, Any], np.ndarray | None],
    None,
]


@dataclass(frozen=True)
class DensePoseBackend:
    predictor: Any
    input_min_size_test: Any = None
    input_max_size_test: Any = None
    inference_lock: threading.RLock = field(
        default_factory=threading.RLock,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class DensePoseFrameOutput:
    row: dict[str, Any]
    labels: np.ndarray | None = None


@dataclass(frozen=True)
class _DensePosePreparedFrame:
    frame_bgr: np.ndarray
    runner_mask: np.ndarray
    inference_frame: np.ndarray
    inference_input: dict[str, Any]
    crop_x: int
    crop_y: int


_DENSEPOSE_BACKEND_CACHE: dict[tuple[Any, ...], DensePoseBackend] = {}
_DENSEPOSE_BACKEND_CACHE_LOCK = threading.RLock()


def build_densepose_progress(
    *,
    phase: str,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
    frame_index: int | None = None,
    usable: bool | None = None,
    inference_input: dict[str, Any] | None = None,
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
    if inference_input is not None:
        payload["inference_input"] = dict(inference_input)
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
    inference_settings: dict[str, Any] | None = None,
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
    if inference_settings is not None:
        values["inference_settings"] = dict(inference_settings)
    else:
        densepose_stage.pop("inference_settings", None)
    run.update_stage("densepose", values, manifest)


def load_densepose_backend(
    *,
    config_path: Path | None = None,
    weights_path: str | None = None,
    confidence_threshold: float = 0.5,
    device: str = "cpu",
    input_min_size_test: int | None = None,
    input_max_size_test: int | None = None,
    cache_enabled: bool = True,
) -> DensePoseBackend:
    if config_path is None or weights_path in (None, ""):
        raise DensePoseSetupError(
            "DensePose config and weights are required. " + DENSEPOSE_SETUP_INSTRUCTIONS
        )

    min_size_override = int(input_min_size_test) if input_min_size_test is not None else None
    max_size_override = int(input_max_size_test) if input_max_size_test is not None else None
    if min_size_override is not None and min_size_override <= 0:
        raise ValueError("DensePose INPUT.MIN_SIZE_TEST override must be positive")
    if max_size_override is not None and max_size_override <= 0:
        raise ValueError("DensePose INPUT.MAX_SIZE_TEST override must be positive")

    key = (
        str(config_path.resolve(strict=False)),
        str(weights_path),
        float(confidence_threshold),
        str(device),
        min_size_override,
        max_size_override,
    )
    with _DENSEPOSE_BACKEND_CACHE_LOCK:
        if cache_enabled and key in _DENSEPOSE_BACKEND_CACHE:
            return _DENSEPOSE_BACKEND_CACHE[key]

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
        if min_size_override is not None:
            cfg.INPUT.MIN_SIZE_TEST = min_size_override
        if max_size_override is not None:
            cfg.INPUT.MAX_SIZE_TEST = max_size_override
        effective_min_size = cfg.INPUT.MIN_SIZE_TEST
        effective_max_size = cfg.INPUT.MAX_SIZE_TEST
        cfg.freeze()
        backend = DensePoseBackend(
            predictor=DefaultPredictor(cfg),
            input_min_size_test=effective_min_size,
            input_max_size_test=effective_max_size,
        )
        if cache_enabled:
            _DENSEPOSE_BACKEND_CACHE[key] = backend
        return backend


def clear_densepose_backend_cache() -> None:
    with _DENSEPOSE_BACKEND_CACHE_LOCK:
        _DENSEPOSE_BACKEND_CACHE.clear()


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def _target_crop_bbox(
    runner_mask: np.ndarray,
    *,
    runner_bbox: list[int],
    padding_ratio: float,
    padding_pixels: int,
) -> list[int]:
    x, y, width, height = runner_bbox
    ratio_padding = int(round(max(width, height) * max(0.0, float(padding_ratio))))
    padding = max(0, int(padding_pixels), ratio_padding)
    frame_height, frame_width = runner_mask.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame_width, x + width + padding)
    y2 = min(frame_height, y + height + padding)
    return [x1, y1, x2 - x1, y2 - y1]


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


def _prepare_densepose_frame(
    frame_bgr: np.ndarray,
    runner_mask: np.ndarray,
    *,
    target_crop_enabled: bool,
    target_crop_padding_ratio: float,
    target_crop_padding_pixels: int,
) -> _DensePosePreparedFrame | DensePoseFrameOutput:
    frame_height, frame_width = frame_bgr.shape[:2]
    runner_bbox = mask_bbox(runner_mask)
    if runner_bbox is None:
        return DensePoseFrameOutput(
            {
                "usable": False,
                "drop_reason": "runner_mask_empty",
                "inference_input": {
                    "target_crop_enabled": bool(target_crop_enabled),
                    "crop_bbox": None,
                    "width": 0,
                    "height": 0,
                },
            }
        )

    crop_bbox = [0, 0, frame_width, frame_height]
    if target_crop_enabled:
        crop_bbox = _target_crop_bbox(
            runner_mask,
            runner_bbox=runner_bbox,
            padding_ratio=target_crop_padding_ratio,
            padding_pixels=target_crop_padding_pixels,
        )
    crop_x, crop_y, crop_width, crop_height = crop_bbox
    inference_frame = frame_bgr[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ].copy()
    inference_mask = runner_mask[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    inference_frame[inference_mask <= 0] = 0
    return _DensePosePreparedFrame(
        frame_bgr=frame_bgr,
        runner_mask=runner_mask,
        inference_frame=inference_frame,
        inference_input={
            "target_crop_enabled": bool(target_crop_enabled),
            "crop_bbox": crop_bbox,
            "width": int(inference_frame.shape[1]),
            "height": int(inference_frame.shape[0]),
        },
        crop_x=crop_x,
        crop_y=crop_y,
    )


def _densepose_output_from_predictions(
    prepared: _DensePosePreparedFrame,
    outputs: dict[str, Any],
    *,
    min_mask_overlap: float,
) -> DensePoseFrameOutput:
    instances = _instances_to_cpu(outputs.get("instances"))
    if instances is None or len(instances) == 0:
        return DensePoseFrameOutput(
            {
                "usable": False,
                "drop_reason": "densepose_missing",
                "inference_input": prepared.inference_input,
            }
        )

    local_boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy() if hasattr(instances, "scores") else np.ones(len(instances))
    densepose = getattr(instances, "pred_densepose", None)

    best_index: int | None = None
    best_overlap = 0.0
    for index, raw_box in enumerate(local_boxes):
        full_frame_box = np.asarray(raw_box, dtype=np.float32).copy()
        full_frame_box[[0, 2]] += prepared.crop_x
        full_frame_box[[1, 3]] += prepared.crop_y
        bbox = _box_xyxy_to_xywh(full_frame_box)
        overlap = _box_mask_overlap(bbox, prepared.runner_mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    if best_index is None or best_overlap < min_mask_overlap:
        return DensePoseFrameOutput(
            {
                "usable": False,
                "drop_reason": "no_detection_on_runner_mask",
                "inference_input": prepared.inference_input,
            }
        )

    selected_box = np.asarray(local_boxes[best_index], dtype=np.float32).copy()
    selected_box[[0, 2]] += prepared.crop_x
    selected_box[[1, 3]] += prepared.crop_y
    row = {
        "usable": True,
        "score": round(float(scores[best_index]), 4),
        "bbox": _box_xyxy_to_xywh(selected_box),
        "mask_overlap": round(best_overlap, 4),
        "drop_reason": None,
        "inference_input": prepared.inference_input,
    }
    chart_result = (
        _chart_result_for_instance(instances, best_index) if densepose is not None else None
    )
    labels = None
    if chart_result is not None:
        row.update(
            _summarize_chart_result(
                chart_result,
                bbox=row["bbox"],
                frame_width=prepared.frame_bgr.shape[1],
                frame_height=prepared.frame_bgr.shape[0],
            )
        )
        labels = chart_result.labels.detach().cpu().numpy().astype("uint8")
    else:
        row["part_count"] = None
        row["part_ids"] = []
    return DensePoseFrameOutput(row=row, labels=labels)


def _default_predictor_batch(
    predictor: Any,
    images: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    """Run DefaultPredictor's exact augmentation/input contract as one model call."""
    import torch

    batched_inputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for source_image in images:
            original_image = source_image
            if predictor.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # Preserve DefaultPredictor's exact call contract. Detectron2's model
            # performs its own device transfer from the input dictionary.
            image.to(predictor.cfg.MODEL.DEVICE)
            batched_inputs.append({"image": image, "height": height, "width": width})
        predictions = predictor.model(batched_inputs)

    if not isinstance(predictions, (list, tuple)) or len(predictions) != len(images):
        raise RuntimeError(
            "DensePose batched predictor returned a different number of outputs than inputs"
        )
    return list(predictions)


def apply_densepose_to_frames_batched(
    frames_bgr: Sequence[np.ndarray],
    runner_masks: Sequence[np.ndarray],
    backend: DensePoseBackend,
    *,
    frame_indices: Sequence[int],
    min_mask_overlap: float = 0.1,
    target_crop_enabled: bool = False,
    target_crop_padding_ratio: float = 0.2,
    target_crop_padding_pixels: int = 16,
) -> list[DensePoseFrameOutput]:
    """PROTOTYPE: microbatch valid frames without changing per-frame semantics."""
    if not (len(frames_bgr) == len(runner_masks) == len(frame_indices)):
        raise ValueError("DensePose batched inputs must have matching lengths")
    if len(frames_bgr) <= 1:
        return [
            apply_densepose_to_frame(
                frame_bgr,
                runner_mask,
                backend,
                frame_index=int(frame_index),
                min_mask_overlap=min_mask_overlap,
                target_crop_enabled=target_crop_enabled,
                target_crop_padding_ratio=target_crop_padding_ratio,
                target_crop_padding_pixels=target_crop_padding_pixels,
            )
            for frame_bgr, runner_mask, frame_index in zip(
                frames_bgr,
                runner_masks,
                frame_indices,
            )
        ]

    prepared_frames: list[_DensePosePreparedFrame] = []
    prepared_positions: list[int] = []
    results: list[DensePoseFrameOutput | None] = [None] * len(frames_bgr)
    for position, (frame_bgr, runner_mask) in enumerate(zip(frames_bgr, runner_masks)):
        prepared = _prepare_densepose_frame(
            frame_bgr,
            runner_mask,
            target_crop_enabled=target_crop_enabled,
            target_crop_padding_ratio=target_crop_padding_ratio,
            target_crop_padding_pixels=target_crop_padding_pixels,
        )
        if isinstance(prepared, DensePoseFrameOutput):
            results[position] = prepared
        else:
            prepared_frames.append(prepared)
            prepared_positions.append(position)

    if prepared_frames:
        with backend.inference_lock:
            predictions = _default_predictor_batch(
                backend.predictor,
                [prepared.inference_frame for prepared in prepared_frames],
            )
        for position, prepared, prediction in zip(
            prepared_positions,
            prepared_frames,
            predictions,
        ):
            results[position] = _densepose_output_from_predictions(
                prepared,
                prediction,
                min_mask_overlap=min_mask_overlap,
            )

    if any(result is None for result in results):
        raise RuntimeError("DensePose batched inference did not resolve every input frame")
    return [result for result in results if result is not None]


def apply_densepose_to_frame(
    frame_bgr: np.ndarray,
    runner_mask: np.ndarray,
    backend: DensePoseBackend,
    *,
    frame_index: int,
    min_mask_overlap: float = 0.1,
    target_crop_enabled: bool = False,
    target_crop_padding_ratio: float = 0.2,
    target_crop_padding_pixels: int = 16,
) -> DensePoseFrameOutput:
    prepared = _prepare_densepose_frame(
        frame_bgr,
        runner_mask,
        target_crop_enabled=target_crop_enabled,
        target_crop_padding_ratio=target_crop_padding_ratio,
        target_crop_padding_pixels=target_crop_padding_pixels,
    )
    if isinstance(prepared, DensePoseFrameOutput):
        return prepared
    with backend.inference_lock:
        outputs = backend.predictor(prepared.inference_frame)
    return _densepose_output_from_predictions(
        prepared,
        outputs,
        min_mask_overlap=min_mask_overlap,
    )


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
    input_min_size_test: int | None = None,
    input_max_size_test: int | None = None,
    target_crop_enabled: bool = False,
    target_crop_padding_ratio: float = 0.2,
    target_crop_padding_pixels: int = 16,
    batch_size: int = DEFAULT_DENSEPOSE_BATCH_SIZE,
    write_qa_overlay: bool = True,
    progress_callback: DensePoseProgressCallback | None = None,
    benchmark_evidence_callback: DensePoseBenchmarkEvidenceCallback | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    run = RunningClipRun(run_dir)
    manifest_path = run.manifest_path
    manifest = run.read_manifest()
    source_segment = run.artifact_path("source_segment", manifest)
    runner_mask = run.artifact_path("runner_mask", manifest)
    densepose_path = run.artifact_path("densepose", manifest)
    qa_overlay_path = run.artifact_path("qa_overlay", manifest)
    effective_padding_ratio = max(0.0, float(target_crop_padding_ratio))
    effective_padding_pixels = max(0, int(target_crop_padding_pixels))
    effective_batch_size = max(1, min(MAX_DENSEPOSE_BATCH_SIZE, int(batch_size)))

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
            input_min_size_test=input_min_size_test,
            input_max_size_test=input_max_size_test,
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

    inference_settings = {
        "target_crop_enabled": bool(target_crop_enabled),
        "target_crop_padding_ratio": effective_padding_ratio,
        "target_crop_padding_pixels": effective_padding_pixels,
        "input_min_size_test": backend.input_min_size_test,
        "input_max_size_test": backend.input_max_size_test,
        "batch_size": effective_batch_size,
        "batched_inference_enabled": effective_batch_size > 1,
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
        if effective_batch_size == 1:
            # Keep the existing production path literal when batching is disabled.
            while True:
                ok, frame = source_capture.read()
                if not ok:
                    break
                labels = None
                mask = _read_mask_frame(mask_capture, width, height)
                if mask is None:
                    row = {
                        "usable": False,
                        "drop_reason": "runner_mask_frame_missing",
                        "inference_input": {
                            "target_crop_enabled": bool(target_crop_enabled),
                            "crop_bbox": None,
                            "width": 0,
                            "height": 0,
                        },
                    }
                else:
                    bbox = mask_bbox(mask)
                    mask_area_ratio = float((mask > 0).sum()) / float(max(width * height, 1))
                    if bbox is None:
                        row = {
                            "usable": False,
                            "drop_reason": "runner_mask_empty",
                            "inference_input": {
                                "target_crop_enabled": bool(target_crop_enabled),
                                "crop_bbox": None,
                                "width": 0,
                                "height": 0,
                            },
                        }
                    else:
                        apply_kwargs: dict[str, Any] = {"frame_index": frame_index}
                        if target_crop_enabled:
                            apply_kwargs.update(
                                {
                                    "target_crop_enabled": True,
                                    "target_crop_padding_ratio": effective_padding_ratio,
                                    "target_crop_padding_pixels": effective_padding_pixels,
                                }
                            )
                        densepose_output = apply_densepose_to_frame(
                            frame,
                            mask,
                            backend,
                            **apply_kwargs,
                        )
                        if isinstance(densepose_output, DensePoseFrameOutput):
                            row = densepose_output.row
                            labels = densepose_output.labels
                        else:
                            row = densepose_output
                        row.setdefault(
                            "inference_input",
                            {
                                "target_crop_enabled": False,
                                "crop_bbox": [0, 0, width, height],
                                "width": width,
                                "height": height,
                            },
                        )
                    row["mask_area_ratio"] = round(mask_area_ratio, 6)
                    row.setdefault("runner_bbox", bbox)
                    if writer is not None:
                        writer.write(_draw_densepose_overlay(frame, mask, row, labels=labels))

                row["frame_index"] = frame_index
                row.setdefault("usable", False)
                row.setdefault("drop_reason", None if row["usable"] else "densepose_missing")
                if benchmark_evidence_callback is not None:
                    benchmark_evidence_callback(
                        frame_index,
                        dict(row),
                        None if labels is None else np.ascontiguousarray(labels).copy(),
                    )
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
                            inference_input=row.get("inference_input"),
                        )
                    )
                frame_index += 1
        else:
            while True:
                batch_records: list[dict[str, Any]] = []
                for _ in range(effective_batch_size):
                    ok, frame = source_capture.read()
                    if not ok:
                        break
                    current_frame_index = frame_index + len(batch_records)
                    mask = _read_mask_frame(mask_capture, width, height)
                    batch_records.append(
                        {
                            "frame": frame,
                            "frame_index": current_frame_index,
                            "mask": mask,
                            "bbox": mask_bbox(mask) if mask is not None else None,
                        }
                    )
                if not batch_records:
                    break

                valid_positions = [
                    position
                    for position, record in enumerate(batch_records)
                    if record["mask"] is not None and record["bbox"] is not None
                ]
                if valid_positions:
                    outputs = apply_densepose_to_frames_batched(
                        [batch_records[position]["frame"] for position in valid_positions],
                        [batch_records[position]["mask"] for position in valid_positions],
                        backend,
                        frame_indices=[
                            batch_records[position]["frame_index"] for position in valid_positions
                        ],
                        target_crop_enabled=target_crop_enabled,
                        target_crop_padding_ratio=effective_padding_ratio,
                        target_crop_padding_pixels=effective_padding_pixels,
                    )
                    for position, output in zip(valid_positions, outputs):
                        batch_records[position]["densepose_output"] = output

                for record in batch_records:
                    frame = record["frame"]
                    current_frame_index = int(record["frame_index"])
                    mask = record["mask"]
                    bbox = record["bbox"]
                    labels = None
                    if mask is None:
                        row = {
                            "usable": False,
                            "drop_reason": "runner_mask_frame_missing",
                            "inference_input": {
                                "target_crop_enabled": bool(target_crop_enabled),
                                "crop_bbox": None,
                                "width": 0,
                                "height": 0,
                            },
                        }
                    else:
                        mask_area_ratio = float((mask > 0).sum()) / float(max(width * height, 1))
                        if bbox is None:
                            row = {
                                "usable": False,
                                "drop_reason": "runner_mask_empty",
                                "inference_input": {
                                    "target_crop_enabled": bool(target_crop_enabled),
                                    "crop_bbox": None,
                                    "width": 0,
                                    "height": 0,
                                },
                            }
                        else:
                            densepose_output = record["densepose_output"]
                            if isinstance(densepose_output, DensePoseFrameOutput):
                                row = densepose_output.row
                                labels = densepose_output.labels
                            else:
                                row = densepose_output
                            row.setdefault(
                                "inference_input",
                                {
                                    "target_crop_enabled": False,
                                    "crop_bbox": [0, 0, width, height],
                                    "width": width,
                                    "height": height,
                                },
                            )
                        row["mask_area_ratio"] = round(mask_area_ratio, 6)
                        row.setdefault("runner_bbox", bbox)
                        if writer is not None:
                            writer.write(_draw_densepose_overlay(frame, mask, row, labels=labels))

                    row["frame_index"] = current_frame_index
                    row.setdefault("usable", False)
                    row.setdefault("drop_reason", None if row["usable"] else "densepose_missing")
                    if benchmark_evidence_callback is not None:
                        benchmark_evidence_callback(
                            current_frame_index,
                            dict(row),
                            None if labels is None else np.ascontiguousarray(labels).copy(),
                        )
                    rows.append(row)
                    if progress_callback and (
                        current_frame_index == 0 or (current_frame_index + 1) % 10 == 0
                    ):
                        progress_callback(
                            build_densepose_progress(
                                phase="running_densepose",
                                processed_frames=current_frame_index + 1,
                                total_frames=total_frames,
                                elapsed_seconds=time.monotonic() - started_at,
                                frame_index=current_frame_index,
                                usable=bool(row.get("usable")),
                                inference_input=row.get("inference_input"),
                            )
                        )
                    frame_index = current_frame_index + 1
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
        inference_settings=inference_settings,
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
        "inference_settings": inference_settings,
    }
