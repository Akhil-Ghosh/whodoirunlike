from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from whodoirunlike.mask_artifacts import encode_uncompressed_rle, mask_bbox, mask_iou
from whodoirunlike.video_io import make_browser_playable_mp4s


INLINE_MASK_BACKEND = "yolo26n_seg_inline"
_SEVERE_IDENTITY_REJECTION_REASONS = frozenset(
    {
        "segmentation_mask_off_track",
        "segmentation_mask_centroid_jump",
    }
)
_APPEARANCE_ONLY_IDENTITY_RISK_REASONS = frozenset(
    {"low_prompt_anchor_similarity"}
)


@dataclass(frozen=True)
class InlineMaskConfig:
    dilation_pixels: int = 5
    mask_threshold: float = 0.5
    minimum_track_box_iou: float = 0.05
    minimum_mask_inside_track_box: float = 0.50
    maximum_area_change_ratio: float = 3.0
    maximum_centroid_jump_ratio: float = 0.28
    temporal_reset_gap_frames: int = 3
    sam_fallback_missing_target_rate_threshold: float = 0.15
    sam_fallback_missing_target_gap_frames: int = 3
    fallback_to_track_box: bool = True
    blank_identity_risk: bool = True
    rescue_appearance_only_identity_risk: bool = False
    identity_rescue_maximum_span_seconds: float = 0.50
    identity_rescue_minimum_confidence: float = 0.50
    identity_rescue_minimum_track_box_iou: float = 0.45
    identity_rescue_minimum_mask_inside_track_box: float = 0.80
    identity_rescue_minimum_adjacent_box_iou: float = 0.55
    identity_rescue_minimum_adjacent_mask_iou: float = 0.45
    identity_rescue_maximum_area_change_ratio: float = 2.0
    identity_rescue_maximum_centroid_jump_ratio: float = 0.08


@dataclass(frozen=True)
class YoloPersonInference:
    detections: np.ndarray
    boxes_xyxy: np.ndarray
    masks: tuple[np.ndarray, ...]

    @property
    def has_masks(self) -> bool:
        return bool(self.masks)


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left[:4])
    rx1, ry1, rx2, ry2 = (float(value) for value in right[:4])
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def parse_yolo_person_inference(
    result: Any,
    *,
    frame_height: int,
    frame_width: int,
    person_class_id: int = 0,
    mask_threshold: float = 0.5,
    include_masks: bool = True,
) -> YoloPersonInference:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return YoloPersonInference(
            detections=np.empty((0, 6), dtype=np.float32),
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            masks=(),
        )

    xyxy = _as_numpy(getattr(boxes, "xyxy", None)).astype(np.float32, copy=False)
    if xyxy.size == 0:
        return YoloPersonInference(
            detections=np.empty((0, 6), dtype=np.float32),
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            masks=(),
        )
    xyxy = xyxy.reshape((-1, 4))
    confidence = (
        _as_numpy(getattr(boxes, "conf", None)).astype(np.float32, copy=False).reshape((-1,))
    )
    classes = _as_numpy(getattr(boxes, "cls", None)).astype(np.float32, copy=False).reshape((-1,))
    if confidence.shape[0] != xyxy.shape[0]:
        confidence = np.ones((xyxy.shape[0],), dtype=np.float32)
    if classes.shape[0] != xyxy.shape[0]:
        classes = np.zeros((xyxy.shape[0],), dtype=np.float32)
    person_indexes = np.flatnonzero(classes.astype(int) == int(person_class_id))
    person_boxes = xyxy[person_indexes]
    detections = np.concatenate(
        [person_boxes, confidence[person_indexes, None], classes[person_indexes, None]],
        axis=1,
    ).astype(np.float32, copy=False)

    masks: list[np.ndarray] = []
    result_masks = getattr(getattr(result, "masks", None), "data", None) if include_masks else None
    masks_array = _as_numpy(result_masks)
    if (
        include_masks
        and masks_array.size
        and masks_array.ndim >= 3
        and masks_array.shape[0] == xyxy.shape[0]
    ):
        for source_index in person_indexes:
            mask = np.squeeze(masks_array[int(source_index)])
            if mask.shape != (frame_height, frame_width):
                mask = cv2.resize(
                    mask.astype(np.float32, copy=False),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks.append((mask > float(mask_threshold)).astype(np.uint8))
    return YoloPersonInference(
        detections=detections,
        boxes_xyxy=person_boxes.astype(np.float32, copy=False),
        masks=tuple(masks),
    )


def attach_segmentation_evidence(
    candidates: Sequence[dict[str, Any]],
    inference: YoloPersonInference,
    *,
    minimum_track_box_iou: float,
) -> None:
    if not inference.has_masks:
        return
    for candidate in candidates:
        track_box = candidate.get("box_xyxy")
        if not isinstance(track_box, Sequence) or len(track_box) < 4:
            continue
        requested_index = candidate.get("detection_index")
        selected_index: int | None = None
        association_method = "box_iou"
        if requested_index is not None:
            index = int(requested_index)
            if 0 <= index < len(inference.masks):
                selected_index = index
                association_method = "tracker_detection_index"
        if selected_index is None and len(inference.boxes_xyxy):
            overlaps = [_bbox_iou(track_box, box) for box in inference.boxes_xyxy]
            selected_index = int(np.argmax(overlaps)) if overlaps else None
        if selected_index is None:
            continue
        overlap = _bbox_iou(track_box, inference.boxes_xyxy[selected_index])
        if overlap < float(minimum_track_box_iou):
            continue
        ok, encoded = cv2.imencode(".png", inference.masks[selected_index] * 255)
        if not ok:
            continue
        candidate["_inline_mask_png"] = encoded.tobytes()
        candidate["_inline_mask_detection_index"] = selected_index
        candidate["_inline_mask_association_method"] = association_method
        candidate["_inline_mask_track_box_iou"] = float(overlap)


def _decode_candidate_mask(
    candidate: Mapping[str, Any], height: int, width: int
) -> np.ndarray | None:
    encoded = candidate.get("_inline_mask_png")
    if not isinstance(encoded, bytes):
        return None
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        return None
    if decoded.shape != (height, width):
        decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
    return (decoded > 0).astype(np.uint8)


def _track_box_mask(
    box: Sequence[int | float],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    x, y, box_width, box_height = (int(round(float(value))) for value in box[:4])
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(x1, min(width, x + max(0, box_width)))
    y2 = max(y1, min(height, y + max(0, box_height)))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return (float(xs.mean()), float(ys.mean()))


def _plausibility_reason(
    mask: np.ndarray,
    *,
    track_box_mask: np.ndarray,
    previous_mask: np.ndarray | None,
    width: int,
    config: InlineMaskConfig,
    frame_gap: int = 1,
) -> tuple[str | None, dict[str, float | None]]:
    area = int(mask.sum())
    box_area = int(track_box_mask.sum())
    inside = int(np.logical_and(mask > 0, track_box_mask > 0).sum())
    inside_ratio = inside / area if area else 0.0
    previous_area = int(previous_mask.sum()) if previous_mask is not None else 0
    if area and previous_area:
        area_change_ratio = max(area / previous_area, previous_area / area)
    else:
        area_change_ratio = None
    centroid = _centroid(mask)
    previous_centroid = _centroid(previous_mask) if previous_mask is not None else None
    centroid_jump = (
        float(np.hypot(centroid[0] - previous_centroid[0], centroid[1] - previous_centroid[1]))
        if centroid is not None and previous_centroid is not None
        else None
    )
    effective_frame_gap = max(1, int(frame_gap))
    maximum_area_change_ratio = (
        max(1.0, float(config.maximum_area_change_ratio)) * effective_frame_gap
    )
    maximum_centroid_jump_pixels = (
        width * config.maximum_centroid_jump_ratio * effective_frame_gap
    )
    metrics: dict[str, float | None] = {
        "mask_inside_track_box": inside_ratio,
        "mask_to_track_box_area": area / box_area if box_area else None,
        "area_change_ratio": area_change_ratio,
        "maximum_area_change_ratio": maximum_area_change_ratio,
        "centroid_jump_pixels": centroid_jump,
        "maximum_centroid_jump_pixels": maximum_centroid_jump_pixels,
    }
    if not area:
        return "segmentation_mask_empty", metrics
    if inside_ratio < config.minimum_mask_inside_track_box:
        return "segmentation_mask_off_track", metrics
    if area_change_ratio is not None and area_change_ratio > maximum_area_change_ratio:
        return "segmentation_mask_area_jump", metrics
    if centroid_jump is not None and centroid_jump > maximum_centroid_jump_pixels:
        return "segmentation_mask_centroid_jump", metrics
    return None, metrics


def _xywh_iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_xyxy = (
        float(left[0]),
        float(left[1]),
        float(left[0]) + float(left[2]),
        float(left[1]) + float(left[3]),
    )
    right_xyxy = (
        float(right[0]),
        float(right[1]),
        float(right[0]) + float(right[2]),
        float(right[1]) + float(right[3]),
    )
    return _bbox_iou(left_xyxy, right_xyxy)


def _appearance_only_identity_risk(
    candidate: Mapping[str, Any] | None,
) -> bool:
    if candidate is None or str(candidate.get("_identity_state") or "") != "identity_risk":
        return False
    reasons = frozenset(str(reason) for reason in candidate.get("_identity_reasons") or [])
    return reasons == _APPEARANCE_ONLY_IDENTITY_RISK_REASONS


def _mask_inside_track_box_ratio(mask: np.ndarray, track_box_mask: np.ndarray) -> float:
    area = int(mask.sum())
    if not area:
        return 0.0
    return float(np.logical_and(mask > 0, track_box_mask > 0).sum() / area)


def _area_change_ratio(left: np.ndarray, right: np.ndarray) -> float:
    left_area = int(left.sum())
    right_area = int(right.sum())
    if not left_area or not right_area:
        return float("inf")
    return max(left_area / right_area, right_area / left_area)


def _centroid_jump_ratio(
    left: np.ndarray,
    right: np.ndarray,
    *,
    width: int,
    height: int,
) -> float:
    left_centroid = _centroid(left)
    right_centroid = _centroid(right)
    if left_centroid is None or right_centroid is None:
        return float("inf")
    distance = float(
        np.hypot(
            left_centroid[0] - right_centroid[0],
            left_centroid[1] - right_centroid[1],
        )
    )
    return distance / max(float(np.hypot(width, height)), 1.0)


def _appearance_only_identity_rescue_frame_indexes(
    *,
    target_candidates: Mapping[int, Mapping[str, Any]],
    height: int,
    width: int,
    fps: float,
    config: InlineMaskConfig,
) -> set[int]:
    risk_indexes = sorted(
        frame_index
        for frame_index, candidate in target_candidates.items()
        if _appearance_only_identity_risk(candidate)
    )
    if not risk_indexes:
        return set()

    spans: list[list[int]] = []
    for frame_index in risk_indexes:
        if not spans or frame_index != spans[-1][-1] + 1:
            spans.append([frame_index])
        else:
            spans[-1].append(frame_index)

    rescued: set[int] = set()
    maximum_span_frames = max(
        1,
        int(
            math.ceil(
                max(0.0, float(fps))
                * max(0.0, float(config.identity_rescue_maximum_span_seconds))
            )
        ),
    )
    for span in spans:
        if len(span) > maximum_span_frames:
            continue
        previous_candidate = target_candidates.get(span[0] - 1)
        next_candidate = target_candidates.get(span[-1] + 1)
        if (
            previous_candidate is None
            or next_candidate is None
            or str(previous_candidate.get("_identity_state") or "") != "usable"
            or str(next_candidate.get("_identity_state") or "") != "usable"
        ):
            continue

        sequence_indexes = [span[0] - 1, *span, span[-1] + 1]
        sequence_candidates = [target_candidates[index] for index in sequence_indexes]
        track_ids = {candidate.get("track_id") for candidate in sequence_candidates}
        if None in track_ids or len(track_ids) != 1:
            continue
        risk_confidences = [
            float(candidate.get("confidence") or 0.0)
            for candidate in sequence_candidates[1:-1]
        ]
        if any(
            not math.isfinite(confidence)
            or confidence < float(config.identity_rescue_minimum_confidence)
            for confidence in risk_confidences
        ):
            continue
        association_ious = [
            float(candidate.get("_inline_mask_track_box_iou") or 0.0)
            for candidate in sequence_candidates
        ]
        if any(
            not math.isfinite(association_iou)
            or association_iou < float(config.identity_rescue_minimum_track_box_iou)
            for association_iou in association_ious
        ):
            continue
        if any(candidate.get("box") is None for candidate in sequence_candidates):
            continue

        sequence_masks = [
            _decode_candidate_mask(candidate, height, width)
            for candidate in sequence_candidates
        ]
        if any(mask is None or not mask.any() for mask in sequence_masks):
            continue
        decoded_masks = [mask for mask in sequence_masks if mask is not None]
        if any(
            _mask_inside_track_box_ratio(
                mask,
                _track_box_mask(candidate["box"], height=height, width=width),
            )
            < float(config.identity_rescue_minimum_mask_inside_track_box)
            for candidate, mask in zip(sequence_candidates, decoded_masks)
        ):
            continue

        adjacent_pairs = zip(
            sequence_candidates,
            sequence_candidates[1:],
            decoded_masks,
            decoded_masks[1:],
        )
        if any(
            _xywh_iou(left_candidate["box"], right_candidate["box"])
            < float(config.identity_rescue_minimum_adjacent_box_iou)
            or mask_iou(left_mask, right_mask)
            < float(config.identity_rescue_minimum_adjacent_mask_iou)
            or _area_change_ratio(left_mask, right_mask)
            > float(config.identity_rescue_maximum_area_change_ratio)
            or _centroid_jump_ratio(
                left_mask,
                right_mask,
                width=width,
                height=height,
            )
            > float(config.identity_rescue_maximum_centroid_jump_ratio)
            for left_candidate, right_candidate, left_mask, right_mask in adjacent_pairs
        ):
            continue
        rescued.update(span)
    return rescued


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = frame.copy()
    color = np.zeros_like(frame)
    color[:, :, 1] = 220
    selected = (mask > 0)[:, :, None]
    result = np.where(selected, (frame * 0.55 + color * 0.45).astype(np.uint8), result)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 250, 243), 2)
    return result


def write_selected_runner_mask_artifacts(
    *,
    frames: Sequence[np.ndarray],
    fps: float,
    target_candidates: Mapping[int, Mapping[str, Any]],
    runner_mask_path: Path,
    masked_runner_path: Path,
    qa_overlay_path: Path,
    metadata_path: Path,
    masks_jsonl_path: Path,
    model: str,
    config: InlineMaskConfig,
    browser_playable: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if not frames:
        raise ValueError("Inline segmentation needs at least one source frame")
    height, width = frames[0].shape[:2]
    identity_rescue_frame_indexes = (
        _appearance_only_identity_rescue_frame_indexes(
            target_candidates=target_candidates,
            height=height,
            width=width,
            fps=fps,
            config=config,
        )
        if config.blank_identity_risk and config.rescue_appearance_only_identity_risk
        else set()
    )
    for path in (
        runner_mask_path,
        masked_runner_path,
        qa_overlay_path,
        metadata_path,
        masks_jsonl_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_writer = cv2.VideoWriter(str(runner_mask_path), fourcc, fps, (width, height), True)
    masked_writer = cv2.VideoWriter(str(masked_runner_path), fourcc, fps, (width, height), True)
    qa_writer = cv2.VideoWriter(str(qa_overlay_path), fourcc, fps, (width, height), True)
    if not mask_writer.isOpened() or not masked_writer.isOpened() or not qa_writer.isOpened():
        mask_writer.release()
        masked_writer.release()
        qa_writer.release()
        raise ValueError("Could not open one or more inline mask output writers")

    kernel: np.ndarray | None = None
    if config.dilation_pixels > 0:
        size = (config.dilation_pixels * 2) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    fallback_frame_indexes: list[int] = []
    severe_rejection_frame_indexes: list[int] = []
    identity_risk_blank_frame_indexes: list[int] = []
    accepted_identity_rescue_frame_indexes: list[int] = []
    temporal_rejection_frame_indexes: list[int] = []
    temporal_baseline_reset_frame_indexes: list[int] = []
    association_ious: list[float] = []
    temporal_ious: list[float] = []
    previous_plausible_mask: np.ndarray | None = None
    previous_plausible_frame_index: int | None = None
    previous_output_mask: np.ndarray | None = None
    previous_output_centroid: tuple[float, float] | None = None
    consecutive_missing_target_frames = 0
    maximum_consecutive_missing_target_frames = 0
    if progress_callback:
        progress_callback("rendering_inline_mask", 0, len(frames))

    try:
        with (
            metadata_path.open("w", encoding="utf-8") as metadata_file,
            masks_jsonl_path.open("w", encoding="utf-8") as masks_file,
        ):
            for frame_index, frame in enumerate(frames):
                candidate = target_candidates.get(frame_index)
                mask = np.zeros((height, width), dtype=np.uint8)
                source = "blank"
                drop_reason: str | None = None
                fallback_reason: str | None = None
                plausibility: dict[str, float | None] = {}
                temporal_frame_gap = (
                    frame_index - previous_plausible_frame_index
                    if previous_plausible_frame_index is not None
                    else None
                )
                identity_state = (
                    str(candidate.get("_identity_state") or "missing") if candidate else "missing"
                )
                identity_rescue_requested = frame_index in identity_rescue_frame_indexes
                track_mask = (
                    _track_box_mask(candidate["box"], height=height, width=width)
                    if candidate is not None and candidate.get("box") is not None
                    else None
                )

                if candidate is None:
                    counts["missing_target_frames"] += 1
                    consecutive_missing_target_frames += 1
                    maximum_consecutive_missing_target_frames = max(
                        maximum_consecutive_missing_target_frames,
                        consecutive_missing_target_frames,
                    )
                    drop_reason = "target_track_missing"
                else:
                    consecutive_missing_target_frames = 0
                    if (
                        config.blank_identity_risk
                        and identity_state != "usable"
                        and not identity_rescue_requested
                    ):
                        counts["identity_risk_blank_frames"] += 1
                        identity_risk_blank_frame_indexes.append(frame_index)
                        drop_reason = "identity_risk_blank"
                    else:
                        mask = _decode_candidate_mask(candidate, height, width)
                        if mask is None:
                            mask = np.zeros((height, width), dtype=np.uint8)
                            fallback_reason = "segmentation_not_associated"
                        elif track_mask is not None:
                            comparison_mask = previous_plausible_mask
                            if (
                                temporal_frame_gap is not None
                                and temporal_frame_gap
                                > max(0, int(config.temporal_reset_gap_frames))
                            ):
                                comparison_mask = None
                                temporal_baseline_reset_frame_indexes.append(frame_index)
                            fallback_reason, plausibility = _plausibility_reason(
                                mask,
                                track_box_mask=track_mask,
                                previous_mask=comparison_mask,
                                width=width,
                                config=config,
                                frame_gap=temporal_frame_gap or 1,
                            )
                        if fallback_reason is None:
                            source = (
                                "yolo_segmentation_identity_rescue"
                                if identity_rescue_requested
                                else "yolo_segmentation"
                            )
                            counts["associated_frames"] += 1
                            if identity_rescue_requested:
                                counts["identity_rescue_frames"] += 1
                                accepted_identity_rescue_frame_indexes.append(frame_index)
                            previous_plausible_mask = mask.copy()
                            previous_plausible_frame_index = frame_index
                            overlap = candidate.get("_inline_mask_track_box_iou")
                            if overlap is not None:
                                association_ious.append(float(overlap))
                        elif identity_rescue_requested:
                            mask.fill(0)
                            drop_reason = fallback_reason
                            counts["rejected_segmentation_frames"] += 1
                            counts["identity_risk_blank_frames"] += 1
                            identity_risk_blank_frame_indexes.append(frame_index)
                            fallback_reasons[fallback_reason] += 1
                            if fallback_reason in _SEVERE_IDENTITY_REJECTION_REASONS:
                                counts["severe_rejection_frames"] += 1
                                severe_rejection_frame_indexes.append(frame_index)
                            if fallback_reason == "segmentation_mask_centroid_jump":
                                temporal_rejection_frame_indexes.append(frame_index)
                        elif fallback_reason in _SEVERE_IDENTITY_REJECTION_REASONS:
                            mask.fill(0)
                            drop_reason = fallback_reason
                            counts["rejected_segmentation_frames"] += 1
                            counts["severe_rejection_frames"] += 1
                            fallback_reasons[fallback_reason] += 1
                            severe_rejection_frame_indexes.append(frame_index)
                            if fallback_reason == "segmentation_mask_centroid_jump":
                                temporal_rejection_frame_indexes.append(frame_index)
                        elif config.fallback_to_track_box and track_mask is not None:
                            mask = track_mask
                            source = "track_box_fallback"
                            counts["track_box_fallback_frames"] += 1
                            fallback_reasons[fallback_reason] += 1
                            fallback_frame_indexes.append(frame_index)
                            if fallback_reason in {
                                "segmentation_mask_area_jump",
                                "segmentation_mask_centroid_jump",
                            }:
                                temporal_rejection_frame_indexes.append(frame_index)
                        else:
                            mask.fill(0)
                            drop_reason = fallback_reason
                            counts["rejected_segmentation_frames"] += 1

                if (
                    kernel is not None
                    and mask.any()
                    and source.startswith("yolo_segmentation")
                ):
                    mask = cv2.dilate(mask, kernel)
                if mask.any():
                    counts["nonempty_frames"] += 1
                centroid = _centroid(mask)
                centroid_delta = (
                    float(
                        np.hypot(
                            centroid[0] - previous_output_centroid[0],
                            centroid[1] - previous_output_centroid[1],
                        )
                    )
                    if centroid is not None and previous_output_centroid is not None
                    else None
                )
                temporal_iou = (
                    mask_iou(previous_output_mask, mask)
                    if previous_output_mask is not None
                    else None
                )
                if temporal_iou is not None:
                    temporal_ious.append(temporal_iou)
                previous_output_mask = mask.copy()
                if centroid is not None:
                    previous_output_centroid = centroid

                gray = mask * 255
                mask_writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                masked = np.zeros_like(frame)
                masked[mask > 0] = frame[mask > 0]
                masked_writer.write(masked)
                qa_writer.write(_overlay(frame, mask))

                area = int(mask.sum())
                time_seconds = round(frame_index / fps, 6) if fps else None
                metadata_row = {
                    "frame_index": frame_index,
                    "time_seconds": time_seconds,
                    "identity_state": identity_state,
                    "source": source,
                    "fallback": source == "track_box_fallback",
                    "fallback_reason": fallback_reason,
                    "drop_reason": drop_reason,
                    "usable": bool(mask.any()) and drop_reason is None,
                    "association_method": candidate.get("_inline_mask_association_method")
                    if candidate
                    else None,
                    "detection_index": candidate.get("_inline_mask_detection_index")
                    if candidate
                    else None,
                    "track_box_iou": round(float(candidate["_inline_mask_track_box_iou"]), 6)
                    if candidate and candidate.get("_inline_mask_track_box_iou") is not None
                    else None,
                    "mask_area": area,
                    "mask_area_ratio": round(area / float(max(height * width, 1)), 8),
                    "centroid": [round(centroid[0], 2), round(centroid[1], 2)]
                    if centroid is not None
                    else None,
                    "centroid_delta_px": round(centroid_delta, 2)
                    if centroid_delta
                    else None,
                    "temporal_frame_gap": temporal_frame_gap,
                    **{
                        key: round(float(value), 6) if value is not None else None
                        for key, value in plausibility.items()
                    },
                }
                metadata_file.write(json.dumps(metadata_row, separators=(",", ":")) + "\n")
                masks_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "time_seconds": time_seconds,
                            "width": width,
                            "height": height,
                            "area": area,
                            "area_ratio": round(area / float(max(height * width, 1)), 8),
                            "bbox": mask_bbox(mask),
                            "temporal_iou_prev": round(temporal_iou, 6)
                            if temporal_iou is not None
                            else None,
                            "rle": encode_uncompressed_rle(mask),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if progress_callback:
                    progress_callback("rendering_inline_mask", frame_index + 1, len(frames))
    finally:
        mask_writer.release()
        masked_writer.release()
        qa_writer.release()
    render_seconds = time.perf_counter() - started_at

    encode_seconds = 0.0
    if browser_playable:
        if progress_callback:
            progress_callback("encoding_inline_mask", len(frames), len(frames))
        encode_started_at = time.perf_counter()
        make_browser_playable_mp4s([runner_mask_path, masked_runner_path, qa_overlay_path])
        encode_seconds = time.perf_counter() - encode_started_at
    if progress_callback:
        progress_callback("writing_inline_mask_outputs", len(frames), len(frames))

    usable_target_frames = (
        len(frames) - counts["missing_target_frames"] - counts["identity_risk_blank_frames"]
    )
    fallback_rate = counts["track_box_fallback_frames"] / max(usable_target_frames, 1)
    degraded_mask_frames = (
        counts["track_box_fallback_frames"] + counts["rejected_segmentation_frames"]
    )
    degraded_rate = degraded_mask_frames / max(usable_target_frames, 1)
    missing_target_rate = counts["missing_target_frames"] / max(len(frames), 1)
    mean_temporal_iou = (
        round(float(np.mean(temporal_ious)), 6) if temporal_ious else None
    )
    summary = {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "output_path": str(masks_jsonl_path),
        "associated_frames": counts["associated_frames"],
        "nonempty_frames": counts["nonempty_frames"],
        "missing_target_frames": counts["missing_target_frames"],
        "missing_target_rate": round(missing_target_rate, 6),
        "maximum_consecutive_missing_target_frames": (
            maximum_consecutive_missing_target_frames
        ),
        "identity_risk_blank_frames": counts["identity_risk_blank_frames"],
        "identity_risk_frames": (
            counts["identity_risk_blank_frames"]
            + counts["identity_rescue_frames"]
        ),
        "identity_rescue_eligible_frames": len(identity_rescue_frame_indexes),
        "identity_rescue_frames": counts["identity_rescue_frames"],
        "identity_rescue_rejected_frames": (
            len(identity_rescue_frame_indexes)
            - counts["identity_rescue_frames"]
        ),
        "rejected_segmentation_frames": counts["rejected_segmentation_frames"],
        "severe_rejection_frames": counts["severe_rejection_frames"],
        "track_box_fallback_frames": counts["track_box_fallback_frames"],
        "track_box_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "fallback_rate_on_usable_target_frames": round(fallback_rate, 6),
        "degraded_mask_frames": degraded_mask_frames,
        "degraded_rate_on_usable_target_frames": round(degraded_rate, 6),
        "mean_track_box_iou": round(float(np.mean(association_ious)), 6)
        if association_ious
        else None,
        "mean_temporal_iou": mean_temporal_iou,
        "mean_mask_churn": round(1.0 - mean_temporal_iou, 6)
        if mean_temporal_iou is not None
        else None,
        "dilation_pixels": config.dilation_pixels,
        "sam_fallback_recommended": bool(
            severe_rejection_frame_indexes or identity_risk_blank_frame_indexes
        )
        or degraded_rate > 0.15
        or missing_target_rate
        > max(0.0, float(config.sam_fallback_missing_target_rate_threshold))
        or maximum_consecutive_missing_target_frames
        >= max(1, int(config.sam_fallback_missing_target_gap_frames)),
    }
    result = {
        "status": "complete",
        "backend": INLINE_MASK_BACKEND,
        "model": model,
        "runner_mask": str(runner_mask_path),
        "masked_runner": str(masked_runner_path),
        "qa_overlay": str(qa_overlay_path),
        "metadata": str(metadata_path),
        "masks_jsonl": str(masks_jsonl_path),
        "summary": summary,
        "mask_summary": summary,
        "timing": {
            "render_seconds": round(render_seconds, 6),
            "encode_seconds": round(encode_seconds, 6),
            "total_seconds": round(time.perf_counter() - started_at, 6),
        },
        "fallback": {
            "used": bool(fallback_frame_indexes),
            "frame_indexes": fallback_frame_indexes,
            "reasons": dict(sorted(fallback_reasons.items())),
            "sam_fallback_recommended": summary["sam_fallback_recommended"],
        },
        "safety": {
            "blank_identity_risk": config.blank_identity_risk,
            "rescue_appearance_only_identity_risk": (
                config.rescue_appearance_only_identity_risk
            ),
            "identity_risk_blank_frame_indexes": identity_risk_blank_frame_indexes,
            "identity_rescue_frame_indexes": accepted_identity_rescue_frame_indexes,
            "identity_rescue_maximum_span_seconds": max(
                0.0,
                float(config.identity_rescue_maximum_span_seconds),
            ),
            "identity_rescue_minimum_confidence": float(
                config.identity_rescue_minimum_confidence
            ),
            "identity_rescue_minimum_track_box_iou": float(
                config.identity_rescue_minimum_track_box_iou
            ),
            "identity_rescue_minimum_mask_inside_track_box": float(
                config.identity_rescue_minimum_mask_inside_track_box
            ),
            "identity_rescue_minimum_adjacent_box_iou": float(
                config.identity_rescue_minimum_adjacent_box_iou
            ),
            "identity_rescue_minimum_adjacent_mask_iou": float(
                config.identity_rescue_minimum_adjacent_mask_iou
            ),
            "identity_rescue_maximum_area_change_ratio": float(
                config.identity_rescue_maximum_area_change_ratio
            ),
            "identity_rescue_maximum_centroid_jump_ratio": float(
                config.identity_rescue_maximum_centroid_jump_ratio
            ),
            "severe_rejection_frame_indexes": severe_rejection_frame_indexes,
            "temporal_rejection_frame_indexes": temporal_rejection_frame_indexes,
            "temporal_baseline_reset_frame_indexes": (
                temporal_baseline_reset_frame_indexes
            ),
            "temporal_reset_gap_frames": max(
                0,
                int(config.temporal_reset_gap_frames),
            ),
            "sam_fallback_missing_target_rate_threshold": max(
                0.0,
                float(config.sam_fallback_missing_target_rate_threshold),
            ),
            "sam_fallback_missing_target_gap_frames": max(
                1,
                int(config.sam_fallback_missing_target_gap_frames),
            ),
            "minimum_mask_inside_track_box": config.minimum_mask_inside_track_box,
            "maximum_area_change_ratio": config.maximum_area_change_ratio,
            "maximum_centroid_jump_ratio": config.maximum_centroid_jump_ratio,
        },
    }
    if not browser_playable:
        result["deferred_browser_encoding"] = {
            "required": True,
            "paths": [
                str(runner_mask_path),
                str(masked_runner_path),
                str(qa_overlay_path),
            ],
        }
    return result
