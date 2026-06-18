from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.mask_artifacts import write_masks_jsonl_from_video
from whodoirunlike.sam2_runner import (
    extract_video_frames,
    inspect_video,
    load_prompt,
    read_json,
    write_json,
    write_mask_outputs,
)


DEFAULT_SAM31_GPU_MODEL = "facebook/sam3.1"
DEFAULT_SAM31_GPU_OBJ_ID = 1
DEFAULT_SAM31_GPU_TRACK_PROMPT_ANCHORS = 6
DEFAULT_SAM31_GPU_DISABLE_DEMO_SUPPRESSION = True


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _relative_prompt_tensors(
    *,
    prompt: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    import torch

    points = prompt.get("points")
    labels = prompt.get("labels")
    if points is not None and len(points) > 0:
        rel_points = np.asarray(points, dtype=np.float32).copy()
        rel_points[:, 0] = rel_points[:, 0] / max(float(width), 1.0)
        rel_points[:, 1] = rel_points[:, 1] / max(float(height), 1.0)
        rel_points = np.clip(rel_points, 0.0, 1.0)
        rel_labels = np.asarray(labels, dtype=np.int32)
        return {
            "points": torch.tensor(rel_points, dtype=torch.float32, device="cuda"),
            "point_labels": torch.tensor(rel_labels, dtype=torch.int32, device="cuda"),
        }

    box = prompt.get("box")
    if box is not None:
        return _box_prompt_tensors(box=np.asarray(box, dtype=np.float32), width=width, height=height)

    raise ValueError("SAM 3.1 GPU requires at least one prompt point or box.")


def _box_prompt_tensors(
    *,
    box: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    import torch

    x1, y1, x2, y2 = [float(value) for value in np.asarray(box).tolist()]
    rel_box = np.array(
        [[
            x1 / max(float(width), 1.0),
            y1 / max(float(height), 1.0),
            max(x2 - x1, 0.0) / max(float(width), 1.0),
            max(y2 - y1, 0.0) / max(float(height), 1.0),
        ]],
        dtype=np.float32,
    )
    rel_box = np.clip(rel_box, 0.0, 1.0)
    return {
        "bounding_boxes": torch.tensor(rel_box, dtype=torch.float32, device="cuda"),
        "bounding_box_labels": torch.tensor([1], dtype=torch.int32, device="cuda"),
    }


def _support_points_from_box(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in np.asarray(box).tolist()]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + width * 0.5
    return np.array(
        [
            [center_x, y1 + height * 0.35],
            [center_x, y1 + height * 0.52],
            [center_x, y1 + height * 0.7],
        ],
        dtype=np.float32,
    )


def _prompt_points_with_box_support(
    prompt: dict[str, Any],
    *,
    box: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    points = prompt.get("points")
    labels = prompt.get("labels")
    point_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    if points is not None and len(points) > 0:
        point_parts.append(np.asarray(points, dtype=np.float32))
        label_parts.append(np.asarray(labels, dtype=np.int32))
    if box is not None:
        support_points = _support_points_from_box(box)
        point_parts.append(support_points)
        label_parts.append(np.ones(len(support_points), dtype=np.int32))
    if not point_parts:
        return None, None
    return np.vstack(point_parts).astype(np.float32), np.concatenate(label_parts).astype(np.int32)


def _seed_points_for_frame(
    *,
    prompt: dict[str, Any],
    box: np.ndarray | None,
    seed_frame: int,
    prompt_frame: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if seed_frame == prompt_frame:
        return _prompt_points_with_box_support(prompt, box=box)
    if box is None:
        return None, None
    support_points = _support_points_from_box(box)
    return support_points, np.ones(len(support_points), dtype=np.int32)


def _has_prompt_points(prompt: dict[str, Any]) -> bool:
    points = prompt.get("points")
    return points is not None and len(points) > 0


def _point_prompt_tensors(
    *,
    points: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    import torch

    rel_points = np.asarray(points, dtype=np.float32).copy()
    rel_points[:, 0] = rel_points[:, 0] / max(float(width), 1.0)
    rel_points[:, 1] = rel_points[:, 1] / max(float(height), 1.0)
    rel_points = np.clip(rel_points, 0.0, 1.0)
    rel_labels = np.asarray(labels, dtype=np.int32)
    return {
        "points": torch.tensor(rel_points, dtype=torch.float32, device="cuda"),
        "point_labels": torch.tensor(rel_labels, dtype=torch.int32, device="cuda"),
    }


def _nearest_track_box(
    *,
    frame_index: int,
    track_boxes: dict[int, np.ndarray],
    max_distance: int,
) -> tuple[int, np.ndarray] | None:
    best: tuple[int, np.ndarray] | None = None
    best_distance: int | None = None
    for candidate_index, box in track_boxes.items():
        distance = abs(candidate_index - frame_index)
        if distance > max_distance:
            continue
        if best_distance is None or distance < best_distance:
            best = (candidate_index, box)
            best_distance = distance
    return best


def _track_prompt_anchors(
    *,
    prompt_frame: int,
    frame_count: int,
    track_boxes: dict[int, np.ndarray],
    max_anchors: int = DEFAULT_SAM31_GPU_TRACK_PROMPT_ANCHORS,
) -> list[int]:
    if not track_boxes or frame_count <= 0 or max_anchors <= 0:
        return []
    desired = [prompt_frame]
    if max_anchors > 1:
        step = max(1, frame_count // max_anchors)
        desired.extend(range(0, frame_count, step))
        desired.append(frame_count - 1)

    anchors: list[int] = []
    for frame_index in desired:
        nearest = _nearest_track_box(
            frame_index=max(0, min(int(frame_index), frame_count - 1)),
            track_boxes=track_boxes,
            max_distance=max(12, frame_count // 12),
        )
        if nearest is None:
            continue
        nearest_index, _ = nearest
        if nearest_index not in anchors:
            anchors.append(nearest_index)
        if len(anchors) >= max_anchors:
            break
    return sorted(anchors)


def _first_track_prompt_frame(
    *,
    track_boxes: dict[int, np.ndarray],
    frame_count: int,
) -> tuple[int, np.ndarray] | None:
    for frame_index in sorted(track_boxes):
        if 0 <= frame_index < frame_count:
            return frame_index, track_boxes[frame_index]
    return None


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _mask_from_outputs(outputs: dict[str, Any], *, obj_id: int) -> np.ndarray | None:
    obj_ids = outputs.get("out_obj_ids")
    masks = outputs.get("out_binary_masks")
    if obj_ids is None or masks is None:
        return None

    ids = _as_numpy(obj_ids).reshape(-1).tolist()
    masks_array = masks
    fallback: np.ndarray | None = None
    for index, raw_id in enumerate(ids):
        mask = _as_numpy(masks_array[index])
        mask = np.squeeze(mask)
        if fallback is None and mask.any():
            fallback = mask
        if int(raw_id) == obj_id:
            return (mask > 0).astype("uint8")
    return (fallback > 0).astype("uint8") if fallback is not None else None


def _first_nonempty_output_object_id(outputs: dict[str, Any]) -> int | None:
    obj_ids = outputs.get("out_obj_ids")
    masks = outputs.get("out_binary_masks")
    if obj_ids is None or masks is None:
        return None

    ids = _as_numpy(obj_ids).reshape(-1).tolist()
    for index, raw_id in enumerate(ids):
        mask = np.squeeze(_as_numpy(masks[index]))
        if mask.any():
            return int(raw_id)
    return None


def _patch_multiplex_init_state_kwargs(predictor: Any) -> None:
    """Filter unsupported SAM 3.1 multiplex init_state kwargs until upstream lands it."""
    import inspect

    model = getattr(predictor, "model", None)
    original_init_state = getattr(model, "init_state", None)
    if model is None or original_init_state is None:
        return

    signature = inspect.signature(original_init_state)
    valid_params = set(signature.parameters)

    def filtered_init_state(*args: Any, **kwargs: Any) -> Any:
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_params}
        return original_init_state(*args, **filtered_kwargs)

    setattr(model, "init_state", filtered_init_state)


def _configure_interactive_tracker_for_user_prompt(predictor: Any) -> dict[str, Any]:
    model = getattr(predictor, "model", None)
    if model is None:
        return {"applied": False, "reason": "missing_model"}
    if not _env_bool(
        "WHODOIRUNLIKE_SAM31_GPU_DISABLE_DEMO_SUPPRESSION",
        default=DEFAULT_SAM31_GPU_DISABLE_DEMO_SUPPRESSION,
    ):
        return {"applied": False, "reason": "disabled_by_env"}

    overrides = {
        "masklet_confirmation_enable": False,
        "hotstart_delay": 0,
        "hotstart_unmatch_thresh": 0,
        "hotstart_dup_thresh": 0,
        "suppress_unmatched_only_within_hotstart": True,
        "suppress_overlapping_based_on_recent_occlusion_threshold": 1.1,
    }
    applied: dict[str, dict[str, Any]] = {}
    for attr, value in overrides.items():
        if hasattr(model, attr):
            previous = getattr(model, attr)
            setattr(model, attr, value)
            applied[attr] = {"previous": previous, "current": value}
    return {"applied": bool(applied), "overrides": applied}


def _collect_stream_masks(
    *,
    predictor: Any,
    session_id: str,
    active_obj_id: int,
    start_frame_index: int,
    masks_by_frame: dict[int, np.ndarray],
    direction: str,
) -> dict[str, Any]:
    responses = 0
    masks = 0
    try:
        for stream_response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": direction,
                "start_frame_index": start_frame_index,
            }
        ):
            responses += 1
            frame_index = int(stream_response["frame_index"])
            mask = _mask_from_outputs(
                stream_response.get("outputs", {}),
                obj_id=active_obj_id,
            )
            if mask is not None:
                masks_by_frame[frame_index] = mask
                masks += 1
    except RuntimeError as exc:
        if "No prompts are received on any frames" not in str(exc):
            raise
        return {
            "direction": direction,
            "responses": responses,
            "masks": masks,
            "warning": str(exc),
        }
    return {"direction": direction, "responses": responses, "masks": masks}


def _propagate_from_seed_frame(
    *,
    predictor: Any,
    session_id: str,
    active_obj_id: int,
    seed_frame: int,
    masks_by_frame: dict[int, np.ndarray],
    label: str,
    directions: tuple[str, ...] = ("forward", "backward"),
) -> list[dict[str, Any]]:
    diagnostics = []
    for direction in directions:
        result = _collect_stream_masks(
            predictor=predictor,
            session_id=session_id,
            active_obj_id=active_obj_id,
            start_frame_index=seed_frame,
            masks_by_frame=masks_by_frame,
            direction=direction,
        )
        result["pass"] = label
        diagnostics.append(result)
    return diagnostics


def _collect_sam31_masks(
    *,
    predictor: Any,
    video_path: Path,
    prompt: dict[str, Any],
    width: int,
    height: int,
    frame_count: int,
    obj_id: int,
    track_boxes: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    import torch

    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(video_path),
        }
    )
    session_id = response["session_id"]
    masks_by_frame: dict[int, np.ndarray] = {}
    diagnostics: dict[str, Any] = {
        "initial_obj_id": obj_id,
        "active_obj_id": obj_id,
        "visual_box_prompt": False,
        "propagation": [],
    }
    try:
        prompt_frame = max(0, min(int(prompt["frame_index"]), max(frame_count - 1, 0)))
        seed_frame = prompt_frame
        seed_box = None
        seed_source = "user_prompt"
        if track_boxes:
            first_track_box = _first_track_prompt_frame(
                track_boxes=track_boxes,
                frame_count=frame_count,
            )
            if first_track_box is not None:
                seed_frame, seed_box = first_track_box
                seed_source = "target_track_first_visible_frame"
        if seed_box is None:
            if seed_frame == prompt_frame and prompt.get("box") is not None:
                seed_box = np.asarray(prompt["box"], dtype=np.float32)
            elif track_boxes:
                nearest_prompt_box = _nearest_track_box(
                    frame_index=prompt_frame,
                    track_boxes=track_boxes,
                    max_distance=max(12, frame_count // 12),
                )
                if nearest_prompt_box is not None:
                    seed_frame, seed_box = nearest_prompt_box
                    seed_source = "target_track_nearest_prompt_frame"
        visual_box = seed_box if seed_box is not None else prompt.get("box")
        seed_points, seed_labels = _seed_points_for_frame(
            prompt=prompt,
            box=seed_box,
            seed_frame=seed_frame,
            prompt_frame=prompt_frame,
        )
        diagnostics["prompt_frame"] = prompt_frame
        diagnostics["seed_frame"] = seed_frame
        diagnostics["seed_source"] = seed_source

        with torch.inference_mode():
            active_obj_id = obj_id
            if visual_box is not None:
                diagnostics["visual_box_prompt"] = True
                box_response = predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": seed_frame,
                        **_box_prompt_tensors(
                            box=np.asarray(visual_box, dtype=np.float32),
                            width=width,
                            height=height,
                        ),
                    }
                )
                box_outputs = box_response.get("outputs", {})
                box_obj_id = _first_nonempty_output_object_id(box_outputs)
                if box_obj_id is not None:
                    active_obj_id = box_obj_id
                    diagnostics["visual_box_obj_id"] = box_obj_id
                    diagnostics["active_obj_id"] = active_obj_id
                box_mask = _mask_from_outputs(box_outputs, obj_id=active_obj_id)
                if box_mask is not None:
                    masks_by_frame[seed_frame] = box_mask

            if seed_points is not None and seed_labels is not None:
                prompt_inputs = _point_prompt_tensors(
                    points=seed_points,
                    labels=seed_labels,
                    width=width,
                    height=height,
                )
            else:
                prompt_inputs = _relative_prompt_tensors(prompt=prompt, width=width, height=height)
            prompt_response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": seed_frame,
                    "obj_id": active_obj_id,
                    **prompt_inputs,
                }
            )
            prompt_outputs = prompt_response.get("outputs", {})
            prompt_obj_id = _first_nonempty_output_object_id(prompt_outputs)
            if prompt_obj_id is not None:
                active_obj_id = prompt_obj_id
                diagnostics["point_prompt_obj_id"] = prompt_obj_id
                diagnostics["active_obj_id"] = active_obj_id
            prompt_mask = _mask_from_outputs(prompt_outputs, obj_id=active_obj_id)
            if prompt_mask is not None:
                masks_by_frame[seed_frame] = prompt_mask

            diagnostics["propagation"].extend(
                _propagate_from_seed_frame(
                    predictor=predictor,
                    session_id=session_id,
                    active_obj_id=active_obj_id,
                    seed_frame=seed_frame,
                    masks_by_frame=masks_by_frame,
                    label="primary_prompt",
                    directions=("forward",) if seed_frame == 0 else ("forward", "backward"),
                )
            )

            track_frame_target = len(track_boxes or {}) or frame_count
            anchor_refine_threshold = max(12, int(track_frame_target * 0.65))
            needs_anchor_refinement = len(masks_by_frame) < anchor_refine_threshold
            diagnostics["anchor_refinement_threshold"] = anchor_refine_threshold
            diagnostics["anchor_refinement_triggered"] = needs_anchor_refinement
            if track_boxes:
                anchor_frames = (
                    _track_prompt_anchors(
                        prompt_frame=seed_frame,
                        frame_count=frame_count,
                        track_boxes=track_boxes,
                    )
                    if needs_anchor_refinement
                    else []
                )
                diagnostics["anchor_refinement_frames"] = anchor_frames
                for anchor_frame in anchor_frames:
                    if anchor_frame == seed_frame:
                        continue
                    points = _support_points_from_box(track_boxes[anchor_frame])
                    labels = np.ones(len(points), dtype=np.int32)
                    anchor_response = predictor.handle_request(
                        request={
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": anchor_frame,
                            "obj_id": active_obj_id,
                            **_point_prompt_tensors(
                                points=points,
                                labels=labels,
                                width=width,
                                height=height,
                            ),
                        }
                    )
                    anchor_mask = _mask_from_outputs(
                        anchor_response.get("outputs", {}),
                        obj_id=active_obj_id,
                    )
                    if anchor_mask is not None:
                        masks_by_frame[anchor_frame] = anchor_mask

                if anchor_frames:
                    diagnostics["propagation"].extend(
                        _propagate_from_seed_frame(
                            predictor=predictor,
                            session_id=session_id,
                            active_obj_id=active_obj_id,
                            seed_frame=seed_frame,
                            masks_by_frame=masks_by_frame,
                            label="anchor_refinement",
                            directions=("forward",) if seed_frame == 0 else ("forward", "backward"),
                        )
                    )
            else:
                diagnostics["anchor_refinement_frames"] = []
    finally:
        predictor.handle_request(
            request={
                "type": "close_session",
                "session_id": session_id,
            }
        )
    return masks_by_frame, diagnostics


def _mask_from_track_box(
    box: np.ndarray,
    *,
    width: int,
    height: int,
    padding_ratio: float = 0.04,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = max(2.0, box_width * padding_ratio)
    pad_y = max(2.0, box_height * padding_ratio)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 1
    return mask


def _interpolated_track_box(
    *,
    frame_index: int,
    track_boxes: dict[int, np.ndarray],
    sorted_indices: list[int],
    max_gap: int,
) -> tuple[np.ndarray | None, bool]:
    exact = track_boxes.get(frame_index)
    if exact is not None:
        return exact, False

    previous_index = None
    next_index = None
    for index in sorted_indices:
        if index < frame_index:
            previous_index = index
        elif index > frame_index:
            next_index = index
            break

    previous_gap = frame_index - previous_index if previous_index is not None else None
    next_gap = next_index - frame_index if next_index is not None else None
    if previous_gap is not None and next_gap is not None and previous_gap + next_gap <= max_gap:
        start_box = track_boxes[previous_index]
        end_box = track_boxes[next_index]
        ratio = previous_gap / float(previous_gap + next_gap)
        return (start_box * (1.0 - ratio)) + (end_box * ratio), True
    if previous_gap is not None and previous_gap <= max_gap:
        return track_boxes[previous_index], True
    if next_gap is not None and next_gap <= max_gap:
        return track_boxes[next_index], True
    return None, False


def _build_track_box_fallback_masks(
    track_boxes: dict[int, np.ndarray],
    *,
    width: int,
    height: int,
    frame_count: int,
    max_interpolation_gap: int = 18,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    sorted_indices = sorted(index for index in track_boxes if 0 <= index < frame_count)
    if not sorted_indices:
        return {}, {
            "backend": "identity_track_box",
            "generated_frames": 0,
            "track_box_frames": 0,
            "interpolated_frames": 0,
        }

    masks_by_frame: dict[int, np.ndarray] = {}
    interpolated_frames = 0
    for frame_index in range(frame_count):
        box, interpolated = _interpolated_track_box(
            frame_index=frame_index,
            track_boxes=track_boxes,
            sorted_indices=sorted_indices,
            max_gap=max_interpolation_gap,
        )
        if box is None:
            continue
        mask = _mask_from_track_box(box, width=width, height=height)
        if mask.any():
            masks_by_frame[frame_index] = mask
            if interpolated:
                interpolated_frames += 1

    return masks_by_frame, {
        "backend": "identity_track_box",
        "generated_frames": len(masks_by_frame),
        "track_box_frames": len(sorted_indices),
        "interpolated_frames": interpolated_frames,
        "max_interpolation_gap": max_interpolation_gap,
    }


def _mask_box_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        dtype=np.float32,
    )


def _box_iou_xyxy(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _filter_masks_to_track_boxes(
    masks_by_frame: dict[int, np.ndarray],
    track_boxes: dict[int, np.ndarray],
    *,
    min_iou: float = 0.025,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if not track_boxes:
        return masks_by_frame, {"enabled": False}

    filtered: dict[int, np.ndarray] = {}
    rejected = 0
    unchecked = 0
    for frame_index, mask in masks_by_frame.items():
        target_box = track_boxes.get(frame_index)
        if target_box is None:
            filtered[frame_index] = mask
            unchecked += 1
            continue
        if _box_iou_xyxy(_mask_box_xyxy(mask), target_box) >= min_iou:
            filtered[frame_index] = mask
        else:
            rejected += 1
    return filtered, {
        "enabled": True,
        "accepted_frames": len(filtered),
        "rejected_frames": rejected,
        "unchecked_frames": unchecked,
        "min_iou": min_iou,
    }


def _identity_track_box_fallback_masks(
    *,
    paths: dict[str, Any],
    width: int,
    height: int,
    frame_count: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    track_boxes = _load_identity_track_boxes(paths=paths, width=width, height=height)
    return _build_track_box_fallback_masks(
        track_boxes,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def _load_identity_track_boxes(
    *,
    paths: dict[str, Any],
    width: int,
    height: int,
) -> dict[int, np.ndarray]:
    from whodoirunlike.sam31_mlx_runner import load_track_boxes

    return load_track_boxes(
        {
            "tracklets_jsonl": paths.get("tracklets_jsonl"),
            "tracklets": paths.get("tracklets"),
        },
        width=width,
        height=height,
    )


def update_manifest_after_sam31_gpu(
    manifest_path: Path,
    metadata_path: Path,
    masks_jsonl_path: Path,
    *,
    checkpoint_path: str | None,
    elapsed_seconds: float,
    mask_summary: dict[str, Any],
    fallback: dict[str, Any] | None = None,
    prompt_summary: dict[str, Any] | None = None,
) -> None:
    manifest = read_json(manifest_path)
    stages = manifest.setdefault("stages", {})
    whole_runner_mask = stages.setdefault("whole_runner_mask", {})
    whole_runner_mask["status"] = "complete"
    whole_runner_mask["recommended_tool"] = "SAM 3.1 CUDA via facebookresearch/sam3"
    whole_runner_mask["backend"] = "sam31_gpu"
    whole_runner_mask["model"] = DEFAULT_SAM31_GPU_MODEL
    whole_runner_mask["checkpoint_path"] = checkpoint_path
    whole_runner_mask["elapsed_seconds"] = round(elapsed_seconds, 3)
    whole_runner_mask["metadata"] = str(metadata_path)
    whole_runner_mask["masks_jsonl"] = str(masks_jsonl_path)
    whole_runner_mask["mask_summary"] = mask_summary
    if prompt_summary:
        whole_runner_mask["prompting"] = prompt_summary
    if fallback:
        whole_runner_mask["fallback"] = fallback
    else:
        whole_runner_mask.pop("fallback", None)
    whole_runner_mask.pop("error", None)
    manifest.setdefault("paths", {})["masks_jsonl"] = str(masks_jsonl_path)
    stages.setdefault("renders", {})["status"] = "partial_complete"
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def run_sam31_gpu_mask(
    *,
    run_dir: Path,
    checkpoint_path: str | None = None,
    force_frames: bool = False,
    obj_id: int = DEFAULT_SAM31_GPU_OBJ_ID,
) -> dict[str, Any]:
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM 3.1 GPU dependencies are not installed. Install the official "
            "facebookresearch/sam3 package in a CUDA 12.6+ / PyTorch 2.7+ environment."
        ) from exc

    start = time.perf_counter()
    manifest_path = run_dir / "cv_run_manifest.json"
    manifest = read_json(manifest_path)
    paths = manifest["paths"]
    source_segment = Path(paths["source_segment"])
    prompt_path = Path(paths["person_prompt"])
    runner_mask_path = Path(paths["runner_mask"])
    masked_runner_path = Path(paths["masked_runner"])
    qa_overlay_path = Path(paths["qa_overlay"])
    metadata_path = run_dir / "runner_mask_metadata.jsonl"
    masks_jsonl_path = Path(str(paths.get("masks_jsonl") or run_dir / "masks.jsonl"))
    frame_dir = run_dir / "sam31_gpu_frames"

    video_meta = inspect_video(source_segment)
    frame_paths = extract_video_frames(source_segment, frame_dir, force=force_frames)
    prompt = load_prompt(prompt_path, video_meta["width"], video_meta["height"])
    prompt_frame = max(0, min(prompt["frame_index"], len(frame_paths) - 1))
    track_boxes = _load_identity_track_boxes(
        paths=paths,
        width=video_meta["width"],
        height=video_meta["height"],
    )
    track_prompt_anchors = _track_prompt_anchors(
        prompt_frame=prompt_frame,
        frame_count=len(frame_paths),
        track_boxes=track_boxes,
    )
    prompt_summary = {
        "mode": "point_track_anchors" if track_boxes else "point_or_box_prompt",
        "uses_user_points": _has_prompt_points(prompt),
        "track_box_frames": len(track_boxes),
        "anchor_frames": track_prompt_anchors,
    }
    resolved_checkpoint = checkpoint_path or os.getenv("WHODOIRUNLIKE_SAM31_GPU_CHECKPOINT")
    resolved_checkpoint = resolved_checkpoint.strip() if resolved_checkpoint else None

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=resolved_checkpoint,
        use_fa3=_env_bool("WHODOIRUNLIKE_SAM31_GPU_USE_FA3", default=False),
        compile=_env_bool("WHODOIRUNLIKE_SAM31_GPU_COMPILE", default=False),
        warm_up=_env_bool("WHODOIRUNLIKE_SAM31_GPU_WARM_UP", default=False),
        default_output_prob_thresh=float(os.getenv("WHODOIRUNLIKE_SAM31_GPU_THRESHOLD", "0.5")),
    )
    _patch_multiplex_init_state_kwargs(predictor)
    tracker_config = _configure_interactive_tracker_for_user_prompt(predictor)
    masks_by_frame, sam_prompt_diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=source_segment,
        prompt=prompt,
        width=video_meta["width"],
        height=video_meta["height"],
        frame_count=len(frame_paths),
        obj_id=obj_id,
        track_boxes=track_boxes,
    )
    prompt_summary["sam31"] = sam_prompt_diagnostics
    prompt_summary["sam31_tracker_config"] = tracker_config
    masks_by_frame, identity_filter = _filter_masks_to_track_boxes(masks_by_frame, track_boxes)
    prompt_summary["identity_filter"] = identity_filter

    write_mask_outputs(
        frame_paths=frame_paths,
        masks_by_frame=masks_by_frame,
        fps=video_meta["fps"],
        runner_mask_path=runner_mask_path,
        masked_runner_path=masked_runner_path,
        qa_overlay_path=qa_overlay_path,
        metadata_path=metadata_path,
    )
    mask_summary = write_masks_jsonl_from_video(runner_mask_path, masks_jsonl_path)
    fallback: dict[str, Any] | None = None
    nonempty_frames = int(mask_summary.get("nonempty_frames") or 0)
    target_frame_count = len(track_boxes)
    sparse_threshold = int(max(1, target_frame_count * 0.65)) if target_frame_count else 0
    fallback_reason = None
    if nonempty_frames == 0:
        fallback_reason = "sam31_gpu_empty_mask"
    elif sparse_threshold and nonempty_frames < sparse_threshold:
        fallback_reason = "sam31_gpu_sparse_or_off_target_mask"

    if fallback_reason:
        fallback_masks, fallback = _identity_track_box_fallback_masks(
            paths=paths,
            width=video_meta["width"],
            height=video_meta["height"],
            frame_count=len(frame_paths),
        )
        fallback["reason"] = fallback_reason
        fallback["sam_detected_frames_before_fallback"] = nonempty_frames
        fallback["preserved_sam_frames"] = len(masks_by_frame)
        if fallback_masks:
            combined_masks = dict(fallback_masks)
            combined_masks.update(masks_by_frame)
            masks_by_frame = combined_masks
            write_mask_outputs(
                frame_paths=frame_paths,
                masks_by_frame=masks_by_frame,
                fps=video_meta["fps"],
                runner_mask_path=runner_mask_path,
                masked_runner_path=masked_runner_path,
                qa_overlay_path=qa_overlay_path,
                metadata_path=metadata_path,
            )
            mask_summary = write_masks_jsonl_from_video(runner_mask_path, masks_jsonl_path)
            fallback["nonempty_frames_after_fallback"] = int(mask_summary.get("nonempty_frames") or 0)
        else:
            fallback["nonempty_frames_after_fallback"] = 0
    elapsed_seconds = time.perf_counter() - start
    update_manifest_after_sam31_gpu(
        manifest_path,
        metadata_path,
        masks_jsonl_path,
        checkpoint_path=resolved_checkpoint,
        elapsed_seconds=elapsed_seconds,
        mask_summary=mask_summary,
        fallback=fallback,
        prompt_summary=prompt_summary,
    )
    return {
        "candidate_id": manifest["candidate_id"],
        "backend": "sam31_gpu",
        "model": DEFAULT_SAM31_GPU_MODEL,
        "checkpoint_path": resolved_checkpoint,
        "frame_count": len(frame_paths),
        "prompt_frame": prompt_frame,
        "box_source": prompt["box_source"],
        "detected_frames": len(masks_by_frame),
        "prompting": prompt_summary,
        "fallback": fallback,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "runner_mask": str(runner_mask_path),
        "masked_runner": str(masked_runner_path),
        "qa_overlay": str(qa_overlay_path),
        "metadata": str(metadata_path),
        "masks_jsonl": str(masks_jsonl_path),
        "mask_summary": mask_summary,
    }
