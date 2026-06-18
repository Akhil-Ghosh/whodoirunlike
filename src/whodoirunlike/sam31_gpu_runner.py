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
) -> tuple[Any, Any]:
    import torch

    points = prompt.get("points")
    labels = prompt.get("labels")
    if points is None or len(points) == 0:
        box = prompt.get("box")
        if box is None:
            raise ValueError("SAM 3.1 GPU requires at least one prompt point or box.")
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        points = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

    rel_points = np.asarray(points, dtype=np.float32).copy()
    rel_points[:, 0] = rel_points[:, 0] / max(float(width), 1.0)
    rel_points[:, 1] = rel_points[:, 1] / max(float(height), 1.0)
    rel_points = np.clip(rel_points, 0.0, 1.0)
    rel_labels = np.asarray(labels, dtype=np.int32)

    return (
        torch.tensor(rel_points, dtype=torch.float32, device="cuda"),
        torch.tensor(rel_labels, dtype=torch.int32, device="cuda"),
    )


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


def _collect_sam31_masks(
    *,
    predictor: Any,
    video_path: Path,
    prompt: dict[str, Any],
    width: int,
    height: int,
    frame_count: int,
    obj_id: int,
) -> dict[int, np.ndarray]:
    import torch

    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(video_path),
        }
    )
    session_id = response["session_id"]
    masks_by_frame: dict[int, np.ndarray] = {}
    try:
        prompt_frame = max(0, min(int(prompt["frame_index"]), max(frame_count - 1, 0)))
        points, point_labels = _relative_prompt_tensors(prompt=prompt, width=width, height=height)
        with torch.inference_mode():
            prompt_response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_frame,
                    "points": points,
                    "point_labels": point_labels,
                    "obj_id": obj_id,
                }
            )
            prompt_mask = _mask_from_outputs(prompt_response.get("outputs", {}), obj_id=obj_id)
            if prompt_mask is not None:
                masks_by_frame[prompt_frame] = prompt_mask

            for stream_response in predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                }
            ):
                frame_index = int(stream_response["frame_index"])
                mask = _mask_from_outputs(stream_response.get("outputs", {}), obj_id=obj_id)
                if mask is not None:
                    masks_by_frame[frame_index] = mask
    finally:
        predictor.handle_request(
            request={
                "type": "close_session",
                "session_id": session_id,
            }
        )
    return masks_by_frame


def update_manifest_after_sam31_gpu(
    manifest_path: Path,
    metadata_path: Path,
    masks_jsonl_path: Path,
    *,
    checkpoint_path: str | None,
    elapsed_seconds: float,
    mask_summary: dict[str, Any],
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
    masks_by_frame = _collect_sam31_masks(
        predictor=predictor,
        video_path=source_segment,
        prompt=prompt,
        width=video_meta["width"],
        height=video_meta["height"],
        frame_count=len(frame_paths),
        obj_id=obj_id,
    )

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
    elapsed_seconds = time.perf_counter() - start
    update_manifest_after_sam31_gpu(
        manifest_path,
        metadata_path,
        masks_jsonl_path,
        checkpoint_path=resolved_checkpoint,
        elapsed_seconds=elapsed_seconds,
        mask_summary=mask_summary,
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
        "elapsed_seconds": round(elapsed_seconds, 3),
        "runner_mask": str(runner_mask_path),
        "masked_runner": str(masked_runner_path),
        "qa_overlay": str(qa_overlay_path),
        "metadata": str(metadata_path),
        "masks_jsonl": str(masks_jsonl_path),
        "mask_summary": mask_summary,
    }
