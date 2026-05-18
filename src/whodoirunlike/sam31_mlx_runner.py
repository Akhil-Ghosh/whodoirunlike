from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.sam2_runner import (
    extract_video_frames,
    inspect_video,
    load_prompt,
    read_json,
    write_json,
    write_mask_outputs,
)

DEFAULT_SAM31_MLX_MODEL = "mlx-community/sam3.1-bf16"
DEFAULT_SAM31_PROMPTS = ("a runner", "a person")


def box_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
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


def box_center(box: np.ndarray | None) -> tuple[float, float] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def normalized_prompt_anchor(prompt: dict[str, Any], width: int, height: int) -> tuple[float, float] | None:
    selection = prompt.get("raw", {}).get("selection", {})
    positive = selection.get("positive_points") or []
    if not positive:
        return box_center(prompt.get("box"))
    xs = [float(point["x"]) * max(width - 1, 1) for point in positive]
    ys = [float(point["y"]) * max(height - 1, 1) for point in positive]
    return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return (float(xs.mean()), float(ys.mean()))


def mask_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def point_score(point: tuple[float, float] | None, box: np.ndarray, width: int, height: int) -> float:
    if point is None:
        return 0.0
    center = box_center(box)
    if center is None:
        return 0.0
    diagonal = max((width**2 + height**2) ** 0.5, 1.0)
    distance = ((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / diagonal)


def choose_detection_index(
    *,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    prompt_box: np.ndarray | None = None,
    prompt_anchor: tuple[float, float] | None = None,
    previous_box: np.ndarray | None = None,
) -> int | None:
    if len(scores) == 0:
        return None

    best_index: int | None = None
    best_score = -1.0
    for index, score in enumerate(scores):
        box = boxes[index]
        mask = masks[index]
        mask_area = float((mask > 0).sum()) / float(width * height)
        continuity = box_iou(box, previous_box)
        prompt_overlap = box_iou(box, prompt_box)
        anchor = point_score(prompt_anchor, box, width, height)
        compact_area = 1.0 if 0.006 <= mask_area <= 0.18 else 0.25
        total = (
            float(score) * 0.45
            + continuity * 2.25
            + prompt_overlap * (1.6 if previous_box is None else 0.4)
            + anchor * (0.85 if previous_box is None else 0.25)
            + compact_area * 0.2
        )
        if total > best_score:
            best_score = total
            best_index = index
    return best_index


def detect_frame(
    *,
    frame_path: Path,
    predictor: Any,
    prompts: Sequence[str],
    threshold: float,
) -> tuple[np.ndarray, Any]:
    try:
        import mlx.core as mx
        from mlx_vlm.models.sam3_1.generate import _detect_with_backbone, _get_backbone_features
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM 3.1 MLX dependencies are not installed. Run: python -m pip install 'mlx-vlm>=0.4.3'"
        ) from exc

    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        raise ValueError(f"Could not read frame: {frame_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = predictor.processor.preprocess_image(image)
    pixel_values = mx.array(inputs["pixel_values"])
    backbone_features = _get_backbone_features(predictor.model, pixel_values)
    result = _detect_with_backbone(
        predictor,
        backbone_features,
        list(prompts),
        image.size,
        threshold,
        encoder_cache={},
    )
    return frame_bgr, result


def update_manifest_after_sam31_mlx(
    manifest_path: Path,
    metadata_path: Path,
    *,
    model_path: str,
    prompts: Sequence[str],
    elapsed_seconds: float,
) -> None:
    manifest = read_json(manifest_path)
    stages = manifest.setdefault("stages", {})
    whole_runner_mask = stages.setdefault("whole_runner_mask", {})
    whole_runner_mask["status"] = "complete"
    whole_runner_mask["backend"] = "sam31_mlx"
    whole_runner_mask["model"] = model_path
    whole_runner_mask["prompts"] = list(prompts)
    whole_runner_mask["elapsed_seconds"] = round(elapsed_seconds, 3)
    whole_runner_mask["metadata"] = str(metadata_path)
    stages.setdefault("renders", {})["status"] = "partial_complete"
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def run_sam31_mlx_mask(
    *,
    run_dir: Path,
    model_path: str = DEFAULT_SAM31_MLX_MODEL,
    prompts: Sequence[str] = DEFAULT_SAM31_PROMPTS,
    threshold: float = 0.18,
    resolution: int = 1008,
    force_frames: bool = False,
) -> dict[str, Any]:
    try:
        from mlx_vlm.generate import wired_limit
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.utils import get_model_path, load_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM 3.1 MLX dependencies are not installed. Run: python -m pip install 'mlx-vlm>=0.4.3'"
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
    frame_dir = run_dir / "sam31_mlx_frames"

    video_meta = inspect_video(source_segment)
    frame_paths = extract_video_frames(source_segment, frame_dir, force=force_frames)
    prompt = load_prompt(prompt_path, video_meta["width"], video_meta["height"])
    prompt_frame = max(0, min(prompt["frame_index"], len(frame_paths) - 1))
    prompt_box = prompt.get("box")
    prompt_anchor = normalized_prompt_anchor(prompt, video_meta["width"], video_meta["height"])

    resolved_model_path = get_model_path(model_path)
    model = load_model(resolved_model_path)
    processor = Sam31Processor.from_pretrained(str(resolved_model_path))
    if resolution != 1008:
        processor.image_size = resolution
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)

    masks_by_frame: dict[int, np.ndarray] = {}
    selected_boxes: dict[int, np.ndarray] = {}
    detection_counts: dict[int, int] = {}

    def process_sequence(indices: Sequence[int], initial_box: np.ndarray | None) -> None:
        previous_box = initial_box
        with wired_limit(model):
            for frame_index in indices:
                _, result = detect_frame(
                    frame_path=frame_paths[frame_index],
                    predictor=predictor,
                    prompts=prompts,
                    threshold=threshold,
                )
                detection_counts[frame_index] = int(len(result.scores))
                selected_index = choose_detection_index(
                    boxes=result.boxes,
                    masks=result.masks,
                    scores=result.scores,
                    width=video_meta["width"],
                    height=video_meta["height"],
                    prompt_box=prompt_box,
                    prompt_anchor=prompt_anchor,
                    previous_box=previous_box,
                )
                if selected_index is None:
                    continue
                mask = result.masks[selected_index].astype("uint8")
                selected_box = mask_box(mask)
                if selected_box is None:
                    selected_box = result.boxes[selected_index]
                masks_by_frame[frame_index] = mask
                selected_boxes[frame_index] = selected_box
                previous_box = selected_box

    process_sequence(list(range(prompt_frame, len(frame_paths))), None)
    prompt_selected_box = selected_boxes.get(prompt_frame, prompt_box)
    if prompt_frame > 0:
        process_sequence(list(range(prompt_frame - 1, -1, -1)), prompt_selected_box)

    write_mask_outputs(
        frame_paths=frame_paths,
        masks_by_frame=masks_by_frame,
        fps=video_meta["fps"],
        runner_mask_path=runner_mask_path,
        masked_runner_path=masked_runner_path,
        qa_overlay_path=qa_overlay_path,
        metadata_path=metadata_path,
    )
    elapsed_seconds = time.perf_counter() - start
    update_manifest_after_sam31_mlx(
        manifest_path,
        metadata_path,
        model_path=model_path,
        prompts=prompts,
        elapsed_seconds=elapsed_seconds,
    )
    return {
        "candidate_id": manifest["candidate_id"],
        "backend": "sam31_mlx",
        "model": model_path,
        "prompts": list(prompts),
        "threshold": threshold,
        "resolution": resolution,
        "frame_count": len(frame_paths),
        "prompt_frame": prompt_frame,
        "detected_frames": len(masks_by_frame),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "runner_mask": str(runner_mask_path),
        "masked_runner": str(masked_runner_path),
        "qa_overlay": str(qa_overlay_path),
        "metadata": str(metadata_path),
    }
