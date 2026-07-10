from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.mask_artifacts import write_masks_jsonl_from_video
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.video_io import make_browser_playable_mp4s


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def extract_video_frames(video_path: Path, frame_dir: Path, *, force: bool = False) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.jpg"))
    if existing and not force:
        return existing

    for frame_path in existing:
        frame_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source segment: {video_path}")

    index = 0
    frame_paths: list[Path] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = frame_dir / f"{index:05d}.jpg"
        if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            cap.release()
            raise ValueError(f"Could not write frame: {frame_path}")
        frame_paths.append(frame_path)
        index += 1
    cap.release()

    if not frame_paths:
        raise ValueError(f"No frames extracted from {video_path}")
    return frame_paths


def inspect_video(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return meta


def normalized_points_to_pixels(points: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [
                float(point["x"]) * max(width - 1, 1),
                float(point["y"]) * max(height - 1, 1),
            ]
            for point in points
        ],
        dtype=np.float32,
    )


def normalized_box_to_pixels(box: dict[str, Any] | None, width: int, height: int) -> np.ndarray | None:
    if not box:
        return None
    x1 = float(box["x"]) * max(width - 1, 1)
    y1 = float(box["y"]) * max(height - 1, 1)
    x2 = (float(box["x"]) + float(box["width"])) * max(width - 1, 1)
    y2 = (float(box["y"]) + float(box["height"])) * max(height - 1, 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_from_positive_points(points: list[dict[str, Any]], margin: float = 0.06) -> dict[str, float] | None:
    if len(points) < 2:
        return None
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    x1 = max(0.0, min(xs) - margin)
    y1 = max(0.0, min(ys) - margin)
    x2 = min(1.0, max(xs) + margin)
    y2 = min(1.0, max(ys) + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x": round(x1, 6),
        "y": round(y1, 6),
        "width": round(x2 - x1, 6),
        "height": round(y2 - y1, 6),
    }


def load_prompt(prompt_path: Path, width: int, height: int) -> dict[str, Any]:
    prompt = read_json(prompt_path)
    selection = prompt.get("selection", {})
    positive = selection.get("positive_points", [])
    negative = selection.get("negative_points", [])
    if selection.get("type") in (None, "", "unset") or (not positive and not selection.get("box")):
        raise ValueError(f"Prompt is unset. Save points or a box first: {prompt_path}")

    points: list[list[float]] = []
    labels: list[int] = []
    if positive:
        points.extend(normalized_points_to_pixels(positive, width, height).tolist())
        labels.extend([1] * len(positive))
    if negative:
        points.extend(normalized_points_to_pixels(negative, width, height).tolist())
        labels.extend([0] * len(negative))

    frame_index = prompt.get("frame", {}).get("frame_index")
    box = selection.get("box") or box_from_positive_points(positive)
    return {
        "frame_index": int(frame_index or 0),
        "points": np.array(points, dtype=np.float32) if points else None,
        "labels": np.array(labels, dtype=np.int32) if labels else None,
        "box": normalized_box_to_pixels(box, width, height),
        "box_source": "explicit" if selection.get("box") else "positive_points" if box else None,
        "raw": prompt,
    }


def mask_to_uint8(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.squeeze(mask)
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype("uint8"), (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype("uint8") * 255


def contour_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    green = np.zeros_like(frame)
    green[:, :, 1] = 220
    alpha_mask = (mask > 0)[:, :, None]
    overlay = np.where(alpha_mask, (frame * 0.55 + green * 0.45).astype("uint8"), overlay)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 250, 243), 2)
    return overlay


def write_mask_outputs(
    *,
    frame_paths: list[Path],
    masks_by_frame: dict[int, np.ndarray],
    fps: float,
    runner_mask_path: Path,
    masked_runner_path: Path,
    qa_overlay_path: Path,
    metadata_path: Path,
) -> None:
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Could not read first extracted frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    runner_mask_path.parent.mkdir(parents=True, exist_ok=True)
    masked_runner_path.parent.mkdir(parents=True, exist_ok=True)
    qa_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    mask_writer = cv2.VideoWriter(str(runner_mask_path), fourcc, fps, (width, height), True)
    masked_writer = cv2.VideoWriter(str(masked_runner_path), fourcc, fps, (width, height), True)
    qa_writer = cv2.VideoWriter(str(qa_overlay_path), fourcc, fps, (width, height), True)
    if not mask_writer.isOpened() or not masked_writer.isOpened() or not qa_writer.isOpened():
        raise ValueError("Could not open one or more output video writers")

    previous_centroid: tuple[float, float] | None = None
    metadata_rows: list[dict[str, Any]] = []
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        mask = mask_to_uint8(masks_by_frame.get(frame_index, np.zeros((height, width))), height, width)
        mask_bool = mask > 0
        area = int(mask_bool.sum())
        area_ratio = area / float(height * width)
        if area:
            ys, xs = np.where(mask_bool)
            centroid = (float(xs.mean()), float(ys.mean()))
        else:
            centroid = None

        if centroid and previous_centroid:
            centroid_delta = float(
                ((centroid[0] - previous_centroid[0]) ** 2 + (centroid[1] - previous_centroid[1]) ** 2)
                ** 0.5
            )
        else:
            centroid_delta = None
        if centroid:
            previous_centroid = centroid

        reason = None
        if area_ratio < 0.002:
            reason = "mask_missing_or_tiny"
        elif centroid_delta and centroid_delta > width * 0.28:
            reason = "centroid_jump_identity_risk"

        mask_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        masked = np.zeros_like(frame)
        masked[mask_bool] = frame[mask_bool]
        masked_writer.write(masked)
        qa_writer.write(contour_overlay(frame, mask))
        metadata_rows.append(
            {
                "frame_index": frame_index,
                "mask_area_ratio": round(area_ratio, 6),
                "centroid": [round(centroid[0], 2), round(centroid[1], 2)] if centroid else None,
                "centroid_delta_px": round(centroid_delta, 2) if centroid_delta else None,
                "usable": reason is None,
                "drop_reason": reason,
            }
        )

    mask_writer.release()
    masked_writer.release()
    qa_writer.release()
    make_browser_playable_mp4s([runner_mask_path, masked_runner_path, qa_overlay_path])
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row) + "\n")


def update_manifest_after_sam2(
    manifest_path: Path,
    metadata_path: Path,
    masks_jsonl_path: Path | None = None,
    mask_summary: dict[str, Any] | None = None,
) -> None:
    clip_run = RunningClipRun(manifest_path.parent)
    manifest = clip_run.read_manifest()
    values: dict[str, Any] = {
        "status": "complete",
        "metadata": str(metadata_path),
    }
    if masks_jsonl_path:
        values["masks_jsonl"] = str(masks_jsonl_path)
    if mask_summary:
        values["mask_summary"] = mask_summary
    manifest["updated_at"] = utc_now_iso()
    clip_run.update_stages(
        {
            "whole_runner_mask": values,
            "renders": {"status": "partial_complete"},
        },
        manifest=manifest,
    )


def choose_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_sam2_mask(
    *,
    run_dir: Path,
    checkpoint: Path,
    model_cfg: str,
    device: str | None = None,
    force_frames: bool = False,
) -> dict[str, Any]:
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM 2 is not installed. Install torch/torchvision, then install "
            "facebookresearch/sam2 with SAM2_BUILD_CUDA=0 on this Mac."
        ) from exc

    clip_run = RunningClipRun(run_dir)
    manifest_path = clip_run.manifest_path
    manifest = clip_run.read_manifest()
    source_segment = clip_run.artifact_path("source_segment", manifest)
    prompt_path = clip_run.artifact_path("person_prompt", manifest)
    runner_mask_path = clip_run.artifact_path("runner_mask", manifest)
    masked_runner_path = clip_run.artifact_path("masked_runner", manifest)
    qa_overlay_path = clip_run.artifact_path("qa_overlay", manifest)
    metadata_path = clip_run.artifact_path("runner_mask_metadata", manifest)
    masks_jsonl_path = clip_run.artifact_path("masks_jsonl", manifest)
    frame_dir = run_dir / "sam2_frames"

    video_meta = inspect_video(source_segment)
    frame_paths = extract_video_frames(source_segment, frame_dir, force=force_frames)
    prompt = load_prompt(prompt_path, video_meta["width"], video_meta["height"])
    prompt_frame = max(0, min(prompt["frame_index"], len(frame_paths) - 1))
    selected_device = device or choose_device()

    predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=selected_device)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if selected_device == "cuda" else nullcontext()
    )

    with torch.inference_mode(), autocast_context:
        inference_state = predictor.init_state(video_path=str(frame_dir))
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_frame,
            obj_id=1,
            points=prompt["points"],
            labels=prompt["labels"],
            box=prompt["box"],
        )
        masks_by_frame: dict[int, np.ndarray] = {}
        for reverse in (False, True):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                reverse=reverse,
            ):
                for index, out_obj_id in enumerate(out_obj_ids):
                    if int(out_obj_id) == 1:
                        masks_by_frame[int(out_frame_idx)] = (
                            (out_mask_logits[index] > 0.0).detach().cpu().numpy()
                        )
                        break

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
    update_manifest_after_sam2(manifest_path, metadata_path, masks_jsonl_path, mask_summary)
    return {
        "candidate_id": manifest["candidate_id"],
        "device": selected_device,
        "frame_count": len(frame_paths),
        "prompt_frame": prompt_frame,
        "box_source": prompt["box_source"],
        "runner_mask": str(runner_mask_path),
        "masked_runner": str(masked_runner_path),
        "qa_overlay": str(qa_overlay_path),
        "metadata": str(metadata_path),
        "masks_jsonl": str(masks_jsonl_path),
        "mask_summary": mask_summary,
    }
