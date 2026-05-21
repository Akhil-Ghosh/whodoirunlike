from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_MANIFEST = REPO_ROOT / "artifacts/evaluation/video_candidates.review20_best.json"
DEFAULT_ANNOTATIONS = REPO_ROOT / "artifacts/review/clip_reviews.json"
DEFAULT_CV_RUN_ROOT = REPO_ROOT / "artifacts/cv_runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


@dataclass(frozen=True)
class ReviewedClip:
    candidate_id: str
    runner_name: str
    runner_slug: str
    title: str
    source_url: str
    channel: str
    video_path: Path
    quality: str
    camera_angle: str
    start_seconds: float
    end_seconds: float
    notes: str
    primary_bucket: str

    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 2)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_reviewed_clip(
    *,
    candidate_id: str | None = None,
    quality: str = "good",
    manifest_path: Path = DEFAULT_REVIEW_MANIFEST,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
) -> ReviewedClip:
    manifest_rows = read_json(manifest_path)
    annotations = read_json(annotations_path).get("annotations", {})
    if not isinstance(manifest_rows, list) or not isinstance(annotations, dict):
        raise ValueError("Expected review manifest list and annotations map")

    rows_by_id = {str(row.get("candidate_id")): row for row in manifest_rows}
    selected_id = candidate_id
    if selected_id is None:
        selected_id = next(
            (
                annotation_id
                for annotation_id, annotation in annotations.items()
                if annotation.get("quality") == quality
            ),
            None,
        )
    if selected_id is None or selected_id not in rows_by_id:
        raise ValueError(f"No reviewed clip found for candidate_id={candidate_id!r}, quality={quality!r}")

    row = rows_by_id[selected_id]
    annotation = annotations.get(selected_id, {})
    start_seconds = _as_float(annotation.get("start_seconds"))
    end_seconds = _as_float(annotation.get("end_seconds"))
    if start_seconds is None or end_seconds is None or end_seconds <= start_seconds:
        raise ValueError(f"Clip {selected_id} needs valid start/end seconds before CV prep")

    video_path = Path(str(row.get("video_path") or ""))
    if not video_path.is_absolute():
        video_path = (REPO_ROOT / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Missing local video for {selected_id}: {video_path}")

    runner_name = str(row.get("runner_name") or "Unknown Runner")
    return ReviewedClip(
        candidate_id=selected_id,
        runner_name=runner_name,
        runner_slug=slugify(str(row.get("runner_slug") or runner_name)),
        title=str(row.get("title") or "Untitled clip"),
        source_url=str(row.get("url") or ""),
        channel=str(row.get("channel") or ""),
        video_path=video_path,
        quality=str(annotation.get("quality") or ""),
        camera_angle=str(annotation.get("camera_angle") or "unknown"),
        start_seconds=round(start_seconds, 2),
        end_seconds=round(end_seconds, 2),
        notes=str(annotation.get("notes") or ""),
        primary_bucket=str(row.get("primary_bucket") or "running"),
    )


def trim_reviewed_segment(clip: ReviewedClip, output_path: Path, *, force: bool = False) -> None:
    if output_path.exists() and not force:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(clip.video_path),
        "-ss",
        f"{clip.start_seconds:.2f}",
        "-to",
        f"{clip.end_seconds:.2f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_prompt_frame(segment_path: Path, output_path: Path, *, force: bool = False) -> dict[str, Any]:
    if output_path.exists() and not force:
        return inspect_video_frame(output_path)

    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open segment: {segment_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    midpoint_frame = max(0, frame_count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read prompt frame from {segment_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise ValueError(f"Could not write prompt frame: {output_path}")

    height, width = frame.shape[:2]
    return {
        "frame_index": midpoint_frame,
        "time_seconds": round(midpoint_frame / fps, 3) if fps else None,
        "width": width,
        "height": height,
    }


def inspect_video_frame(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return {"frame_index": None, "time_seconds": None, "width": width, "height": height}


def build_prompt_payload(clip: ReviewedClip, prompt_frame_path: Path, frame_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "candidate_id": clip.candidate_id,
        "runner_name": clip.runner_name,
        "prompt_frame": str(prompt_frame_path),
        "frame": frame_meta,
        "selection": {
            "type": "unset",
            "positive_points": [],
            "negative_points": [],
            "box": None,
            "mask_path": None,
        },
        "instructions": (
            "Select the target runner identity with one torso/hip positive point first. "
            "Use a loose box and negative points only when nearby runners overlap."
        ),
        "updated_at": utc_now_iso(),
    }


def build_track_seed_payload(clip: ReviewedClip, prompt_path: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "candidate_id": clip.candidate_id,
        "runner_name": clip.runner_name,
        "status": "pending_detector_tracker",
        "target_track_id": None,
        "prompt_path": str(prompt_path),
        "detector": {
            "preferred": "YOLO11/YOLO26 person detector or segmenter",
            "default_model": "yolo11n.pt",
            "class_filter": ["person"],
        },
        "tracker": {
            "preferred": "BoT-SORT",
            "backend": "boxmot_botsort",
            "ab_test": ["boxmot_deepocsort", "boxmot_bytetrack"],
            "track_buffer": 45,
            "high_thresh": 0.5,
            "low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "match_thresh": 0.8,
        },
        "reid": {
            "preferred": "Torchreid OSNet-AIN x1.0",
            "default_weights": "osnet_x0_25_msmt17.pt",
            "cosine_accept": 0.65,
            "cosine_recover": 0.58,
        },
        "identity_risk_flags": [
            "target similarity below recovery threshold",
            "area jump above rolling median tolerance",
            "centroid velocity spike after overlap",
            "near-tied tracker candidates",
        ],
        "updated_at": utc_now_iso(),
    }


def build_view_bucket_payload(clip: ReviewedClip) -> dict[str, Any]:
    return {
        "version": 1,
        "candidate_id": clip.candidate_id,
        "runner_name": clip.runner_name,
        "view_bucket": clip.camera_angle or "unknown",
        "source": "human_review",
        "multi_view_mode": "single_view_or_unsynchronized",
        "notes": (
            "Use geometry only for synchronized overlapping cameras. "
            "For unrelated clips, treat angle matching as retrieval metadata."
        ),
        "updated_at": utc_now_iso(),
    }


def build_cv_run_manifest(clip: ReviewedClip, run_dir: Path) -> dict[str, Any]:
    prompt_path = run_dir / "person_prompt.json"
    paths = {
        "source_segment": str(run_dir / "source_segment.mp4"),
        "prompt_frame": str(run_dir / "prompt_frame.jpg"),
        "person_prompt": str(prompt_path),
        "target_prompt": str(prompt_path),
        "track_seed": str(run_dir / "track_seed.json"),
        "view_bucket": str(run_dir / "view_bucket.json"),
        "tracklets": str(run_dir / "tracklets.parquet"),
        "tracklets_jsonl": str(run_dir / "tracklets.jsonl"),
        "reid": str(run_dir / "reid.parquet"),
        "reid_jsonl": str(run_dir / "reid.jsonl"),
        "masks_jsonl": str(run_dir / "masks.jsonl"),
        "mask_logits": str(run_dir / "mask_logits.zarr"),
        "poses": str(run_dir / "poses.parquet"),
        "pose_landmarks": str(run_dir / "pose_landmarks.jsonl"),
        "runner_mask": str(run_dir / "runner_mask.mp4"),
        "densepose": str(run_dir / "densepose.jsonl"),
        "densepose_parquet": str(run_dir / "densepose.parquet"),
        "fused_form": str(run_dir / "fused_form.jsonl"),
        "fused_form_parquet": str(run_dir / "fused_form.parquet"),
        "skeleton_render": str(run_dir / "skeleton_render.mp4"),
        "masked_runner": str(run_dir / "masked_runner.mp4"),
        "qa_overlay": str(run_dir / "qa_overlay.mp4"),
        "fused_overlay": str(run_dir / "fused_overlay.mp4"),
        "qc_metrics": str(run_dir / "qc_metrics.json"),
        "features": str(run_dir / "features.json"),
        "form_features": str(run_dir / "form_features.json"),
        "form_feature_arrays": str(run_dir / "form_features.npz"),
        "mmpose_landmarks": str(run_dir / "mmpose_landmarks.jsonl"),
        "openpose_landmarks": str(run_dir / "openpose_landmarks.jsonl"),
        "openpose_skeleton_render": str(run_dir / "openpose_skeleton_render.mp4"),
        "openpose_qa_overlay": str(run_dir / "openpose_qa_overlay.mp4"),
        "pose_comparison": str(run_dir / "pose_comparison.json"),
    }
    return {
        "version": 1,
        "created_at": utc_now_iso(),
        "candidate_id": clip.candidate_id,
        "runner_name": clip.runner_name,
        "runner_slug": clip.runner_slug,
        "implementation_goal": "identity_stable_runner_clip",
        "source": {
            "platform": "youtube" if "youtube.com" in clip.source_url else "local",
            "url": clip.source_url,
            "title": clip.title,
            "channel": clip.channel,
            "video_path": str(clip.video_path),
        },
        "review": {
            "quality": clip.quality,
            "camera_angle": clip.camera_angle,
            "primary_bucket": clip.primary_bucket,
            "start_seconds": clip.start_seconds,
            "end_seconds": clip.end_seconds,
            "duration_seconds": clip.duration_seconds,
            "notes": clip.notes,
        },
        "paths": paths,
        "stages": {
            "trim": {"status": "complete", "output": paths["source_segment"]},
            "clip_curation": {
                "status": "manual_interval_selected",
                "recommended_tool": "run_clip_curation.py ranked windows before review",
            },
            "person_prompt": {
                "status": "needs_selection",
                "recommended_tool": "single positive torso/hip click plus optional box/negative points",
                "output": paths["person_prompt"],
            },
            "detector_tracker": {
                "status": "pending_prompt",
                "recommended_tool": "YOLO11/YOLO26 person detections + BoT-SORT + OSNet ReID",
                "track_seed": paths["track_seed"],
                "tracklets": paths["tracklets"],
                "tracklets_jsonl": paths["tracklets_jsonl"],
                "reid": paths["reid"],
                "reid_jsonl": paths["reid_jsonl"],
            },
            "whole_runner_mask": {
                "status": "pending_prompt",
                "recommended_tool": "SAM 3.1 MLX now; Cutie planned after target track is known",
                "output": paths["runner_mask"],
                "metadata": str(run_dir / "runner_mask_metadata.jsonl"),
                "masks_jsonl": paths["masks_jsonl"],
                "gated_by": "detector_tracker",
            },
            "pose": {
                "status": "pending",
                "recommended_tool": "RTMPose/RTMW via RTMLib preferred; OpenPose and MediaPipe as baselines",
                "output": paths["pose_landmarks"],
                "parquet": paths["poses"],
            },
            "densepose": {
                "status": "pending_runner_mask",
                "recommended_tool": "Detectron2 projects/DensePose",
                "output": paths["densepose"],
                "parquet": paths["densepose_parquet"],
            },
            "fused_form": {
                "status": "pending_pose_and_densepose",
                "recommended_tool": "identity-gated mask + pose confidence + optional DensePose fusion",
                "output": paths["fused_form"],
                "parquet": paths["fused_form_parquet"],
                "overlay": paths["fused_overlay"],
            },
            "renders": {
                "status": "pending",
                "outputs": [
                    paths["skeleton_render"],
                    paths["masked_runner"],
                    paths["qa_overlay"],
                    paths["fused_overlay"],
                ],
            },
            "features": {"status": "pending", "output": paths["features"]},
            "qc_metrics": {
                "status": "pending",
                "recommended_tool": "TISR, occlusion recovery latency, mask churn, pose visibility",
                "output": paths["qc_metrics"],
            },
            "form_features": {
                "status": "pending_fused_form",
                "recommended_tool": "Pose sequence + fused confidence feature compiler",
                "output": paths["form_features"],
                "arrays": paths["form_feature_arrays"],
            },
            "artifact_tables": {
                "status": "pending",
                "recommended_tool": "JSONL to Parquet artifact exporter",
                "outputs": [paths["poses"], paths["densepose_parquet"], paths["fused_form_parquet"]],
            },
            "openpose": {
                "status": "pending_optional",
                "recommended_tool": "OpenPose BODY_25 optional benchmark",
                "output": paths["openpose_landmarks"],
                "comparison": paths["pose_comparison"],
            },
        },
        "occlusion_policy": {
            "drop_frame_when": [
                "target mask missing",
                "target tracker missing or identity-risk flagged",
                "pose_confidence_mean below threshold",
                "visible key landmarks absent for more than a short gap",
                "mask area jumps enough to imply target switch",
            ],
            "short_gap_strategy": "interpolate pose only for short low-risk gaps; never interpolate through identity risk",
            "long_gap_strategy": "split into subsegments, recover with ReID, or request another prompt",
            "matching_strategy": "weight similarity by per-frame confidence and ignore dropped frames",
        },
    }


def prepare_single_clip_cv_run(
    *,
    candidate_id: str | None = None,
    quality: str = "good",
    manifest_path: Path = DEFAULT_REVIEW_MANIFEST,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    output_root: Path = DEFAULT_CV_RUN_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    clip = load_reviewed_clip(
        candidate_id=candidate_id,
        quality=quality,
        manifest_path=manifest_path,
        annotations_path=annotations_path,
    )
    run_dir = output_root / clip.candidate_id
    segment_path = run_dir / "source_segment.mp4"
    prompt_frame_path = run_dir / "prompt_frame.jpg"
    prompt_path = run_dir / "person_prompt.json"
    manifest_output_path = run_dir / "cv_run_manifest.json"

    trim_reviewed_segment(clip, segment_path, force=force)
    frame_meta = extract_prompt_frame(segment_path, prompt_frame_path, force=force)
    if force or not prompt_path.exists():
        write_json(prompt_path, build_prompt_payload(clip, prompt_frame_path, frame_meta))
    track_seed_path = run_dir / "track_seed.json"
    if force or not track_seed_path.exists():
        write_json(track_seed_path, build_track_seed_payload(clip, prompt_path))
    view_bucket_path = run_dir / "view_bucket.json"
    if force or not view_bucket_path.exists():
        write_json(view_bucket_path, build_view_bucket_payload(clip))
    manifest = build_cv_run_manifest(clip, run_dir)
    write_json(manifest_output_path, manifest)
    return manifest
