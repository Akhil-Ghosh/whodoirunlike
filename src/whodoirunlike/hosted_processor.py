from __future__ import annotations

import json
import importlib.util
import mimetypes
import os
import platform
import re
import secrets
import shutil
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from whodoirunlike.cv_flow import utc_now_iso, write_json
from whodoirunlike.full_pipeline import run_full_cv_pipeline
from whodoirunlike.identity_runner import DEFAULT_IDENTITY_BACKEND, identity_setup_status
from whodoirunlike.sam2_runner import inspect_video
from whodoirunlike.sam31_gpu_runner import DEFAULT_SAM31_GPU_MODEL
from whodoirunlike.sam31_mlx_runner import DEFAULT_SAM31_MLX_MODEL


DEFAULT_HOSTED_RUN_ROOT = Path(os.getenv("WHODOIRUNLIKE_HOSTED_RUN_ROOT", "artifacts/hosted_runs"))
REPO_ROOT = Path(__file__).resolve().parents[2]
DENSEPOSE_CONFIG_ENV = "DENSEPOSE_CONFIG"
DENSEPOSE_WEIGHTS_ENV = "DENSEPOSE_WEIGHTS"
DENSEPOSE_DEVICE_ENV = "DENSEPOSE_DEVICE"
DENSEPOSE_DEFAULT_CONFIG = (
    REPO_ROOT / "models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
)
DENSEPOSE_DEFAULT_WEIGHTS = (
    REPO_ROOT / "models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl"
)
RUN_ID_PATTERN = re.compile(r"^[a-f0-9-]{32,36}$", re.IGNORECASE)
PROCESSOR_USER_AGENT = "Mozilla/5.0 (compatible; whodoirunlike-processor/1.0)"
DEFAULT_HOSTED_MASK_BACKEND = "sam31_gpu"

router = APIRouter()


class WorkerJobSource(BaseModel):
    url: str
    key: str
    filename: str | None = None
    content_type: str
    size_bytes: int


class WorkerJobRequest(BaseModel):
    run_id: str
    source: WorkerJobSource
    callback_base_url: str


def _hosted_run_root() -> Path:
    root = DEFAULT_HOSTED_RUN_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _processor_secret() -> str:
    return os.getenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "")


def _require_processor_auth(request: Request) -> None:
    expected = _processor_secret()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Set WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET before accepting hosted jobs.",
        )

    auth = request.headers.get("authorization") or ""
    prefix = "Bearer "
    if not auth.startswith(prefix):
        raise HTTPException(status_code=401, detail="Processor authorization required.")

    supplied = auth[len(prefix) :]
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Processor authorization required.")


def _validate_job_payload(payload: WorkerJobRequest) -> None:
    if not RUN_ID_PATTERN.fullmatch(payload.run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    callback = payload.callback_base_url.rstrip("/")
    if not payload.source.url.startswith(f"{callback}/"):
        raise HTTPException(status_code=400, detail="Source URL must belong to callback_base_url.")


def _request_headers() -> dict[str, str]:
    token = _processor_secret()
    if not token:
        raise RuntimeError("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET is not configured")
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": PROCESSOR_USER_AGENT,
    }


def _download_source(payload: WorkerJobRequest, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(payload.source.url, headers=_request_headers(), method="GET")
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def _post_worker_report(
    *,
    callback_base_url: str,
    run_id: str,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{callback_base_url.rstrip('/')}/v1/jobs/{run_id}/report",
        data=body,
        method="POST",
        headers={
            **_request_headers(),
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(request, timeout=30):
        return


def _put_worker_artifact(
    *,
    callback_base_url: str,
    run_id: str,
    name: str,
    path: Path,
) -> None:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as artifact:
        request = urllib.request.Request(
            f"{callback_base_url.rstrip('/')}/v1/jobs/{run_id}/artifacts/{name}",
            data=artifact,
            method="PUT",
            headers={
                **_request_headers(),
                "Content-Type": content_type,
                "Content-Length": str(path.stat().st_size),
            },
        )
        with urllib.request.urlopen(request, timeout=120):
            return


def _write_prompt_frame(source_path: Path, prompt_frame_path: Path) -> dict[str, Any]:
    prompt_frame_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source upload: {source_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read first frame from source upload: {source_path}")
    if not cv2.imwrite(str(prompt_frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"Could not write prompt frame: {prompt_frame_path}")
    return {
        "frame_index": 0,
        "image_path": str(prompt_frame_path),
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
    }


def _default_target_prompt(prompt_frame_path: Path, frame_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "source": "hosted_upload_auto_center_v1",
        "selection": {
            "type": "auto_center_runner",
            "positive_points": [
                {
                    "x": 0.5,
                    "y": 0.52,
                    "label": "target_runner_center",
                }
            ],
            "negative_points": [],
            "box": {
                "x": 0.32,
                "y": 0.14,
                "width": 0.36,
                "height": 0.74,
            },
        },
        "frame": {
            **frame_meta,
            "image_path": str(prompt_frame_path),
        },
        "notes": "Auto-seeded for public uploads. Works best when the target runner is centered in frame.",
    }


def _manifest_paths(run_dir: Path) -> dict[str, str]:
    return {
        "source_segment": str(run_dir / "source_segment.mp4"),
        "prompt_frame": str(run_dir / "prompt_frame.jpg"),
        "person_prompt": str(run_dir / "person_prompt.json"),
        "target_prompt": str(run_dir / "person_prompt.json"),
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


def _write_hosted_manifest(
    *,
    run_dir: Path,
    payload: WorkerJobRequest,
    source_path: Path,
    video_meta: dict[str, Any],
) -> Path:
    prompt_frame_path = run_dir / "prompt_frame.jpg"
    frame_meta = _write_prompt_frame(source_path, prompt_frame_path)
    prompt_path = run_dir / "person_prompt.json"
    prompt = _default_target_prompt(prompt_frame_path, frame_meta)
    write_json(prompt_path, prompt)

    paths = _manifest_paths(run_dir)
    fps = float(video_meta.get("fps") or 0.0)
    frame_count = int(video_meta.get("frame_count") or 0)
    duration_seconds = round(frame_count / fps, 3) if fps else None

    write_json(
        run_dir / "track_seed.json",
        {
            "candidate_id": payload.run_id,
            "runner_name": "Uploaded runner",
            "prompt_path": str(prompt_path),
            "target_lock_method": "auto_center_prompt",
            "updated_at": utc_now_iso(),
        },
    )
    write_json(
        run_dir / "view_bucket.json",
        {
            "candidate_id": payload.run_id,
            "runner_name": "Uploaded runner",
            "view_bucket": "unknown",
            "source": "hosted_upload",
            "updated_at": utc_now_iso(),
        },
    )

    manifest = {
        "version": 1,
        "created_at": utc_now_iso(),
        "candidate_id": payload.run_id,
        "runner_name": "Uploaded runner",
        "runner_slug": "uploaded-runner",
        "implementation_goal": "identity_stable_runner_clip",
        "source": {
            "platform": "hosted_upload",
            "worker_source_url": payload.source.url,
            "worker_object_key": payload.source.key,
            "filename": payload.source.filename,
            "content_type": payload.source.content_type,
            "size_bytes": payload.source.size_bytes,
            "video_path": str(source_path),
        },
        "review": {
            "quality": "hosted_upload",
            "camera_angle": "unknown",
            "primary_bucket": "running",
            "duration_seconds": duration_seconds,
            "notes": "Auto-seeded public upload. Review the target prompt before adding this clip to a reference set.",
        },
        "video": video_meta,
        "paths": paths,
        "stages": {
            "upload": {"status": "complete", "output": str(source_path)},
            "person_prompt": {"status": "auto_seeded", "output": str(prompt_path)},
            "detector_tracker": {"status": "pending"},
            "whole_runner_mask": {"status": "pending"},
            "pose": {"status": "pending"},
            "densepose": {"status": "pending"},
            "fused_form": {"status": "pending"},
            "features": {"status": "pending"},
            "qc_metrics": {"status": "pending"},
        },
    }
    manifest_path = run_dir / "cv_run_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _artifact_files(run_dir: Path) -> list[Path]:
    candidates = [
        "cv_run_manifest.json",
        "person_prompt.json",
        "track_seed.json",
        "view_bucket.json",
        "tracklets.jsonl",
        "tracklets.parquet",
        "reid.jsonl",
        "reid.parquet",
        "runner_mask.mp4",
        "masked_runner.mp4",
        "qa_overlay.mp4",
        "skeleton_render.mp4",
        "fused_overlay.mp4",
        "pose_landmarks.jsonl",
        "mmpose_landmarks.jsonl",
        "openpose_landmarks.jsonl",
        "densepose.jsonl",
        "densepose.parquet",
        "fused_form.jsonl",
        "fused_form.parquet",
        "features.json",
        "form_features.json",
        "form_features.npz",
        "qc_metrics.json",
        "hosted_pipeline_result.json",
    ]
    return [run_dir / name for name in candidates if (run_dir / name).is_file()]


def _upload_artifacts(payload: WorkerJobRequest, run_dir: Path) -> list[str]:
    uploaded: list[str] = []
    for path in _artifact_files(run_dir):
        _put_worker_artifact(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            name=path.name,
            path=path,
        )
        uploaded.append(path.name)
    return uploaded


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_repo_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (REPO_ROOT / raw_path).resolve()


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _dependency_status(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _secret_status() -> dict[str, Any]:
    return {
        "ready": bool(_processor_secret()),
        "reasons": [] if _processor_secret() else ["Set WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET."],
        "env": "WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET",
    }


def _mask_backend() -> str:
    return os.getenv("WHODOIRUNLIKE_MASK_BACKEND", DEFAULT_HOSTED_MASK_BACKEND).strip() or DEFAULT_HOSTED_MASK_BACKEND


def sam31_gpu_setup_status() -> dict[str, Any]:
    dependencies = _dependency_status(["torch", "sam3", "PIL", "cv2", "numpy"])
    missing = [name for name, available in dependencies.items() if not available]
    reasons: list[str] = []
    if missing:
        reasons.append("Install SAM 3.1 GPU dependencies in this image: " + ", ".join(missing))

    torch_version = None
    cuda_version = None
    cuda_available = False
    if dependencies["torch"]:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())

    checkpoint_path = os.getenv("WHODOIRUNLIKE_SAM31_GPU_CHECKPOINT", "").strip()
    has_hf_token = bool(
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
        or checkpoint_path
    )
    if not has_hf_token:
        reasons.append(
            "Set HF_TOKEN with access to facebook/sam3.1, or set "
            "WHODOIRUNLIKE_SAM31_GPU_CHECKPOINT to a local checkpoint path."
        )
    if not cuda_available:
        reasons.append("SAM 3.1 GPU requires a CUDA GPU; torch.cuda.is_available() is false.")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "backend": "sam31_gpu",
        "model": DEFAULT_SAM31_GPU_MODEL,
        "checkpoint_path": checkpoint_path or None,
        "dependencies": dependencies,
        "torch": {
            "version": torch_version,
            "cuda_version": cuda_version,
            "cuda_available": cuda_available,
        },
        "env": {
            "backend": "WHODOIRUNLIKE_MASK_BACKEND",
            "checkpoint": "WHODOIRUNLIKE_SAM31_GPU_CHECKPOINT",
            "hf_token": "HF_TOKEN",
        },
        "install_command": (
            "Install PyTorch CUDA 12.6+ and the official facebookresearch/sam3 package."
        ),
    }


def sam31_mlx_setup_status() -> dict[str, Any]:
    dependencies = _dependency_status(["mlx", "mlx_vlm", "PIL", "cv2", "numpy"])
    missing = [name for name, available in dependencies.items() if not available]
    reasons: list[str] = []
    if missing:
        reasons.append("Install SAM 3.1 MLX dependencies in this venv: " + ", ".join(missing))

    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine not in {"arm64", "aarch64"}:
        reasons.append("sam31_mlx is intended for Apple Silicon. Use a Mac processor or switch mask backend.")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "backend": "sam31_mlx",
        "model": os.getenv("WHODOIRUNLIKE_SAM31_MLX_MODEL", DEFAULT_SAM31_MLX_MODEL),
        "quality_mode": os.getenv("WHODOIRUNLIKE_MASK_QUALITY_MODE", "native"),
        "platform": {"system": system, "machine": machine},
        "dependencies": dependencies,
        "install_command": 'python -m pip install -e ".[sam31]"',
    }


def mask_setup_status(mask_backend: str) -> dict[str, Any]:
    normalized = mask_backend.strip().lower()
    if normalized in {"sam31_gpu", "sam3.1_gpu", "sam31_cuda", "sam3.1_cuda"}:
        return sam31_gpu_setup_status()
    if normalized in {"sam31", "sam31_mlx", "sam3.1_mlx", "mlx"}:
        return sam31_mlx_setup_status()
    return {
        "ready": False,
        "reasons": ["WHODOIRUNLIKE_MASK_BACKEND must be sam31_gpu for RunPod or sam31_mlx locally."],
        "backend": mask_backend,
    }


def densepose_setup_status() -> dict[str, Any]:
    config_value = os.getenv(DENSEPOSE_CONFIG_ENV, "").strip()
    weights_value = os.getenv(DENSEPOSE_WEIGHTS_ENV, "").strip()
    device = os.getenv(DENSEPOSE_DEVICE_ENV, "cpu").strip() or "cpu"
    reasons: list[str] = []

    config_path: Path | None = None
    if config_value:
        config_path = _resolve_repo_path(config_value)
        if not config_path.exists():
            reasons.append(f"{DENSEPOSE_CONFIG_ENV} does not exist: {config_path}")
    elif DENSEPOSE_DEFAULT_CONFIG.exists():
        config_path = DENSEPOSE_DEFAULT_CONFIG
    else:
        reasons.append(
            f"Set {DENSEPOSE_CONFIG_ENV} or download the default config to {DENSEPOSE_DEFAULT_CONFIG}"
        )

    weights_for_runner: str | None = None
    if weights_value:
        if _is_url(weights_value):
            weights_for_runner = weights_value
        else:
            weights_path = _resolve_repo_path(weights_value)
            weights_for_runner = str(weights_path)
            if not weights_path.exists():
                reasons.append(f"{DENSEPOSE_WEIGHTS_ENV} does not exist: {weights_path}")
    elif DENSEPOSE_DEFAULT_WEIGHTS.exists():
        weights_for_runner = str(DENSEPOSE_DEFAULT_WEIGHTS)
    else:
        reasons.append(
            f"Set {DENSEPOSE_WEIGHTS_ENV} or download default weights to {DENSEPOSE_DEFAULT_WEIGHTS}"
        )

    dependencies = _dependency_status(["detectron2", "densepose"])
    missing_dependencies = [name for name, available in dependencies.items() if not available]
    if missing_dependencies:
        reasons.append(
            "Install optional DensePose dependencies in this venv: "
            + ", ".join(missing_dependencies)
        )

    return {
        "ready": not reasons,
        "reasons": reasons,
        "config_path": str(config_path) if config_path else None,
        "weights": weights_for_runner,
        "device": device,
        "using_defaults": {
            "config": not bool(config_value) and config_path == DENSEPOSE_DEFAULT_CONFIG,
            "weights": not bool(weights_value) and weights_for_runner == str(DENSEPOSE_DEFAULT_WEIGHTS),
        },
        "env": {
            "config": DENSEPOSE_CONFIG_ENV,
            "weights": DENSEPOSE_WEIGHTS_ENV,
            "device": DENSEPOSE_DEVICE_ENV,
        },
        "dependencies": dependencies,
        "install_command": (
            "Install Detectron2 for this Python/PyTorch platform, then expose "
            "Detectron2 projects/DensePose."
        ),
    }


def pose_setup_status(pose_backend: str) -> dict[str, Any]:
    if pose_backend.startswith("mmpose_"):
        from whodoirunlike.mmpose_runner import mmpose_setup_status

        return mmpose_setup_status(pose_backend)
    if pose_backend == "mediapipe":
        dependencies = _dependency_status(["mediapipe", "cv2", "numpy"])
        missing = [name for name, available in dependencies.items() if not available]
        return {
            "ready": not missing,
            "reasons": [] if not missing else ["Install MediaPipe dependencies: " + ", ".join(missing)],
            "backend": "mediapipe",
            "dependencies": dependencies,
            "install_command": "python -m pip install -e .",
        }

    from whodoirunlike.openpose_runner import openpose_setup_status

    return openpose_setup_status()


def _readiness_check(label: str, callback: Any) -> dict[str, Any]:
    try:
        return callback()
    except Exception as error:
        return {
            "ready": False,
            "reasons": [
                f"{label} readiness check failed: {type(error).__name__}: {str(error)[:500]}"
            ],
            "error_type": type(error).__name__,
        }


def processor_readiness() -> dict[str, Any]:
    identity_backend = os.getenv("WHODOIRUNLIKE_IDENTITY_BACKEND", DEFAULT_IDENTITY_BACKEND)
    pose_backend = os.getenv("WHODOIRUNLIKE_POSE_BACKEND", "mmpose_rtmpose_l_384")
    mask_backend = _mask_backend()
    skip_densepose = _env_bool("WHODOIRUNLIKE_SKIP_DENSEPOSE")
    checks = {
        "processor_secret": _readiness_check("processor_secret", _secret_status),
        "identity": _readiness_check(
            "identity",
            lambda: identity_setup_status(identity_backend),
        ),
        "mask": _readiness_check("mask", lambda: mask_setup_status(mask_backend)),
        "pose": _readiness_check("pose", lambda: pose_setup_status(pose_backend)),
        "densepose": (
            {"ready": True, "skipped": True, "reasons": ["WHODOIRUNLIKE_SKIP_DENSEPOSE=true"]}
            if skip_densepose
            else _readiness_check("densepose", densepose_setup_status)
        ),
    }
    return {
        "ready_for_full_pipeline": all(bool(check.get("ready")) for check in checks.values()),
        "identity_backend": identity_backend,
        "pose_backend": pose_backend,
        "mask_backend": mask_backend,
        "skip_densepose": skip_densepose,
        "checks": checks,
    }


def process_hosted_job(payload: WorkerJobRequest, *, raise_on_error: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    run_dir = _hosted_run_root() / payload.run_id
    source_path = run_dir / "source_segment.mp4"
    try:
        _post_worker_report(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={"status": "running", "progress": {"phase": "downloading_upload"}},
        )
        _download_source(payload, source_path)
        video_meta = inspect_video(source_path)
        _write_hosted_manifest(
            run_dir=run_dir,
            payload=payload,
            source_path=source_path,
            video_meta=video_meta,
        )

        _post_worker_report(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={"status": "running", "progress": {"phase": "running_full_cv_pipeline"}},
        )
        result = run_full_cv_pipeline(
            run_dir=run_dir,
            identity_backend=os.getenv("WHODOIRUNLIKE_IDENTITY_BACKEND", DEFAULT_IDENTITY_BACKEND),
            pose_backend=os.getenv("WHODOIRUNLIKE_POSE_BACKEND", "mmpose_rtmpose_l_384"),
            mask_backend=_mask_backend(),
            mask_quality_mode=os.getenv("WHODOIRUNLIKE_MASK_QUALITY_MODE", "native"),
            skip_densepose=_env_bool("WHODOIRUNLIKE_SKIP_DENSEPOSE"),
        )
        write_json(run_dir / "hosted_pipeline_result.json", result)

        _post_worker_report(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={"status": "running", "progress": {"phase": "uploading_artifacts"}},
        )
        uploaded = _upload_artifacts(payload, run_dir)
        _post_worker_report(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={
                "status": "complete",
                "progress": {
                    "phase": "complete",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                "summary": {
                    "steps": result.get("steps", []),
                    "artifacts_uploaded": uploaded,
                    "run_dir": str(run_dir),
                },
            },
        )
        return {
            "status": "complete",
            "run_id": payload.run_id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "artifacts_uploaded": uploaded,
        }
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        error_traceback = traceback.format_exc(limit=8)
        write_json(
            run_dir / "hosted_job_error.json",
            {
                "run_id": payload.run_id,
                "error": str(exc),
                "traceback": error_traceback,
                "failed_at": utc_now_iso(),
            },
        )
        try:
            _post_worker_report(
                callback_base_url=payload.callback_base_url,
                run_id=payload.run_id,
                payload={
                    "status": "failed",
                    "progress": {"phase": "failed"},
                    "error": f"{exc}\n\n{error_traceback[-2000:]}",
                },
            )
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            pass
        if raise_on_error:
            raise
        return {
            "status": "failed",
            "run_id": payload.run_id,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


@router.get("/v1/processor/health")
def processor_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "whodoirunlike-processor",
        "hosted_run_root": str(_hosted_run_root()),
        "has_processor_secret": bool(_processor_secret()),
        "readiness": processor_readiness(),
    }


@router.post("/v1/processor/jobs", status_code=202)
async def start_processor_job(
    payload: WorkerJobRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _require_processor_auth(request)
    _validate_job_payload(payload)
    background_tasks.add_task(process_hosted_job, payload)
    return {
        "run_id": payload.run_id,
        "status": "accepted",
        "message": "Hosted CV pipeline job accepted.",
    }
