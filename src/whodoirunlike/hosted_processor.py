from __future__ import annotations

import hashlib
import json
import importlib.util
import logging
import mimetypes
import os
import platform
import re
import secrets
import shutil
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from whodoirunlike.cv_flow import utc_now_iso, write_json
from whodoirunlike.full_pipeline import (
    DEFAULT_INLINE_SEGMENTATION_MODEL,
    INLINE_MASK_BACKENDS,
    run_full_cv_pipeline,
)
from whodoirunlike.identity_runner import (
    BOXMOT_BACKENDS,
    DEFAULT_IDENTITY_BACKEND,
    identity_setup_status,
)
from whodoirunlike.processing_telemetry import (
    ProcessingTelemetry,
    create_hosted_telemetry,
    ensure_attempt_id,
    input_metadata_from_video,
)
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam2_runner import inspect_video
from whodoirunlike.sam31_gpu_runner import DEFAULT_SAM31_GPU_MODEL
from whodoirunlike.sam31_loader_config import sam31_exact_cv2_loader_settings
from whodoirunlike.sam31_mlx_runner import DEFAULT_SAM31_MLX_MODEL
from whodoirunlike.video_io import make_browser_playable_mp4


LOGGER = logging.getLogger(__name__)
DEFAULT_HOSTED_RUN_ROOT = Path(os.getenv("WHODOIRUNLIKE_HOSTED_RUN_ROOT", "artifacts/hosted_runs"))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSET_ROOT = REPO_ROOT / "site/public/assets/demos"
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
SPEED_PROFILE_ENV = "WHODOIRUNLIKE_SPEED_PROFILE"
MAX_SAFE_SPEED_PROFILE = "max_safe"
# Job reports can require multiple conditional R2 writes. These calls are
# best-effort but synchronous, so a Worker outage can add up to 10 seconds at a
# report boundary; the larger deadline prevents partial status mutations under
# normal remote-write latency.
DEFAULT_REPORT_TIMEOUT_SECONDS = 10.0
# Each callback still performs several durable Worker writes, but telemetry is
# delivered by a bounded concurrent pool so normal processing no longer leaves
# a large serial backlog. Retain the larger default as a correctness backstop;
# production templates may use a shorter explicit deadline after canarying the
# configured sender concurrency.
DEFAULT_TELEMETRY_DRAIN_TIMEOUT_SECONDS = 180.0
DEFAULT_TELEMETRY_SNAPSHOT_TIMEOUT_SECONDS = 0.25
DEFAULT_CALLBACK_ORIGINS = (
    "https://api.whodoirunlike.com",
    "https://staging-api.whodoirunlike.com",
)
DEFAULT_ARTIFACT_PUBLISH_WORKERS = 4
MAX_ARTIFACT_PUBLISH_WORKERS = 8
DEFAULT_ARTIFACT_PUBLISH_ATTEMPTS = 3
MAX_ARTIFACT_PUBLISH_ATTEMPTS = 5
DEFAULT_ARTIFACT_PUBLISH_BACKOFF_SECONDS = 0.1
MAX_ARTIFACT_PUBLISH_BACKOFF_SECONDS = 2.0
MAX_ARTIFACT_FINALIZE_BODY_BYTES = 64 * 1024
MAX_ARTIFACT_FINALIZE_COUNT = 64
MAX_R2_OBJECT_BYTES = 5 * 1024 * 1024 * 1024
ARTIFACT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
R2_OBJECT_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
ARTIFACT_CONTENT_TYPE_PATTERN = re.compile(
    r"^[a-zA-Z0-9!#$&^_.+-]+/[a-zA-Z0-9!#$&^_.+-]+(?:\s*;[^\r\n]*)?$"
)


def _close_telemetry_delivery(
    telemetry: ProcessingTelemetry,
    *,
    run_id: str,
    attempt_id: str,
) -> bool:
    timeout = _env_float(
        "WHODOIRUNLIKE_TELEMETRY_DRAIN_TIMEOUT_SECONDS",
        DEFAULT_TELEMETRY_DRAIN_TIMEOUT_SECONDS,
    )
    delivered = telemetry.close(timeout=timeout)
    if delivered:
        return True

    # Only validated identifiers, the configured deadline, and integer counters
    # are logged. Event bodies, callback URLs, headers, and secrets stay out of
    # the diagnostic record.
    diagnostic = {
        "event": "processing_telemetry_drain_exhausted",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "timeout_seconds": timeout,
        **telemetry.delivery_measurements(),
    }
    LOGGER.error(
        "Processing telemetry drain exhausted: %s",
        json.dumps(diagnostic, separators=(",", ":"), sort_keys=True),
    )
    return False


DEVELOPMENT_CALLBACK_ORIGINS = (
    "http://127.0.0.1:8787",
    "http://localhost:8787",
)
COLE_DEMO_SOURCE_SHA256 = "a8146591119c5439cc01168df63fa6144a7a55ff6817726946e1e8f5bc381617"
COLE_DEMO_PROMPT_BOX = {
    "x": 0.624283,
    "y": 0.175162,
    "width": 0.182908,
    "height": 0.772011,
}
_HOSTED_PUBLISHABLE_ARTIFACTS = (
    ("person_prompt", "person_prompt.json"),
    ("track_seed", "track_seed.json"),
    ("view_bucket", "view_bucket.json"),
    ("tracklets_jsonl", "tracklets.jsonl"),
    ("tracklets", "tracklets.parquet"),
    ("reid_jsonl", "reid.jsonl"),
    ("reid", "reid.parquet"),
    ("runner_mask", "runner_mask.mp4"),
    ("masked_runner", "masked_runner.mp4"),
    ("qa_overlay", "qa_overlay.mp4"),
    ("skeleton_render", "skeleton_render.mp4"),
    ("fused_overlay", "fused_overlay.mp4"),
    ("pose_landmarks", "pose_landmarks.jsonl"),
    ("mmpose_landmarks", "mmpose_landmarks.jsonl"),
    ("openpose_landmarks", "openpose_landmarks.jsonl"),
    ("densepose", "densepose.jsonl"),
    ("densepose_parquet", "densepose.parquet"),
    ("fused_form", "fused_form.jsonl"),
    ("fused_form_parquet", "fused_form.parquet"),
    ("features", "features.json"),
    ("form_features", "form_features.json"),
    ("form_feature_arrays", "form_features.npz"),
    ("qc_metrics", "qc_metrics.json"),
    ("hosted_pipeline_result", "hosted_pipeline_result.json"),
)
_RESULT_READY_ARTIFACT_NAMES = ("fused_overlay.mp4",)
_DEFERRED_BROWSER_ARTIFACT_NAMES = frozenset(
    {"runner_mask.mp4", "masked_runner.mp4"}
)
_PROCESSOR_STARTED_AT = time.monotonic()
_PROCESSOR_INVOCATION_LOCK = threading.Lock()
_PROCESSOR_INVOCATION_COUNT = 0

router = APIRouter()


class WorkerJobSource(BaseModel):
    url: str
    key: str
    filename: str | None = None
    content_type: str
    size_bytes: int


class WorkerJobRequest(BaseModel):
    run_id: str
    attempt_id: str | None = None
    attempt_number: int | None = None
    attempt_started_at: str | None = None
    processor_enqueued_at: str | None = None
    telemetry_sequence_start: int | None = None
    runpod_job_id: str | None = None
    runpod_delay_time_ms: float | None = None
    source: WorkerJobSource
    callback_base_url: str
    target_prompt: dict[str, Any] | None = None


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
    callback_origin = _canonical_callback_origin(payload.callback_base_url)
    if callback_origin is None or callback_origin not in _allowed_callback_origins():
        raise HTTPException(status_code=400, detail="callback_base_url origin is not allowed.")
    source_origin = _canonical_callback_origin(payload.source.url, require_origin_only=False)
    source_url = urllib.parse.urlsplit(payload.source.url)
    expected_path = f"/v1/jobs/{payload.run_id}/source"
    if (
        source_origin != callback_origin
        or source_url.path != expected_path
        or bool(source_url.query)
        or bool(source_url.fragment)
    ):
        raise HTTPException(
            status_code=400,
            detail="Source URL must be the expected job source on callback_base_url.",
        )


def _is_explicit_development() -> bool:
    return os.getenv("WHODOIRUNLIKE_ENVIRONMENT", "").strip().lower() in {
        "dev",
        "development",
        "local",
        "test",
    }


def _canonical_callback_origin(
    value: str,
    *,
    require_origin_only: bool = True,
) -> str | None:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not hostname or parsed.username is not None or parsed.password is not None:
        return None
    if scheme != "https":
        is_localhost = hostname in {"localhost", "127.0.0.1", "::1"}
        if scheme != "http" or not is_localhost or not _is_explicit_development():
            return None
    if require_origin_only and (parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        return None
    default_port = 443 if scheme == "https" else 80
    formatted_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{scheme}://{formatted_host}{f':{port}' if port and port != default_port else ''}"


def _allowed_callback_origins() -> frozenset[str]:
    configured = os.getenv("WHODOIRUNLIKE_CALLBACK_ORIGINS", "").strip()
    candidates = (
        [part.strip() for part in configured.split(",") if part.strip()]
        if configured
        else [
            *DEFAULT_CALLBACK_ORIGINS,
            *(DEVELOPMENT_CALLBACK_ORIGINS if _is_explicit_development() else ()),
        ]
    )
    return frozenset(
        origin
        for candidate in candidates
        if (origin := _canonical_callback_origin(candidate)) is not None
    )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demo_upload_profile(source_path: Path) -> dict[str, Any] | None:
    source_sha256 = _file_sha256(source_path)
    if source_sha256 != COLE_DEMO_SOURCE_SHA256:
        return None

    return {
        "id": "cole_hocker_reference_v1",
        "source_sha256": source_sha256,
        "runner_name": "Cole Hocker",
        "runner_slug": "cole-hocker",
        "prompt_frame_index": 130,
        "prompt_box": COLE_DEMO_PROMPT_BOX,
        "reference_artifacts": {
            "fused_overlay.mp4": "cole-fused.mp4",
            "skeleton_render.mp4": "cole-skeleton.mp4",
            "masked_runner.mp4": "cole-isolation.mp4",
        },
    }


def _active_demo_profile(
    demo_profile: dict[str, Any] | None,
    uploaded_prompt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if uploaded_prompt is not None:
        return None
    return demo_profile


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
            **(
                {"X-Processing-Attempt-Id": str(payload["attempt_id"])}
                if payload.get("attempt_id")
                else {}
            ),
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=_env_float("WHODOIRUNLIKE_REPORT_TIMEOUT_SECONDS", DEFAULT_REPORT_TIMEOUT_SECONDS),
    ):
        return


def _post_worker_report_best_effort(
    *,
    callback_base_url: str,
    run_id: str,
    payload: dict[str, Any],
) -> bool:
    try:
        _post_worker_report(
            callback_base_url=callback_base_url,
            run_id=run_id,
            payload=payload,
        )
        return True
    except Exception:
        return False


def _put_worker_artifact(
    *,
    callback_base_url: str,
    run_id: str,
    attempt_id: str,
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
                "X-Processing-Attempt-Id": attempt_id,
                "Content-Type": content_type,
                "Content-Length": str(path.stat().st_size),
            },
        )
        with urllib.request.urlopen(request, timeout=120):
            return


def _artifact_content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _artifact_publish_retry_settings() -> tuple[int, float]:
    attempts = min(
        MAX_ARTIFACT_PUBLISH_ATTEMPTS,
        _env_int(
            "WHODOIRUNLIKE_ARTIFACT_PUBLISH_ATTEMPTS",
            DEFAULT_ARTIFACT_PUBLISH_ATTEMPTS,
            minimum=1,
        ),
    )
    backoff_seconds = min(
        MAX_ARTIFACT_PUBLISH_BACKOFF_SECONDS,
        _env_float(
            "WHODOIRUNLIKE_ARTIFACT_PUBLISH_BACKOFF_SECONDS",
            DEFAULT_ARTIFACT_PUBLISH_BACKOFF_SECONDS,
        ),
    )
    return attempts, backoff_seconds


def _retryable_artifact_publish_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429} or error.code >= 500
    return isinstance(
        error,
        (TimeoutError, ConnectionError, urllib.error.URLError, OSError),
    )


def _artifact_publish_retry_delay(backoff_seconds: float, attempt_index: int) -> None:
    time.sleep(min(backoff_seconds * (2**attempt_index), MAX_ARTIFACT_PUBLISH_BACKOFF_SECONDS))


def _put_worker_artifact_deferred(
    *,
    callback_base_url: str,
    run_id: str,
    attempt_id: str,
    name: str,
    path: Path,
) -> dict[str, Any]:
    if not ARTIFACT_NAME_PATTERN.fullmatch(name) or name in _RESULT_READY_ARTIFACT_NAMES:
        raise ValueError("Deferred artifact name is invalid.")
    size_bytes = path.stat().st_size
    if not 0 <= size_bytes <= MAX_R2_OBJECT_BYTES:
        raise ValueError("Deferred artifact size is invalid.")
    content_type = _artifact_content_type(path)
    endpoint = (
        f"{callback_base_url.rstrip('/')}/v1/jobs/{run_id}/artifacts/"
        f"{urllib.parse.quote(name, safe='')}?defer_index=1"
    )
    attempts, backoff_seconds = _artifact_publish_retry_settings()
    response_body: bytes | None = None
    for attempt_index in range(attempts):
        try:
            with path.open("rb") as artifact:
                request = urllib.request.Request(
                    endpoint,
                    data=artifact,
                    method="PUT",
                    headers={
                        **_request_headers(),
                        "X-Processing-Attempt-Id": attempt_id,
                        "Accept": "application/json",
                        "Content-Type": content_type,
                        "Content-Length": str(size_bytes),
                    },
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    response_body = response.read(MAX_ARTIFACT_FINALIZE_BODY_BYTES + 1)
            break
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            if not _retryable_artifact_publish_error(error) or attempt_index + 1 >= attempts:
                raise
            _artifact_publish_retry_delay(backoff_seconds, attempt_index)

    if response_body is None:
        raise RuntimeError("Deferred artifact upload completed without a response.")

    if len(response_body) > MAX_ARTIFACT_FINALIZE_BODY_BYTES:
        raise ValueError("Deferred artifact response is too large.")
    try:
        receipt = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Deferred artifact response must be valid JSON.") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("run_id") != run_id
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("artifact") != name
        or receipt.get("status") != "stored_unindexed"
        or receipt.get("content_type") != content_type
        or not isinstance(receipt.get("object_version"), str)
        or not R2_OBJECT_VERSION_PATTERN.fullmatch(receipt["object_version"])
        or receipt.get("size_bytes") != size_bytes
    ):
        raise ValueError("Deferred artifact response metadata did not match the upload.")
    return {
        "name": name,
        "content_type": content_type,
        "object_version": receipt["object_version"],
        "size_bytes": size_bytes,
    }


def _finalize_worker_artifacts(
    *,
    callback_base_url: str,
    run_id: str,
    attempt_id: str,
    artifacts: list[dict[str, Any]],
) -> None:
    if not artifacts:
        return
    if len(artifacts) > MAX_ARTIFACT_FINALIZE_COUNT:
        raise ValueError("Deferred artifact finalization contains too many artifacts.")
    names: set[str] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    for artifact in artifacts:
        name = artifact.get("name")
        content_type = artifact.get("content_type")
        object_version = artifact.get("object_version")
        size_bytes = artifact.get("size_bytes")
        if (
            not isinstance(name, str)
            or not ARTIFACT_NAME_PATTERN.fullmatch(name)
            or name in _RESULT_READY_ARTIFACT_NAMES
            or name in names
            or not isinstance(content_type, str)
            or not 3 <= len(content_type) <= 200
            or not ARTIFACT_CONTENT_TYPE_PATTERN.fullmatch(content_type)
            or not isinstance(object_version, str)
            or not R2_OBJECT_VERSION_PATTERN.fullmatch(object_version)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or not 0 <= size_bytes <= MAX_R2_OBJECT_BYTES
        ):
            raise ValueError("Deferred artifact finalization metadata is invalid.")
        names.add(name)
        normalized_artifacts.append(
            {
                "name": name,
                "content_type": content_type,
                "object_version": object_version,
                "size_bytes": size_bytes,
            }
        )

    body = json.dumps(
        {
            "attempt_id": attempt_id,
            "artifacts": normalized_artifacts,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(body) > MAX_ARTIFACT_FINALIZE_BODY_BYTES:
        raise ValueError("Deferred artifact finalization body is too large.")
    endpoint = f"{callback_base_url.rstrip('/')}/v1/jobs/{run_id}/artifacts/finalize"
    timeout = _env_float(
        "WHODOIRUNLIKE_REPORT_TIMEOUT_SECONDS",
        DEFAULT_REPORT_TIMEOUT_SECONDS,
    )
    attempts, backoff_seconds = _artifact_publish_retry_settings()
    for attempt_index in range(attempts):
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                **_request_headers(),
                "X-Processing-Attempt-Id": attempt_id,
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(body)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout):
                return
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            if not _retryable_artifact_publish_error(error) or attempt_index + 1 >= attempts:
                raise
            _artifact_publish_retry_delay(backoff_seconds, attempt_index)


def _write_prompt_frame(
    source_path: Path,
    prompt_frame_path: Path,
    *,
    frame_index: int = 0,
) -> dict[str, Any]:
    prompt_frame_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source upload: {source_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected_frame_index = max(0, int(frame_index))
    if frame_count:
        selected_frame_index = min(selected_frame_index, frame_count - 1)
    if selected_frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_index)
    ok, frame = cap.read()
    if (not ok or frame is None) and selected_frame_index:
        selected_frame_index = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read prompt frame from source upload: {source_path}")
    if not cv2.imwrite(str(prompt_frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"Could not write prompt frame: {prompt_frame_path}")
    return {
        "frame_index": selected_frame_index,
        "time_seconds": round(selected_frame_index / fps, 3) if fps else None,
        "image_path": str(prompt_frame_path),
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
    }


def _default_target_prompt(
    prompt_frame_path: Path,
    frame_meta: dict[str, Any],
    *,
    demo_profile: dict[str, Any] | None = None,
    uploaded_prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if demo_profile:
        return {
            "version": 1,
            "source": "hosted_upload_demo_profile_v1",
            "selection": {
                "type": "reference_box",
                "positive_points": [],
                "negative_points": [],
                "box": demo_profile["prompt_box"],
            },
            "frame": {
                **frame_meta,
                "image_path": str(prompt_frame_path),
            },
            "subject": {
                "runner_name": demo_profile["runner_name"],
                "profile_id": demo_profile["id"],
            },
            "notes": "Seeded from the validated local reference run for this public Cole Hocker demo clip.",
        }

    if uploaded_prompt:
        selection = uploaded_prompt.get("selection", {})
        return {
            "version": 1,
            "source": "hosted_upload_user_prompt_v1",
            "selection": {
                "type": selection.get("type") or ("box" if selection.get("box") else "point"),
                "positive_points": selection.get("positive_points") or [],
                "negative_points": selection.get("negative_points") or [],
                **({"box": selection["box"]} if selection.get("box") else {}),
            },
            "frame": {
                **frame_meta,
                "image_path": str(prompt_frame_path),
            },
            "notes": "Selected in the public upload UI.",
        }

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


def _uploaded_prompt_frame_index(
    uploaded_prompt: dict[str, Any] | None,
    video_meta: dict[str, Any],
) -> int:
    if not uploaded_prompt:
        return 0
    frame = uploaded_prompt.get("frame") or {}
    frame_count = max(0, int(video_meta.get("frame_count") or 0))
    fps = float(video_meta.get("fps") or 0.0)
    frame_index: int
    if frame.get("frame_index") is not None:
        frame_index = int(frame.get("frame_index") or 0)
    elif frame.get("time_seconds") is not None and fps > 0:
        frame_index = int(round(float(frame.get("time_seconds") or 0.0) * fps))
    else:
        frame_index = 0
    if frame_count:
        return max(0, min(frame_index, frame_count - 1))
    return max(0, frame_index)


def _write_hosted_manifest(
    *,
    run_dir: Path,
    payload: WorkerJobRequest,
    source_path: Path,
    video_meta: dict[str, Any],
    demo_profile: dict[str, Any] | None = None,
    uploaded_prompt: dict[str, Any] | None = None,
) -> Path:
    run = RunningClipRun(run_dir)
    demo_profile = _active_demo_profile(demo_profile, uploaded_prompt)
    prompt_frame_path = run_dir / "prompt_frame.jpg"
    prompt_frame_index = (
        int(demo_profile["prompt_frame_index"])
        if demo_profile
        else _uploaded_prompt_frame_index(uploaded_prompt, video_meta)
    )
    frame_meta = _write_prompt_frame(
        source_path,
        prompt_frame_path,
        frame_index=prompt_frame_index,
    )
    prompt_path = run_dir / "person_prompt.json"
    prompt = _default_target_prompt(
        prompt_frame_path,
        frame_meta,
        demo_profile=demo_profile,
        uploaded_prompt=uploaded_prompt,
    )
    write_json(prompt_path, prompt)

    paths = run.canonical_paths()
    fps = float(video_meta.get("fps") or 0.0)
    frame_count = int(video_meta.get("frame_count") or 0)
    duration_seconds = round(frame_count / fps, 3) if fps else None
    runner_name = demo_profile["runner_name"] if demo_profile else "Uploaded runner"
    runner_slug = demo_profile["runner_slug"] if demo_profile else "uploaded-runner"
    target_lock_method = (
        "demo_reference_prompt"
        if demo_profile
        else "uploaded_runner_prompt"
        if uploaded_prompt
        else "auto_center_prompt"
    )
    prompt_stage_status = "user_selected" if uploaded_prompt and not demo_profile else "auto_seeded"

    write_json(
        run_dir / "track_seed.json",
        {
            "candidate_id": payload.run_id,
            "runner_name": runner_name,
            "prompt_path": str(prompt_path),
            "target_lock_method": target_lock_method,
            "updated_at": utc_now_iso(),
        },
    )
    write_json(
        run_dir / "view_bucket.json",
        {
            "candidate_id": payload.run_id,
            "runner_name": runner_name,
            "view_bucket": "unknown",
            "source": "hosted_upload",
            "updated_at": utc_now_iso(),
        },
    )

    manifest = {
        "version": 1,
        "created_at": utc_now_iso(),
        "candidate_id": payload.run_id,
        "runner_name": runner_name,
        "runner_slug": runner_slug,
        "implementation_goal": "identity_stable_runner_clip",
        "source": {
            "platform": "hosted_upload",
            "worker_source_url": payload.source.url,
            "worker_object_key": payload.source.key,
            "filename": payload.source.filename,
            "content_type": payload.source.content_type,
            "size_bytes": payload.source.size_bytes,
            "video_path": str(source_path),
            "sha256": demo_profile.get("source_sha256") if demo_profile else None,
        },
        "review": {
            "quality": "hosted_upload",
            "camera_angle": "unknown",
            "primary_bucket": "running",
            "duration_seconds": duration_seconds,
            "notes": (
                "Target runner selected in the public upload UI."
                if uploaded_prompt and not demo_profile
                else "Auto-seeded public upload. Review the target prompt before adding this clip to a reference set."
            ),
        },
        "video": video_meta,
        "target_prompt_source": prompt["source"],
        "demo_profile": (
            {
                "id": demo_profile["id"],
                "reference_artifacts": list(demo_profile["reference_artifacts"].keys()),
            }
            if demo_profile
            else None
        ),
        "paths": paths,
        "stages": {
            "upload": {"status": "complete", "output": str(source_path)},
            "person_prompt": {"status": prompt_stage_status, "output": str(prompt_path)},
            "detector_tracker": {"status": "pending"},
            "whole_runner_mask": {"status": "pending"},
            "pose": {"status": "pending"},
            "densepose": {"status": "pending"},
            "fused_form": {"status": "pending"},
            "features": {"status": "pending"},
            "qc_metrics": {"status": "pending"},
        },
    }
    return run.write_manifest(manifest)


def _apply_demo_reference_artifacts(
    *,
    run_dir: Path,
    demo_profile: dict[str, Any] | None,
) -> list[str]:
    if not demo_profile:
        return []

    run = RunningClipRun(run_dir)
    copied: list[str] = []
    for output_name, asset_name in demo_profile["reference_artifacts"].items():
        source = DEMO_ASSET_ROOT / asset_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing demo reference artifact: {source}")
        target = run_dir / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(output_name)

    if run.manifest_path.is_file():
        run.update_stage(
            "demo_reference_artifacts",
            {
                "status": "complete",
                "profile_id": demo_profile["id"],
                "outputs": copied,
                "updated_at": utc_now_iso(),
            },
            manifest=run.read_manifest(),
        )

    return copied


def _attach_demo_reference_summary(
    result: dict[str, Any],
    *,
    demo_profile: dict[str, Any] | None,
    copied_artifacts: list[str],
) -> dict[str, Any]:
    if not demo_profile or not copied_artifacts:
        return result

    updated = dict(result)
    updated["demo_profile"] = {
        "id": demo_profile["id"],
        "source_sha256": demo_profile["source_sha256"],
        "reference_artifacts_applied": copied_artifacts,
    }
    updated["steps"] = [
        *list(result.get("steps", [])),
        {
            "stage": "demo_reference_artifacts",
            "result": {
                "status": "complete",
                "profile_id": demo_profile["id"],
                "outputs": copied_artifacts,
            },
        },
    ]
    return updated


def _artifact_files(run_dir: Path) -> list[Path]:
    run = RunningClipRun(run_dir)
    return [
        *run.existing_artifacts(keys=(), extra_names=("cv_run_manifest.json",)),
        *run.existing_artifacts(
            keys=(key for key, _ in _HOSTED_PUBLISHABLE_ARTIFACTS),
        ),
    ]


def _public_artifact_name(
    run: RunningClipRun,
    path: Path,
    manifest: dict[str, Any] | None,
) -> str:
    if path == run.manifest_path:
        return "cv_run_manifest.json"
    for key, public_name in _HOSTED_PUBLISHABLE_ARTIFACTS:
        if path == run.artifact_path(key, manifest=manifest):
            return public_name
    return path.name


def _deferred_browser_encoding_paths(
    manifest: dict[str, Any] | None,
) -> set[Path]:
    if not isinstance(manifest, dict):
        return set()
    stage = (manifest.get("stages") or {}).get("whole_runner_mask") or {}
    deferred = stage.get("deferred_browser_encoding") or {}
    if stage.get("backend") != "yolo26n_seg_inline" or deferred.get("required") is not True:
        return set()
    values = deferred.get("paths") or []
    return {
        Path(str(value)).resolve(strict=False)
        for value in values
        if Path(str(value)).name in _DEFERRED_BROWSER_ARTIFACT_NAMES
    }


def _finalized_deferred_browser_payload(
    value: Any,
    *,
    completed_at: str,
) -> tuple[Any, bool]:
    changed = False
    if isinstance(value, list):
        items = []
        for item in value:
            finalized, item_changed = _finalized_deferred_browser_payload(
                item,
                completed_at=completed_at,
            )
            items.append(finalized)
            changed = changed or item_changed
        return items, changed
    if not isinstance(value, dict):
        return value, False

    finalized_object: dict[str, Any] = {}
    for key, item in value.items():
        finalized, item_changed = _finalized_deferred_browser_payload(
            item,
            completed_at=completed_at,
        )
        finalized_object[key] = finalized
        changed = changed or item_changed

    deferred = finalized_object.get("deferred_browser_encoding")
    if isinstance(deferred, dict) and deferred.get("required") is True:
        finalized_object["deferred_browser_encoding"] = {
            **deferred,
            "required": False,
            "paths": [],
            "completed_at": completed_at,
        }
        changed = True
    return finalized_object, changed


def _write_json_atomically(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _finalize_deferred_browser_encoding(
    run: RunningClipRun,
    *,
    completed_at: str,
) -> None:
    manifest = run.read_manifest()
    result_path = run.artifact_path("hosted_pipeline_result", manifest)
    if result_path.is_file():
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        finalized_result, result_changed = _finalized_deferred_browser_payload(
            result_payload,
            completed_at=completed_at,
        )
        if result_changed:
            _write_json_atomically(result_path, finalized_result)

    run.update_stage(
        "whole_runner_mask",
        {
            "deferred_browser_encoding": {
                "required": False,
                "paths": [],
                "completed_at": completed_at,
            }
        },
    )


def _upload_artifacts_in_parallel(
    payload: WorkerJobRequest,
    *,
    run: RunningClipRun,
    artifact_attempt_id: str,
    upload_queue: list[tuple[str, Path]],
    deferred_paths: set[Path],
    telemetry: ProcessingTelemetry | None,
) -> list[str]:
    result_ready_queue = [
        item for item in upload_queue if item[0] in _RESULT_READY_ARTIFACT_NAMES
    ]
    secondary_queue = [
        item for item in upload_queue if item[0] not in _RESULT_READY_ARTIFACT_NAMES
    ]
    uploaded: list[str] = []

    for public_name, path in result_ready_queue:
        artifact_type = public_name.rsplit(".", 1)[0]
        span_context = (
            telemetry.span(
                "artifact_publish",
                "publish",
                measurements={
                    "artifact_type": artifact_type,
                    "bytes": path.stat().st_size,
                },
            )
            if telemetry is not None
            else nullcontext()
        )
        with span_context:
            _put_worker_artifact(
                callback_base_url=payload.callback_base_url,
                run_id=payload.run_id,
                attempt_id=artifact_attempt_id,
                name=public_name,
                path=path,
            )
        uploaded.append(public_name)
        if telemetry is not None:
            telemetry.result_ready(
                {
                    "artifact_type": artifact_type,
                    "bytes": path.stat().st_size,
                }
            )

    encoded_deferred_paths: set[Path] = set()
    for public_name, path in secondary_queue:
        resolved_path = path.resolve(strict=False)
        if resolved_path not in deferred_paths:
            continue
        artifact_type = public_name.rsplit(".", 1)[0]
        encode_context = (
            telemetry.span(
                "artifact_publish",
                "encode",
                measurements={
                    "artifact_type": artifact_type,
                    "bytes": path.stat().st_size,
                    "deferred_from_stage": "target_tracking",
                },
            )
            if telemetry is not None
            else nullcontext()
        )
        with encode_context:
            make_browser_playable_mp4(path)
        encoded_deferred_paths.add(resolved_path)
    if deferred_paths and encoded_deferred_paths == deferred_paths:
        _finalize_deferred_browser_encoding(run, completed_at=utc_now_iso())

    def upload_secondary(item: tuple[str, Path]) -> dict[str, Any]:
        public_name, path = item
        artifact_type = public_name.rsplit(".", 1)[0]
        span_context = (
            telemetry.span(
                "artifact_publish",
                "publish",
                measurements={
                    "artifact_type": artifact_type,
                    "bytes": path.stat().st_size,
                },
            )
            if telemetry is not None
            else nullcontext()
        )
        with span_context:
            return _put_worker_artifact_deferred(
                callback_base_url=payload.callback_base_url,
                run_id=payload.run_id,
                attempt_id=artifact_attempt_id,
                name=public_name,
                path=path,
            )

    receipts: list[dict[str, Any]] = []
    if secondary_queue:
        configured_workers = _env_int(
            "WHODOIRUNLIKE_ARTIFACT_PUBLISH_WORKERS",
            DEFAULT_ARTIFACT_PUBLISH_WORKERS,
            minimum=1,
        )
        worker_count = min(
            MAX_ARTIFACT_PUBLISH_WORKERS,
            configured_workers,
            len(secondary_queue),
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="artifact-publish",
        ) as executor:
            receipts = list(executor.map(upload_secondary, secondary_queue))

        finalize_context = (
            telemetry.span(
                "artifact_publish",
                "publish",
                measurements={
                    "artifact_type": "secondary_artifact_index",
                    "artifact_count": len(receipts),
                    "bytes": sum(int(receipt["size_bytes"]) for receipt in receipts),
                },
            )
            if telemetry is not None
            else nullcontext()
        )
        with finalize_context:
            _finalize_worker_artifacts(
                callback_base_url=payload.callback_base_url,
                run_id=payload.run_id,
                attempt_id=artifact_attempt_id,
                artifacts=receipts,
            )
        uploaded.extend(public_name for public_name, _ in secondary_queue)
    return uploaded


def _upload_artifacts(
    payload: WorkerJobRequest,
    run_dir: Path,
    *,
    telemetry: ProcessingTelemetry | None = None,
) -> list[str]:
    run = RunningClipRun(run_dir)
    manifest = run.read_manifest() if run.manifest_path.is_file() else None
    artifact_attempt_id = ensure_attempt_id(payload.attempt_id)
    uploaded: list[str] = []
    upload_queue = [
        (_public_artifact_name(run, path, manifest), path)
        for path in _artifact_files(run_dir)
    ]
    deferred_paths = _deferred_browser_encoding_paths(manifest)
    if deferred_paths:
        upload_queue.sort(
            key=lambda item: (
                0
                if item[0] in _RESULT_READY_ARTIFACT_NAMES
                else 1
                if item[1].resolve(strict=False) in deferred_paths
                else 3
                if item[0] == "cv_run_manifest.json"
                else 2
            )
        )
    else:
        upload_queue.sort(
            key=lambda item: (
                item[0] not in _RESULT_READY_ARTIFACT_NAMES,
                _RESULT_READY_ARTIFACT_NAMES.index(item[0])
                if item[0] in _RESULT_READY_ARTIFACT_NAMES
                else 0,
            )
        )
    if _env_bool("WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH", False):
        return _upload_artifacts_in_parallel(
            payload,
            run=run,
            artifact_attempt_id=artifact_attempt_id,
            upload_queue=upload_queue,
            deferred_paths=deferred_paths,
            telemetry=telemetry,
        )
    encoded_deferred_paths: set[Path] = set()
    deferred_state_finalized = False
    for public_name, path in upload_queue:
        artifact_type = public_name.rsplit(".", 1)[0]
        resolved_path = path.resolve(strict=False)
        if resolved_path in deferred_paths:
            encode_context = (
                telemetry.span(
                    "artifact_publish",
                    "encode",
                    measurements={
                        "artifact_type": artifact_type,
                        "bytes": path.stat().st_size,
                        "deferred_from_stage": "target_tracking",
                    },
                )
                if telemetry is not None
                else nullcontext()
            )
            with encode_context:
                make_browser_playable_mp4(path)
            encoded_deferred_paths.add(resolved_path)
        if (
            deferred_paths
            and encoded_deferred_paths == deferred_paths
            and not deferred_state_finalized
        ):
            _finalize_deferred_browser_encoding(
                run,
                completed_at=utc_now_iso(),
            )
            deferred_state_finalized = True
        span_context = (
            telemetry.span(
                "artifact_publish",
                "publish",
                measurements={
                    "artifact_type": artifact_type,
                    "bytes": path.stat().st_size,
                },
            )
            if telemetry is not None
            else nullcontext()
        )
        with span_context:
            _put_worker_artifact(
                callback_base_url=payload.callback_base_url,
                run_id=payload.run_id,
                attempt_id=artifact_attempt_id,
                name=public_name,
                path=path,
            )
        uploaded.append(public_name)
        if telemetry is not None and public_name in _RESULT_READY_ARTIFACT_NAMES:
            telemetry.result_ready(
                {
                    "artifact_type": artifact_type,
                    "bytes": path.stat().st_size,
                }
            )
    return uploaded


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _sam31_input_loader_settings() -> dict[str, Any]:
    return sam31_exact_cv2_loader_settings().to_dict()


def _inline_mask_settings() -> dict[str, Any]:
    return {
        "identity_detector_model": os.getenv(
            "WHODOIRUNLIKE_IDENTITY_DETECTOR_MODEL",
            "",
        ).strip()
        or None,
        "inline_mask_dilation_pixels": _env_int(
            "WHODOIRUNLIKE_INLINE_MASK_DILATION_PIXELS",
            5,
        ),
        "inline_mask_temporal_reset_gap_frames": _env_int(
            "WHODOIRUNLIKE_INLINE_MASK_TEMPORAL_RESET_GAP_FRAMES",
            3,
            minimum=0,
        ),
        "inline_mask_fallback_to_track_box": _env_bool(
            "WHODOIRUNLIKE_INLINE_MASK_FALLBACK_TO_TRACK_BOX",
            True,
        ),
        "inline_mask_defer_browser_encoding": _env_bool(
            "WHODOIRUNLIKE_INLINE_MASK_DEFER_BROWSER_ENCODING",
            True,
        ),
        "inline_mask_sam_fallback": _env_bool(
            "WHODOIRUNLIKE_INLINE_MASK_SAM_FALLBACK",
            True,
        ),
        "inline_mask_fallback_backend": os.getenv(
            "WHODOIRUNLIKE_INLINE_MASK_FALLBACK_BACKEND",
            "sam31_gpu",
        ).strip()
        or "sam31_gpu",
    }


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _invocation_runtime_metadata(payload: WorkerJobRequest) -> dict[str, Any]:
    global _PROCESSOR_INVOCATION_COUNT
    with _PROCESSOR_INVOCATION_LOCK:
        _PROCESSOR_INVOCATION_COUNT += 1
        invocation_index = _PROCESSOR_INVOCATION_COUNT
    return {
        "execution_environment": "runpod" if payload.runpod_job_id else "direct",
        "environment": (
            os.getenv("WHODOIRUNLIKE_ENVIRONMENT", "").strip()
            or ("production" if payload.runpod_job_id else "development")
        ),
        "runpod_job_id": payload.runpod_job_id,
        "runpod_delay_time_ms": payload.runpod_delay_time_ms,
        "attempt_started_at": payload.attempt_started_at,
        "processor_enqueued_at": payload.processor_enqueued_at,
        "attempt_number": payload.attempt_number,
        "process_invocation_index": invocation_index,
        "cold_start": invocation_index == 1,
        "process_uptime_seconds": max(0.0, time.monotonic() - _PROCESSOR_STARTED_AT),
    }


def _elapsed_from_timestamp(value: str | None, end: datetime) -> tuple[float, bool]:
    if not value:
        return 0.0, False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (end - parsed.astimezone(timezone.utc)).total_seconds()), True
    except (TypeError, ValueError, OverflowError):
        return 0.0, False


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
    dependencies = _dependency_status(["torch", "sam3", "einops", "PIL", "cv2", "numpy"])
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
    if normalized in INLINE_MASK_BACKENDS:
        configured_model = os.getenv(
            "WHODOIRUNLIKE_IDENTITY_DETECTOR_MODEL",
            "",
        ).strip()
        model = configured_model or DEFAULT_INLINE_SEGMENTATION_MODEL
        identity_backend = os.getenv(
            "WHODOIRUNLIKE_IDENTITY_BACKEND",
            DEFAULT_IDENTITY_BACKEND,
        )
        identity_status = identity_setup_status(identity_backend)
        reasons = list(identity_status.get("reasons") or [])
        canonical_identity_backend = identity_status.get("backend")
        if canonical_identity_backend not in BOXMOT_BACKENDS:
            reasons.append(
                "YOLO26 inline segmentation requires a BoxMOT identity backend "
                "so masks can be associated with the selected runner track."
            )
        model_path: Path | None = None
        if configured_model and not _is_url(configured_model):
            model_path = _resolve_repo_path(configured_model)
            if not model_path.is_file():
                reasons.append(
                    "WHODOIRUNLIKE_IDENTITY_DETECTOR_MODEL does not exist as a "
                    f"local file: {model_path}"
                )
        model_name = Path(urllib.parse.urlsplit(model).path).name.lower()
        model_stem = Path(model_name).stem
        if model_stem.startswith(("yolo", "yolov")) and "seg" not in model_stem:
            reasons.append(
                "WHODOIRUNLIKE_IDENTITY_DETECTOR_MODEL names a YOLO detection-only "
                f"asset ({model_name}); inline masking requires a segmentation model."
            )
        return {
            "ready": bool(identity_status.get("ready")) and not reasons,
            "reasons": reasons,
            "backend": "yolo26n_seg_inline",
            "model": model,
            "model_validation": {
                "source": (
                    "local"
                    if model_path is not None
                    else "remote"
                    if configured_model and _is_url(configured_model)
                    else "default_asset"
                ),
                "local_path": str(model_path) if model_path is not None else None,
            },
            "identity": identity_status,
        }
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


def _speed_profile_status() -> dict[str, Any]:
    raw_profile = os.getenv(SPEED_PROFILE_ENV, "").strip()
    profile = raw_profile.lower() or None
    if profile is None:
        return {
            "ready": True,
            "reasons": [],
            "profile": None,
            "active": False,
        }
    if profile != MAX_SAFE_SPEED_PROFILE:
        return {
            "ready": False,
            "reasons": [
                f"Unsupported {SPEED_PROFILE_ENV}={raw_profile!r}; "
                f"expected {MAX_SAFE_SPEED_PROFILE!r} or no profile."
            ],
            "profile": profile,
            "active": True,
        }

    reasons: list[str] = []

    def require(
        name: str,
        expected: str,
        predicate: Any,
    ) -> None:
        raw_value = os.getenv(name)
        value = (raw_value or "").strip()
        if raw_value is None or not value:
            reasons.append(
                f"{name} must be explicitly set to {expected} for "
                f"{SPEED_PROFILE_ENV}={MAX_SAFE_SPEED_PROFILE}."
            )
            return
        if not predicate(value):
            reasons.append(
                f"{name}={value!r} does not satisfy the {MAX_SAFE_SPEED_PROFILE} "
                f"requirement; expected {expected}."
            )

    def truthy(value: str) -> bool:
        return value.lower() in {"1", "true", "yes", "on"}

    def falsey(value: str) -> bool:
        return value.lower() in {"0", "false", "no", "off"}

    def cuda(value: str) -> bool:
        return re.fullmatch(r"cuda(?::\d+)?", value.lower()) is not None

    require("WHODOIRUNLIKE_ENVIRONMENT", "scratch", lambda value: value.lower() == "scratch")
    require(
        "WHODOIRUNLIKE_MASK_BACKEND",
        "a YOLO26 inline segmentation backend",
        lambda value: value.lower() in INLINE_MASK_BACKENDS,
    )
    require("WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE", "true", truthy)
    require("WHODOIRUNLIKE_PARALLEL_POST_FUSION", "true", truthy)
    require(
        "WHODOIRUNLIKE_POSE_BACKEND",
        "an mmpose backend",
        lambda value: value.lower().startswith("mmpose_"),
    )
    require("MMPOSE_DEVICE", "cuda or cuda:<index>", cuda)
    require("MMPOSE_USE_DETECTOR", "false", falsey)
    require(
        "RTMW_RUNTIME_BACKEND",
        "onnxruntime",
        lambda value: value.lower() == "onnxruntime",
    )
    require("WHODOIRUNLIKE_SKIP_DENSEPOSE", "false", falsey)
    require("DENSEPOSE_DEVICE", "cuda or cuda:<index>", cuda)
    require("DENSEPOSE_TARGET_CROP_ENABLED", "true", truthy)
    require(
        "DENSEPOSE_INPUT_MIN_SIZE_TEST",
        "512",
        lambda value: value == "512",
    )
    require(
        "DENSEPOSE_INPUT_MAX_SIZE_TEST",
        "960",
        lambda value: value == "960",
    )
    require("WHODOIRUNLIKE_INLINE_MASK_DEFER_BROWSER_ENCODING", "true", truthy)
    require("WHODOIRUNLIKE_INLINE_MASK_SAM_FALLBACK", "true", truthy)
    require(
        "WHODOIRUNLIKE_INLINE_MASK_FALLBACK_BACKEND",
        "sam31_gpu",
        lambda value: value.lower()
        in {"sam31_gpu", "sam3.1_gpu", "sam31_cuda", "sam3.1_cuda"},
    )
    return {
        "ready": not reasons,
        "reasons": reasons,
        "profile": profile,
        "active": True,
    }


def processor_readiness() -> dict[str, Any]:
    identity_backend = os.getenv("WHODOIRUNLIKE_IDENTITY_BACKEND", DEFAULT_IDENTITY_BACKEND)
    pose_backend = os.getenv("WHODOIRUNLIKE_POSE_BACKEND", "mmpose_rtmpose_l_384")
    mask_backend = _mask_backend()
    skip_densepose = _env_bool("WHODOIRUNLIKE_SKIP_DENSEPOSE")
    parallel_mask_presentation = _env_bool("WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION")
    parallel_pose_densepose = _env_bool("WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE")
    parallel_post_fusion = _env_bool("WHODOIRUNLIKE_PARALLEL_POST_FUSION")
    inline_mask_settings = _inline_mask_settings()
    sam31_input_loader = _sam31_input_loader_settings()
    execution_policy_reasons: list[str] = []
    if (
        parallel_pose_densepose
        and not skip_densepose
        and not pose_backend.startswith("mmpose_")
    ):
        execution_policy_reasons.append(
            "WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE requires an mmpose backend "
            "with isolated QA output."
        )
    if parallel_mask_presentation and (
        mask_backend.strip().lower()
        not in {"sam31_gpu", "sam3.1_gpu", "sam31_cuda", "sam3.1_cuda"}
        or not pose_backend.startswith("mmpose_")
        or skip_densepose
    ):
        execution_policy_reasons.append(
            "WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION requires sam31_gpu, "
            "an mmpose backend, and DensePose enabled."
        )
    if (
        sam31_input_loader["enabled"]
        and not sam31_input_loader["concurrency_ready"]
    ):
        execution_policy_reasons.append(
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER requires "
            "WHODOIRUNLIKE_PROCESSOR_CONCURRENCY=1; configured concurrency is "
            f"{sam31_input_loader['configured_concurrency']}."
        )
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
        "execution_policy": {
            "ready": not execution_policy_reasons,
            "reasons": execution_policy_reasons,
        },
        "speed_profile": _speed_profile_status(),
    }
    return {
        "ready_for_full_pipeline": all(bool(check.get("ready")) for check in checks.values()),
        "speed_profile": checks["speed_profile"]["profile"],
        "identity_backend": identity_backend,
        "pose_backend": pose_backend,
        "mask_backend": mask_backend,
        "skip_densepose": skip_densepose,
        "parallel_mask_presentation": parallel_mask_presentation,
        "parallel_pose_densepose": parallel_pose_densepose,
        "parallel_post_fusion": parallel_post_fusion,
        "inline_mask": inline_mask_settings,
        "sam31_input_loader": sam31_input_loader,
        "checks": checks,
    }


def process_hosted_job(payload: WorkerJobRequest, *, raise_on_error: bool = False) -> dict[str, Any]:
    _validate_job_payload(payload)
    started = time.monotonic()
    invocation_started_at = datetime.now(timezone.utc)
    run_dir = _hosted_run_root() / payload.run_id
    source_path = run_dir / "source_segment.mp4"
    attempt_id = ensure_attempt_id(payload.attempt_id)
    if payload.attempt_id != attempt_id:
        payload = payload.model_copy(update={"attempt_id": attempt_id})
    identity_backend = os.getenv("WHODOIRUNLIKE_IDENTITY_BACKEND", DEFAULT_IDENTITY_BACKEND)
    pose_backend = os.getenv("WHODOIRUNLIKE_POSE_BACKEND", "mmpose_rtmpose_l_384")
    mask_backend = _mask_backend()
    skip_densepose = _env_bool("WHODOIRUNLIKE_SKIP_DENSEPOSE")
    parallel_mask_presentation = _env_bool("WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION")
    parallel_pose_densepose = _env_bool("WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE")
    parallel_post_fusion = _env_bool("WHODOIRUNLIKE_PARALLEL_POST_FUSION")
    inline_mask_settings = _inline_mask_settings()
    sam31_input_loader = _sam31_input_loader_settings()
    attempt_elapsed_offset, has_attempt_timing = _elapsed_from_timestamp(
        payload.attempt_started_at or payload.processor_enqueued_at,
        invocation_started_at,
    )
    telemetry = create_hosted_telemetry(
        run_id=payload.run_id,
        attempt_id=attempt_id,
        run_dir=run_dir,
        callback_base_url=payload.callback_base_url,
        auth_token=_processor_secret(),
        input_metadata={
            "size_bytes": payload.source.size_bytes,
            "content_type": payload.source.content_type,
        },
        runtime_metadata={
            **_invocation_runtime_metadata(payload),
            "identity_backend": identity_backend,
            "pose_backend": pose_backend,
            "mask_backend": mask_backend,
            "skip_densepose": skip_densepose,
            "parallel_mask_presentation": parallel_mask_presentation,
            "parallel_pose_densepose": parallel_pose_densepose,
            "parallel_post_fusion": parallel_post_fusion,
            "sam31_input_loader_mode": sam31_input_loader["mode"],
            "sam31_exact_cv2_loader_enabled": sam31_input_loader["enabled"],
            "sam31_exact_cv2_chunk_frames": sam31_input_loader["chunk_frames"],
            "sam31_exact_cv2_max_frames": sam31_input_loader["max_frames"],
            "sam31_exact_cv2_max_destination_bytes": sam31_input_loader[
                "max_destination_bytes"
            ],
            "sam31_exact_cv2_required_concurrency": sam31_input_loader[
                "required_concurrency"
            ],
            "sam31_exact_cv2_configured_concurrency": sam31_input_loader[
                "configured_concurrency"
            ],
            "sam31_exact_cv2_concurrency_ready": sam31_input_loader[
                "concurrency_ready"
            ],
            **inline_mask_settings,
            "attempt_timing_available": has_attempt_timing,
        },
        sequence_start=max(100, payload.telemetry_sequence_start or 100),
        attempt_elapsed_offset_seconds=(
            attempt_elapsed_offset + max(0.0, time.monotonic() - started)
        ),
    )
    try:
        speed_profile_status = _speed_profile_status()
        if speed_profile_status["active"] and not speed_profile_status["ready"]:
            reasons = "; ".join(speed_profile_status["reasons"])
            raise RuntimeError(
                f"Speed profile {speed_profile_status['profile']!r} is not ready: {reasons}"
            )
        _post_worker_report_best_effort(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={
                "status": "running",
                "attempt_id": attempt_id,
                "progress": {"phase": "downloading_upload"},
            },
        )
        with telemetry.stage("source_download") as stage_boundary:
            with telemetry.span("source_download", "write"):
                _download_source(payload, source_path)
            stage_boundary.add_measurements(
                {
                    "bytes": source_path.stat().st_size,
                    "expected_bytes": payload.source.size_bytes,
                }
            )

        with telemetry.stage("run_preparation"):
            with telemetry.span("run_preparation", "decode"):
                demo_profile = _demo_upload_profile(source_path)
                active_demo_profile = _active_demo_profile(demo_profile, payload.target_prompt)
                video_meta = inspect_video(source_path)
                telemetry.update_input(
                    input_metadata_from_video(
                        video_meta,
                        size_bytes=payload.source.size_bytes,
                        content_type=payload.source.content_type,
                    )
                )
            with telemetry.span("run_preparation", "write"):
                _write_hosted_manifest(
                    run_dir=run_dir,
                    payload=payload,
                    source_path=source_path,
                    video_meta=video_meta,
                    demo_profile=active_demo_profile,
                    uploaded_prompt=payload.target_prompt,
                )

        _post_worker_report_best_effort(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={
                "status": "running",
                "attempt_id": attempt_id,
                "progress": {"phase": "running_full_cv_pipeline"},
            },
        )
        result = run_full_cv_pipeline(
            run_dir=run_dir,
            identity_backend=identity_backend,
            pose_backend=pose_backend,
            mask_backend=mask_backend,
            mask_quality_mode=os.getenv("WHODOIRUNLIKE_MASK_QUALITY_MODE", "native"),
            skip_densepose=skip_densepose,
            parallel_mask_presentation=parallel_mask_presentation,
            parallel_pose_densepose=parallel_pose_densepose,
            parallel_post_fusion=parallel_post_fusion,
            **inline_mask_settings,
            telemetry=telemetry,
        )
        with telemetry.stage("analysis_complete"):
            with telemetry.span("analysis_complete", "postprocess"):
                demo_artifacts = _apply_demo_reference_artifacts(
                    run_dir=run_dir,
                    demo_profile=active_demo_profile,
                )
                result = _attach_demo_reference_summary(
                    result,
                    demo_profile=active_demo_profile,
                    copied_artifacts=demo_artifacts,
                )
            with telemetry.span("analysis_complete", "write"):
                write_json(run_dir / "hosted_pipeline_result.json", result)
        telemetry.analysis_completed(
            {
                "pipeline_stage_count": len(result.get("steps", [])),
            }
        )

        _post_worker_report_best_effort(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={
                "status": "running",
                "attempt_id": attempt_id,
                "progress": {"phase": "uploading_artifacts"},
            },
        )
        with telemetry.stage("artifact_publish") as publish_boundary:
            uploaded = _upload_artifacts(payload, run_dir, telemetry=telemetry)
            publish_boundary.add_measurements({"artifact_count": len(uploaded)})
        telemetry.flush_delivery(
            timeout=_env_float(
                "WHODOIRUNLIKE_TELEMETRY_SNAPSHOT_TIMEOUT_SECONDS",
                DEFAULT_TELEMETRY_SNAPSHOT_TIMEOUT_SECONDS,
            )
        )
        telemetry.attempt_completed(
            {
                "artifact_count": len(uploaded),
                "pipeline_stage_count": len(result.get("steps", [])),
                **telemetry.delivery_measurements(),
            }
        )
        _post_worker_report_best_effort(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={
                "status": "complete",
                "attempt_id": attempt_id,
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
            "attempt_id": attempt_id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "artifacts_uploaded": uploaded,
        }
    except Exception as exc:
        telemetry.flush_delivery(
            timeout=_env_float(
                "WHODOIRUNLIKE_TELEMETRY_SNAPSHOT_TIMEOUT_SECONDS",
                DEFAULT_TELEMETRY_SNAPSHOT_TIMEOUT_SECONDS,
            )
        )
        telemetry.attempt_failed(exc, telemetry.delivery_measurements())
        run_dir.mkdir(parents=True, exist_ok=True)
        error_traceback = traceback.format_exc(limit=8)
        write_json(
            run_dir / "hosted_job_error.json",
            {
                "run_id": payload.run_id,
                "attempt_id": attempt_id,
                "error": str(exc),
                "traceback": error_traceback,
                "failed_at": utc_now_iso(),
            },
        )
        _post_worker_report_best_effort(
            callback_base_url=payload.callback_base_url,
            run_id=payload.run_id,
            payload={
                "status": "failed",
                "attempt_id": attempt_id,
                "progress": {"phase": "failed"},
                "error": f"{exc}\n\n{error_traceback[-2000:]}",
            },
        )
        if raise_on_error:
            raise
        return {
            "status": "failed",
            "run_id": payload.run_id,
            "attempt_id": attempt_id,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        _close_telemetry_delivery(
            telemetry,
            run_id=payload.run_id,
            attempt_id=attempt_id,
        )


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
