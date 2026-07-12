from __future__ import annotations

import os
from typing import Any


def _sam31_input_loader_health() -> dict[str, Any]:
    enabled = os.getenv(
        "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    try:
        chunk_frames = int(
            os.getenv("WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES", "8")
        )
    except (TypeError, ValueError):
        chunk_frames = 8
    chunk_frames = max(1, min(64, chunk_frames))
    return {
        "mode": "exact_cv2" if enabled else "upstream",
        "enabled": enabled,
        "chunk_frames": chunk_frames,
    }


def _shallow_health() -> dict[str, Any]:
    return {
        "ready_for_invocation": True,
        "has_processor_secret": bool(os.getenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "").strip()),
        "has_hf_token": bool(
            os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
        ),
        "identity_backend": os.getenv("WHODOIRUNLIKE_IDENTITY_BACKEND", ""),
        "pose_backend": os.getenv("WHODOIRUNLIKE_POSE_BACKEND", ""),
        "mask_backend": os.getenv("WHODOIRUNLIKE_MASK_BACKEND", ""),
        "skip_densepose": os.getenv("WHODOIRUNLIKE_SKIP_DENSEPOSE", ""),
        "sam31_input_loader": _sam31_input_loader_health(),
    }


def processor_readiness() -> dict[str, Any]:
    from whodoirunlike.hosted_processor import processor_readiness as read_processor_readiness

    return read_processor_readiness()


def process_hosted_job(request: Any, *, raise_on_error: bool) -> dict[str, Any]:
    from whodoirunlike.hosted_processor import process_hosted_job as process_worker_job

    return process_worker_job(request, raise_on_error=raise_on_error)


def _parse_worker_job_request(payload: dict[str, Any]) -> Any:
    from whodoirunlike.hosted_processor import WorkerJobRequest

    return WorkerJobRequest.model_validate(payload)


def _provider_delay_milliseconds(event: dict[str, Any]) -> float | None:
    candidates = (
        ("delayTime", 1.0),
        ("delay_time_ms", 1.0),
        ("delayTimeMs", 1.0),
        ("delay_seconds", 1000.0),
    )
    for key, multiplier in candidates:
        value = event.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed >= 0 and parsed != float("inf"):
            return parsed * multiplier
    return None


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod job input must be a processor payload object.")

    if payload.get("type") == "health":
        if payload.get("level") != "deep":
            return {
                "status": "ok",
                "health": _shallow_health(),
            }
        return {
            "status": "ok",
            "readiness": processor_readiness(),
        }

    request_payload = dict(payload)
    runpod_job_id = event.get("id")
    if runpod_job_id not in (None, "") and not request_payload.get("runpod_job_id"):
        request_payload["runpod_job_id"] = str(runpod_job_id)
    provider_delay_ms = _provider_delay_milliseconds(event)
    if provider_delay_ms is not None and request_payload.get("runpod_delay_time_ms") is None:
        request_payload["runpod_delay_time_ms"] = provider_delay_ms
    request = _parse_worker_job_request(request_payload)
    return process_hosted_job(request, raise_on_error=True)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
