from __future__ import annotations

import os
from typing import Any


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

    request = _parse_worker_job_request(payload)
    return process_hosted_job(request, raise_on_error=True)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
