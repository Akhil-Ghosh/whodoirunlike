from __future__ import annotations

import os
from typing import Any

from whodoirunlike.hosted_processor import (
    WorkerJobRequest,
    process_hosted_job,
    processor_readiness,
)


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

    request = WorkerJobRequest.model_validate(payload)
    return process_hosted_job(request, raise_on_error=True)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
