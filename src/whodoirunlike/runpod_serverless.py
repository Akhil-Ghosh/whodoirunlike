from __future__ import annotations

from typing import Any

from whodoirunlike.hosted_processor import (
    WorkerJobRequest,
    process_hosted_job,
    processor_readiness,
)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod job input must be a processor payload object.")

    if payload.get("type") == "health":
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
