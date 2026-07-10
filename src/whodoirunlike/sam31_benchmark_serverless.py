from __future__ import annotations

import os
from typing import Any

from whodoirunlike.sam31_benchmark import (
    BENCHMARK_FIXTURE_ID,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_TYPE,
    VARIANTS,
    run_benchmark,
)


def _enabled() -> bool:
    return os.getenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod job input must be a benchmark payload object.")

    request_type = payload.get("type")
    if request_type == "health":
        return {
            "status": "ok",
            "service": "sam31_speed_lab",
            "benchmark_enabled": _enabled(),
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "fixture_id": BENCHMARK_FIXTURE_ID,
            "variant_ids": sorted(VARIANTS),
            "has_hf_token": bool(
                os.getenv("HF_TOKEN", "").strip()
                or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
            ),
        }

    if request_type != BENCHMARK_TYPE:
        raise ValueError("This isolated endpoint accepts only health and sam31_benchmark jobs.")
    if not _enabled():
        raise RuntimeError("SAM 3.1 benchmark execution is disabled for this image.")
    return run_benchmark(payload)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
