from __future__ import annotations

import os
from typing import Any

from whodoirunlike.pipeline_parity import (
    DEFAULT_PIPELINE_PROFILE_MATRIX,
    PIPELINE_BENCHMARK_PROFILES,
    run_full_pipeline_benchmark,
)
from whodoirunlike.sam31_benchmark import (
    BENCHMARK_FIXTURE_IDS,
    BENCHMARK_VARIANT_IDS,
    run_benchmark,
)
from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    EXACT_CANDIDATE_COMMIT,
    EXACT_CANDIDATE_IMAGE_DIGEST,
    verify_non_overlay_production_files,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_TYPE = "sam31_benchmark"


def _enabled() -> bool:
    return os.getenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _base_contract(*, enforce: bool) -> dict[str, Any]:
    image_role = os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate")
    contract = verify_non_overlay_production_files(image_role)
    if enforce and not contract["passed"]:
        raise RuntimeError(
            "Benchmark base image contract failed: " + ", ".join(contract["mismatches"])
        )
    return contract


def _contract_enforced() -> bool:
    return os.getenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", "").strip().lower() in {
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
        contract = _base_contract(enforce=False) if _contract_enforced() else None
        return {
            "status": "ok",
            "service": "sam31_speed_lab",
            "benchmark_enabled": _enabled(),
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "fixture_id": CANONICAL_FRAME130_FIXTURE_ID,
            "fixture_ids": list(BENCHMARK_FIXTURE_IDS),
            "scope_ids": ["mask", "full"],
            "variant_ids": sorted(BENCHMARK_VARIANT_IDS),
            "full_profile_ids": sorted(PIPELINE_BENCHMARK_PROFILES),
            "default_full_profile_ids": list(DEFAULT_PIPELINE_PROFILE_MATRIX),
            "candidate_commit": os.getenv(
                "WHODOIRUNLIKE_CANDIDATE_COMMIT",
                EXACT_CANDIDATE_COMMIT,
            ),
            "candidate_image_digest": os.getenv(
                "WHODOIRUNLIKE_CANDIDATE_IMAGE_DIGEST",
                EXACT_CANDIDATE_IMAGE_DIGEST,
            ),
            "base_image_role": os.getenv(
                "WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE",
                "candidate",
            ),
            "base_processor_commit": os.getenv(
                "WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT",
                EXACT_CANDIDATE_COMMIT,
            ),
            "base_processor_image_digest": os.getenv(
                "WHODOIRUNLIKE_BASE_PROCESSOR_IMAGE_DIGEST",
                EXACT_CANDIDATE_IMAGE_DIGEST,
            ),
            "code_overlay_commit": os.getenv(
                "WHODOIRUNLIKE_CODE_OVERLAY_COMMIT",
                os.getenv("WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT", EXACT_CANDIDATE_COMMIT),
            ),
            "code_overlay_source": os.getenv(
                "WHODOIRUNLIKE_CODE_OVERLAY_SOURCE",
                "base_image",
            ),
            "code_overlay_reference_image_digest": os.getenv(
                "WHODOIRUNLIKE_CODE_OVERLAY_REFERENCE_IMAGE_DIGEST",
                os.getenv(
                    "WHODOIRUNLIKE_BASE_PROCESSOR_IMAGE_DIGEST",
                    EXACT_CANDIDATE_IMAGE_DIGEST,
                ),
            ),
            "dependency_base_role": os.getenv(
                "WHODOIRUNLIKE_DEPENDENCY_BASE_ROLE",
                os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate"),
            ),
            "dependency_base_commit": os.getenv(
                "WHODOIRUNLIKE_DEPENDENCY_BASE_COMMIT",
                os.getenv("WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT", EXACT_CANDIDATE_COMMIT),
            ),
            "dependency_base_image_digest": os.getenv(
                "WHODOIRUNLIKE_DEPENDENCY_BASE_IMAGE_DIGEST",
                os.getenv(
                    "WHODOIRUNLIKE_BASE_PROCESSOR_IMAGE_DIGEST",
                    EXACT_CANDIDATE_IMAGE_DIGEST,
                ),
            ),
            "base_contract": contract or {"status": "not_enforced"},
            "has_hf_token": bool(
                os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
            ),
        }

    if request_type != BENCHMARK_TYPE:
        raise ValueError("This isolated endpoint accepts only health and sam31_benchmark jobs.")
    if not _enabled():
        raise RuntimeError("SAM 3.1 benchmark execution is disabled for this image.")
    if _contract_enforced():
        _base_contract(enforce=True)
    if payload.get("scope", "mask") == "full":
        return run_full_pipeline_benchmark(payload)
    return run_benchmark(payload)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
