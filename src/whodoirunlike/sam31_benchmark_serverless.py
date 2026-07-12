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
    validate_final_candidate_identity,
    verify_densepose_weights,
    verify_non_overlay_production_files,
)
from whodoirunlike.sam31_loader_config import sam31_exact_cv2_loader_settings


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


def _final_candidate_identity(*, enforce: bool) -> dict[str, Any] | None:
    if os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate") != "final_candidate":
        return None
    result = validate_final_candidate_identity(
        commit=os.getenv("WHODOIRUNLIKE_CANDIDATE_COMMIT", ""),
        image_digest=os.getenv("WHODOIRUNLIKE_CANDIDATE_IMAGE_DIGEST", ""),
        image_reference=os.getenv("WHODOIRUNLIKE_FINAL_CANDIDATE_IMAGE_REFERENCE", ""),
    )
    if enforce and not result["passed"]:
        failed = [name for name, passed in result["checks"].items() if not passed]
        raise RuntimeError("Final candidate identity failed: " + ", ".join(failed))
    return result


def _enforce_final_candidate_request(payload: dict[str, Any]) -> None:
    if os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate") != "final_candidate":
        return
    if payload.get("scope") != "full":
        raise ValueError("Final candidate parity accepts only the full-pipeline scope.")
    if payload.get("profile_ids") != ["production_final_candidate"]:
        raise ValueError(
            "Final candidate parity requires exactly production_final_candidate."
        )
    if not isinstance(payload.get("artifact_sink"), dict):
        raise ValueError("Final candidate parity requires the exact control handoff sink.")


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod job input must be a benchmark payload object.")

    request_type = payload.get("type")
    if request_type == "health":
        image_role = os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate")
        final_candidate = image_role == "final_candidate"
        contract = _base_contract(enforce=False) if _contract_enforced() else None
        final_candidate_identity = _final_candidate_identity(enforce=False)
        input_loader = sam31_exact_cv2_loader_settings().to_dict()
        densepose_weights = (
            verify_densepose_weights()
            if final_candidate
            else {"status": "not_applicable"}
        )
        return {
            "status": "ok",
            "service": "sam31_speed_lab",
            "benchmark_enabled": _enabled(),
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "fixture_id": CANONICAL_FRAME130_FIXTURE_ID,
            "fixture_ids": list(BENCHMARK_FIXTURE_IDS),
            "scope_ids": ["full"] if final_candidate else ["mask", "full"],
            "variant_ids": sorted(BENCHMARK_VARIANT_IDS),
            "full_profile_ids": (
                ["production_final_candidate"]
                if final_candidate
                else sorted(PIPELINE_BENCHMARK_PROFILES)
            ),
            "default_full_profile_ids": (
                ["production_final_candidate"]
                if final_candidate
                else list(DEFAULT_PIPELINE_PROFILE_MATRIX)
            ),
            "candidate_commit": os.getenv(
                "WHODOIRUNLIKE_CANDIDATE_COMMIT",
                EXACT_CANDIDATE_COMMIT,
            ),
            "candidate_image_digest": os.getenv(
                "WHODOIRUNLIKE_CANDIDATE_IMAGE_DIGEST",
                EXACT_CANDIDATE_IMAGE_DIGEST,
            ),
            "base_image_role": image_role,
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
            "final_candidate_image_reference": os.getenv(
                "WHODOIRUNLIKE_FINAL_CANDIDATE_IMAGE_REFERENCE",
                "",
            ),
            "final_candidate_identity": final_candidate_identity
            or {"status": "not_applicable"},
            "sam31_input_loader": input_loader,
            "densepose_weights": densepose_weights,
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
        _final_candidate_identity(enforce=True)
    _enforce_final_candidate_request(payload)
    if payload.get("scope", "mask") == "full":
        return run_full_pipeline_benchmark(payload)
    return run_benchmark(payload)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
