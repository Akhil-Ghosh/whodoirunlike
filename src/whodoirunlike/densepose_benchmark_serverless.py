from __future__ import annotations

import multiprocessing
import os
from multiprocessing.connection import Connection
from typing import Any

from whodoirunlike.densepose_benchmark import (
    ALLOWED_BATCH_SIZES,
    BENCHMARK_RESULT_TYPE,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_TYPE,
    BENCHMARK_PROFILE_IDS,
    CANONICAL_FIXTURE_ID,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    BenchmarkRequest,
    BenchmarkResponseTooLarge,
    benchmark_profile,
    bounded_failure,
    ensure_bounded_response,
    run_benchmark,
    runtime_identity,
    validate_request,
)


ENABLE_ENV = "WHODOIRUNLIKE_ENABLE_DENSEPOSE_BATCH_BENCHMARK"
CHILD_TIMEOUT_SECONDS = 840.0


def _enabled() -> bool:
    return os.getenv(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def health() -> dict[str, Any]:
    profile_id = benchmark_profile()
    try:
        runtime_identity()
        runtime_identity_pinned = True
    except ValueError:
        runtime_identity_pinned = False
    return ensure_bounded_response(
        {
            "type": BENCHMARK_RESULT_TYPE,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "ok",
            "service": "densepose_batch_benchmark",
            "benchmark_enabled": _enabled(),
            "runtime_identity_pinned": runtime_identity_pinned,
            "fixture_id": CANONICAL_FIXTURE_ID,
            "benchmark_profile": profile_id,
            "benchmark_profile_ids": sorted(BENCHMARK_PROFILE_IDS),
            "allowed_batch_sizes": list(ALLOWED_BATCH_SIZES),
            "request_limit_bytes": MAX_REQUEST_BYTES,
            "response_limit_bytes": MAX_RESPONSE_BYTES,
            "isolation": "fresh_spawned_process_per_matrix",
        }
    )


def _error_code(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return "cuda_out_of_memory"
    if isinstance(exc, BenchmarkResponseTooLarge):
        return "response_too_large"
    return "benchmark_execution_failed"


def _child_entry(request: BenchmarkRequest, sender: Connection) -> None:
    try:
        response = run_benchmark(request)
        response = ensure_bounded_response(response)
    except BaseException as exc:
        response = bounded_failure(
            _error_code(exc),
            exception_type=type(exc).__name__,
        )
    try:
        sender.send(response)
    finally:
        sender.close()


def run_benchmark_isolated(
    request: BenchmarkRequest,
    *,
    timeout_seconds: float = CHILD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_entry,
        args=(request, sender),
        name="densepose-batch-benchmark",
        daemon=False,
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout=max(1.0, float(timeout_seconds))):
            process.terminate()
            process.join(timeout=10)
            return bounded_failure("benchmark_timeout")
        try:
            response = receiver.recv()
        except EOFError:
            response = bounded_failure("benchmark_child_exited")
    finally:
        receiver.close()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        return bounded_failure("benchmark_child_shutdown_timeout")
    if not isinstance(response, dict):
        return bounded_failure("benchmark_invalid_child_response")
    try:
        return ensure_bounded_response(response)
    except BenchmarkResponseTooLarge:
        return bounded_failure("response_too_large")


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod job input must be a benchmark payload object")
    if payload.get("type") == "health":
        return health()
    if payload.get("type") != BENCHMARK_TYPE:
        raise ValueError("This isolated endpoint accepts only health and DensePose benchmark jobs")
    if not _enabled():
        raise RuntimeError(f"DensePose batch benchmark is disabled; set {ENABLE_ENV}=true")
    request = validate_request(payload)
    return run_benchmark_isolated(request)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
