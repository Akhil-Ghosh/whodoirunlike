#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    load_local_fixture_assets,
)


VARIANT_IDS = ("production_candidate_public_entrypoint",)
FULL_PROFILE_IDS = (
    "downstream_baseline_control",
    "downstream_candidate_control",
    "downstream_candidate_optimized",
    "production_control",
    "production_candidate",
)
FULL_PROFILE_MATRICES = {
    "three-arm": [
        "downstream_baseline_control",
        "downstream_candidate_control",
        "downstream_candidate_optimized",
    ],
    "production": ["production_control", "production_candidate"],
    "production-reversed": ["production_candidate", "production_control"],
    "authoritative-control": ["production_control"],
    "authoritative-candidate": ["production_candidate"],
}


def _resolve_full_profile_ids(
    *,
    profile_id: str | None,
    profile_matrix: str | None,
) -> list[str]:
    if profile_id and profile_matrix:
        raise ValueError("Choose either --profile-id or --profile-matrix, not both.")
    if profile_id:
        if profile_id not in FULL_PROFILE_IDS:
            raise ValueError(f"Unsupported full profile: {profile_id}")
        return [profile_id]
    matrix_id = profile_matrix or "three-arm"
    try:
        return list(FULL_PROFILE_MATRICES[matrix_id])
    except KeyError as exc:
        raise ValueError(f"Unsupported full profile matrix: {matrix_id}") from exc


def _request_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    data = (
        json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else None
    )
    headers = {"Accept": "application/json", "User-Agent": "wdirl-sam31-speed-lab/1"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc


def _create_artifact_sink(
    *,
    api_base_url: str,
    source_clip: Path,
) -> dict[str, str]:
    content_type = mimetypes.guess_type(source_clip.name)[0] or "video/mp4"
    body = source_clip.read_bytes()
    request = Request(
        f"{api_base_url.rstrip('/')}/v1/uploads",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "X-Original-Filename": "parity-scratch-source.mp4",
            "X-Clip-Consent": "operator-parity-scratch",
            "User-Agent": "wdirl-sam31-speed-lab/1",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            created = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        response_body = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Could not create parity artifact sink ({exc.code}): {response_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not create parity artifact sink: {exc.reason}") from exc
    run_id = created.get("run_id")
    attempt_id = created.get("attempt_id")
    if not isinstance(run_id, str) or not isinstance(attempt_id, str):
        raise RuntimeError("Parity artifact sink response omitted run_id or attempt_id.")
    return {
        "callback_base_url": api_base_url.rstrip("/"),
        "run_id": run_id,
        "attempt_id": attempt_id,
    }


def _load_or_create_artifact_sink(
    *,
    sink_file: Path,
    api_base_url: str,
    source_clip: Path,
) -> dict[str, str]:
    if sink_file.is_file():
        payload = json.loads(sink_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Artifact sink file must contain a JSON object.")
        required = {"callback_base_url", "run_id", "attempt_id"}
        if set(payload) != required or not all(
            isinstance(payload.get(key), str) and payload[key] for key in required
        ):
            raise RuntimeError("Artifact sink file has an invalid descriptor.")
        return {key: str(payload[key]) for key in required}
    if not source_clip.is_file():
        raise RuntimeError(f"Canonical source clip is unavailable: {source_clip}")
    sink = _create_artifact_sink(
        api_base_url=api_base_url,
        source_clip=source_clip,
    )
    _write_private_json(sink_file, sink)
    return sink


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "wdirl-sam31-speed-lab/1"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except HTTPError as exc:
        body = exc.read(1024).decode("utf-8", errors="replace")
        raise RuntimeError(f"Could not fetch benchmark artifact ({exc.code}): {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not fetch benchmark artifact: {exc.reason}") from exc


def _encode_assets(raw_assets: dict[str, bytes]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw in raw_assets.items():
        encoding = "gzip+base64" if name == "tracklets_jsonl" else "base64"
        digest = hashlib.sha256(raw).hexdigest()
        encoded_raw = (
            gzip.compress(raw, compresslevel=9, mtime=0) if encoding == "gzip+base64" else raw
        )
        result[name] = {
            "encoding": encoding,
            "sha256": digest,
            "data": base64.b64encode(encoded_raw).decode("ascii"),
        }
    return result


def _build_assets(
    *,
    fixture_id: str = CANONICAL_FRAME130_FIXTURE_ID,
    fixture_root: Path | None = None,
) -> dict[str, Any]:
    if fixture_root is None:
        raise RuntimeError(
            f"Fixture {fixture_id} requires --fixture-root with private local assets."
        )
    return _encode_assets(load_local_fixture_assets(fixture_id, fixture_root))


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _cancel_job(base_url: str, job_id: str, api_key: str) -> None:
    try:
        _request_json(
            f"{base_url}/cancel/{job_id}",
            api_key=api_key,
            payload={},
            timeout_seconds=60,
        )
        print(f"Cancelled RunPod job {job_id}.", file=sys.stderr, flush=True)
    except RuntimeError as exc:
        print(f"warning: could not cancel RunPod job {job_id}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated SAM 3.1 speed-lab fixture.")
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--scope", choices=("mask", "full"), default="full")
    parser.add_argument("--variant-id", choices=VARIANT_IDS)
    parser.add_argument("--profile-id", choices=FULL_PROFILE_IDS)
    parser.add_argument("--profile-matrix", choices=tuple(FULL_PROFILE_MATRICES))
    parser.add_argument(
        "--artifact-sink-file",
        type=Path,
        help=(
            "Private JSON descriptor for a scratch Cloudflare/R2 sink. The first run "
            "creates it; control and candidate runs reuse the same file."
        ),
    )
    parser.add_argument(
        "--sink-api-base-url",
        default="https://api.whodoirunlike.com",
    )
    parser.add_argument(
        "--source-clip",
        type=Path,
        default=Path("site/public/assets/demos/cole-source.mp4"),
        help="Canonical clip used only to create an unstarted scratch sink job.",
    )
    parser.add_argument(
        "--fixture-id",
        choices=(CANONICAL_FRAME130_FIXTURE_ID,),
        default=CANONICAL_FRAME130_FIXTURE_ID,
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        help=(
            "Private local fixture directory containing tracklets.jsonl and runner_mask.mp4. "
            "Files below artifacts/ are already gitignored."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-zA-Z0-9]+", args.endpoint_id):
        parser.error("--endpoint-id contains unexpected characters")
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        parser.error("RUNPOD_API_KEY must be set in the environment")

    if args.fixture_root is None:
        parser.error(f"{CANONICAL_FRAME130_FIXTURE_ID} requires --fixture-root")
    if args.scope == "mask" and (args.profile_id or args.profile_matrix):
        parser.error("--profile-id and --profile-matrix require --scope full")
    if args.scope == "full" and args.variant_id:
        parser.error("--variant-id requires --scope mask")
    if args.scope == "mask" and args.variant_id is None:
        args.variant_id = VARIANT_IDS[0]
    if args.artifact_sink_file is not None and args.scope != "full":
        parser.error("--artifact-sink-file requires --scope full")
    try:
        profile_ids = (
            _resolve_full_profile_ids(
                profile_id=args.profile_id,
                profile_matrix=args.profile_matrix,
            )
            if args.scope == "full"
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.artifact_sink_file is not None and profile_ids not in (
        ["production_control"],
        ["production_candidate"],
    ):
        parser.error(
            "--artifact-sink-file requires authoritative-control, "
            "authoritative-candidate, or the matching single --profile-id"
        )
    print(f"Loading and verifying fixture {args.fixture_id}...", flush=True)
    benchmark_input = {
        "type": "sam31_benchmark",
        "schema_version": 1,
        "scope": args.scope,
        "fixture_id": args.fixture_id,
        "assets": _build_assets(
            fixture_id=args.fixture_id,
            fixture_root=args.fixture_root,
        ),
    }
    if args.scope == "mask":
        benchmark_input["variant_id"] = args.variant_id
    else:
        benchmark_input["profile_ids"] = profile_ids
        if args.artifact_sink_file is not None:
            benchmark_input["artifact_sink"] = _load_or_create_artifact_sink(
                sink_file=args.artifact_sink_file,
                api_base_url=args.sink_api_base_url,
                source_clip=args.source_clip,
            )
    payload = {
        "input": benchmark_input,
        "policy": {"executionTimeout": 900000, "ttl": 3600000},
    }
    request_size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    if request_size >= 10_000_000:
        raise RuntimeError("Benchmark request exceeds RunPod's 10 MB async request limit.")
    selection = args.variant_id if args.scope == "mask" else ",".join(profile_ids or [])
    print(f"Submitting {args.scope}:{selection} ({request_size:,} request bytes)...", flush=True)
    base_url = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    submission = _request_json(
        f"{base_url}/run",
        api_key=api_key,
        payload=payload,
        timeout_seconds=90,
    )
    job_id = submission.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("RunPod submission did not return a job ID.")
    print(f"RunPod job: {job_id}", flush=True)

    try:
        deadline = time.monotonic() + args.timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            status_payload = _request_json(
                f"{base_url}/status/{job_id}",
                api_key=api_key,
                timeout_seconds=60,
            )
            status = status_payload.get("status")
            if status != last_status:
                delay_ms = status_payload.get("delayTime")
                execution_ms = status_payload.get("executionTime")
                print(
                    f"status={status} delay_ms={delay_ms} execution_ms={execution_ms}",
                    flush=True,
                )
                last_status = status
            if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                _write_private_json(args.output, status_payload)
                print(f"Saved result to {args.output}", flush=True)
                return 0 if status == "COMPLETED" else 1
            time.sleep(max(1.0, args.poll_seconds))
    except KeyboardInterrupt:
        _cancel_job(base_url, job_id, api_key)
        raise

    _cancel_job(base_url, job_id, api_key)
    raise TimeoutError(f"Timed out waiting for RunPod job {job_id}; cancellation was requested.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
