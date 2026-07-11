#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RUN_ID = "11c51cf1-c4d0-42ef-a2e1-cb9e2605ef1b"
BASE_ARTIFACT_URL = f"https://api.whodoirunlike.com/v1/artifacts/{RUN_ID}"
ASSETS = {
    "person_prompt_json": {
        "url": f"{BASE_ARTIFACT_URL}/person_prompt.json",
        "encoding": "base64",
        "sha256": "66d0760138febcd8fee2b7d944aedd68bd7ada665cc47c7253de376cf065e26c",
    },
    "tracklets_jsonl": {
        "url": f"{BASE_ARTIFACT_URL}/tracklets.jsonl",
        "encoding": "gzip+base64",
        "sha256": "47dea72891c0de6b95e7a255506c1afac1f7ee6525c2d1afe4589544fd760010",
    },
    "baseline_runner_mask_mp4": {
        "url": f"{BASE_ARTIFACT_URL}/runner_mask.mp4",
        "encoding": "base64",
        "sha256": "0edf35fb0837d4083f0f73103631b10972c69347cc13ffccafa2cb78634c443f",
    },
}
VARIANT_IDS = (
    "production_control",
    "preseed_single_pass",
    "preseed_single_pass_frame_dir",
    "preseed_single_pass_offload_video_cpu",
    "preseed_single_pass_max_objects_1",
    "probe_then_anchor_1",
    "probe_then_anchor_8",
    "probe_then_anchor_24",
    "probe_then_anchor_64",
)


def _request_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    data = (
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if payload is not None
        else None
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


def _build_assets() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in ASSETS.items():
        raw = _download(str(spec["url"]))
        digest = hashlib.sha256(raw).hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"Downloaded benchmark artifact {name} failed SHA-256 verification.")
        encoded_raw = gzip.compress(raw, compresslevel=9, mtime=0) if spec["encoding"] == "gzip+base64" else raw
        result[name] = {
            "encoding": spec["encoding"],
            "sha256": digest,
            "data": base64.b64encode(encoded_raw).decode("ascii"),
        }
    return result


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
    parser.add_argument("--variant-id", required=True, choices=VARIANT_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-zA-Z0-9]+", args.endpoint_id):
        parser.error("--endpoint-id contains unexpected characters")
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        parser.error("RUNPOD_API_KEY must be set in the environment")

    print("Fetching and verifying the fixed production comparison assets...", flush=True)
    payload = {
        "input": {
            "type": "sam31_benchmark",
            "schema_version": 1,
            "fixture_id": "cole-8.68s-260f-v1",
            "variant_id": args.variant_id,
            "assets": _build_assets(),
        },
        "policy": {"executionTimeout": 900000, "ttl": 3600000},
    }
    request_size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    if request_size >= 10_000_000:
        raise RuntimeError("Benchmark request exceeds RunPod's 10 MB async request limit.")
    print(f"Submitting {args.variant_id} ({request_size:,} request bytes)...", flush=True)
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
