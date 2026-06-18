#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; whodoirunlike-smoke/1.0)",
}


def _json_request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={**DEFAULT_HEADERS, **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def _upload_clip(api_base_url: str, clip_path: Path) -> dict[str, Any]:
    content_type = mimetypes.guess_type(clip_path.name)[0] or "application/octet-stream"
    return _json_request(
        f"{api_base_url.rstrip('/')}/v1/uploads",
        method="POST",
        body=clip_path.read_bytes(),
        headers={
            "Content-Type": content_type,
            "X-Original-Filename": clip_path.name,
            "X-Clip-Consent": "operator-smoke-test",
        },
        timeout=120,
    )


def _poll_job(
    api_base_url: str,
    run_id: str,
    *,
    timeout_seconds: int,
    interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = _json_request(f"{api_base_url.rstrip('/')}/v1/jobs/{run_id}")
        last_payload = payload
        status = str(payload.get("status") or "")
        progress = payload.get("progress") or {}
        phase = progress.get("phase") if isinstance(progress, dict) else None
        print(f"{run_id}: {status}{f' ({phase})' if phase else ''}", flush=True)
        if status in {"complete", "failed"}:
            return payload
        time.sleep(interval_seconds)
    raise TimeoutError(
        f"Timed out waiting for {run_id}. Last payload: {json.dumps(last_payload, indent=2)}"
    )


def _assert_processor_ready(processor_base_url: str) -> dict[str, Any]:
    payload = _json_request(f"{processor_base_url.rstrip('/')}/v1/processor/health")
    readiness = payload.get("readiness") if isinstance(payload, dict) else None
    if not isinstance(readiness, dict) or not readiness.get("ready_for_full_pipeline"):
        raise RuntimeError(
            "Processor is not ready for the full pipeline:\n" + json.dumps(payload, indent=2)
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test the hosted Cloudflare Worker -> processor upload flow."
    )
    parser.add_argument("--api-base-url", default="https://api.whodoirunlike.com")
    parser.add_argument(
        "--processor-base-url",
        default="",
        help="Optional processor URL to check before uploading, for example http://127.0.0.1:8000.",
    )
    parser.add_argument("--clip", required=True, type=Path, help="Path to a short MP4/MOV/WebM clip.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    clip_path = args.clip.expanduser().resolve()
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    if args.processor_base_url:
        processor = _assert_processor_ready(args.processor_base_url)
        print(
            "Processor readiness:",
            json.dumps(processor.get("readiness", {}), indent=2),
            flush=True,
        )

    upload = _upload_clip(args.api_base_url, clip_path)
    run_id = str(upload["run_id"])
    print("Uploaded:", json.dumps(upload, indent=2), flush=True)

    started = _json_request(
        f"{args.api_base_url.rstrip('/')}/v1/jobs/{run_id}/start",
        method="POST",
        timeout=60,
    )
    print("Started:", json.dumps(started, indent=2), flush=True)

    final = _poll_job(
        args.api_base_url,
        run_id,
        timeout_seconds=args.timeout_seconds,
        interval_seconds=args.interval_seconds,
    )
    print("Final:", json.dumps(final, indent=2), flush=True)

    if final.get("status") != "complete":
        return 1
    artifacts = final.get("artifacts") or {}
    if not artifacts:
        print("Job completed but no artifacts were returned.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
