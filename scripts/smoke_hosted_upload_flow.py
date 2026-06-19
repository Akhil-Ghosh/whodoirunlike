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


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _upload_clip(
    api_base_url: str,
    clip_path: Path,
    *,
    runner_prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content_type = mimetypes.guess_type(clip_path.name)[0] or "application/octet-stream"
    headers = {
        "Content-Type": content_type,
        "X-Original-Filename": clip_path.name,
        "X-Clip-Consent": "operator-smoke-test",
    }
    if runner_prompt:
        headers["X-Runner-Prompt"] = json.dumps(runner_prompt, separators=(",", ":"))
    return _json_request(
        f"{api_base_url.rstrip('/')}/v1/uploads",
        method="POST",
        body=clip_path.read_bytes(),
        headers=headers,
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


def _download_file(url: str, target_path: Path, *, timeout: int = 180) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response, target_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def _download_artifacts(final: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    artifacts = final.get("artifacts") or {}
    downloaded: dict[str, Path] = {}
    if not isinstance(artifacts, dict):
        return downloaded
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not record.get("href"):
            continue
        artifact_name = Path(str(name)).name
        target_path = out_dir / artifact_name
        _download_file(str(record["href"]), target_path)
        downloaded[artifact_name] = target_path
    return downloaded


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _assert_manifest_quality(
    manifest: dict[str, Any],
    *,
    require_no_fallback: bool = False,
    require_seed_source: str | None = None,
    min_nonempty_frame_rate: float = 0.0,
    min_identity_accepted_rate: float = 0.0,
) -> None:
    mask_stage = _nested_get(manifest, ("stages", "whole_runner_mask")) or {}
    if not isinstance(mask_stage, dict):
        raise RuntimeError("Manifest is missing stages.whole_runner_mask.")

    fallback = mask_stage.get("fallback")
    if require_no_fallback and fallback:
        raise RuntimeError("Mask stage used fallback:\n" + json.dumps(fallback, indent=2))

    if require_seed_source:
        seed_source = _nested_get(mask_stage, ("prompting", "sam31", "seed_source"))
        if seed_source != require_seed_source:
            raise RuntimeError(
                f"Expected SAM seed source {require_seed_source!r}; got {seed_source!r}."
            )

    frame_count = _int_value(_nested_get(manifest, ("video", "frame_count")))
    nonempty_frames = _int_value(_nested_get(mask_stage, ("mask_summary", "nonempty_frames")))
    if min_nonempty_frame_rate > 0:
        if frame_count <= 0:
            raise RuntimeError("Manifest video.frame_count is missing or zero.")
        nonempty_rate = nonempty_frames / frame_count
        if nonempty_rate < min_nonempty_frame_rate:
            raise RuntimeError(
                "Mask non-empty frame rate is too low: "
                f"{nonempty_frames}/{frame_count} = {nonempty_rate:.3f}; "
                f"required {min_nonempty_frame_rate:.3f}."
            )

    if min_identity_accepted_rate > 0:
        identity_filter = _nested_get(mask_stage, ("prompting", "identity_filter")) or {}
        if not isinstance(identity_filter, dict) or not identity_filter.get("enabled"):
            raise RuntimeError("Manifest is missing enabled prompting.identity_filter.")
        accepted_frames = _int_value(identity_filter.get("accepted_frames"))
        denominator = frame_count or (
            accepted_frames
            + _int_value(identity_filter.get("rejected_frames"))
            + _int_value(identity_filter.get("unchecked_frames"))
        )
        if denominator <= 0:
            raise RuntimeError("Identity filter frame denominator is missing or zero.")
        accepted_rate = accepted_frames / denominator
        if accepted_rate < min_identity_accepted_rate:
            raise RuntimeError(
                "Identity-filter accepted frame rate is too low: "
                f"{accepted_frames}/{denominator} = {accepted_rate:.3f}; "
                f"required {min_identity_accepted_rate:.3f}."
            )


def _read_video_frame(video_path: Path, frame_index: int) -> Any:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for contact sheet: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _video_frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count


def _label_frame(frame: Any, label: str) -> Any:
    import cv2

    labeled = frame.copy()
    cv2.putText(
        labeled,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _write_contact_sheet(
    *,
    source_video: Path,
    artifacts: dict[str, Path],
    output_path: Path,
    columns: int = 6,
) -> None:
    import cv2
    import numpy as np

    rows = [
        ("source", source_video),
        ("qa", artifacts.get("qa_overlay.mp4")),
        ("masked", artifacts.get("masked_runner.mp4")),
        ("fused", artifacts.get("fused_overlay.mp4")),
    ]
    rows = [(label, path) for label, path in rows if path and path.is_file()]
    if not rows:
        raise RuntimeError("No videos available for contact sheet.")

    source_frame_count = _video_frame_count(source_video)
    if source_frame_count <= 0:
        raise RuntimeError(f"Could not inspect source video: {source_video}")
    sample_indices = np.linspace(
        max(0, int(source_frame_count * 0.06)),
        max(0, int(source_frame_count * 0.90)),
        num=columns,
        dtype=int,
    ).tolist()
    tile_width = 240
    tile_height = 135
    row_images = []
    for label, video_path in rows:
        tiles = []
        frame_count = max(1, _video_frame_count(video_path))
        for source_index in sample_indices:
            frame_index = min(frame_count - 1, source_index)
            frame = _read_video_frame(video_path, frame_index)
            if frame is None:
                frame = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            tiles.append(_label_frame(frame, f"{label} {frame_index}"))
        row_images.append(np.hstack(tiles))
    sheet = np.vstack(row_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


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
    parser.add_argument(
        "--prompt-json",
        type=Path,
        default=None,
        help="Optional runner prompt JSON file to send as the X-Runner-Prompt header.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory where completed artifacts and job JSON are downloaded.",
    )
    parser.add_argument(
        "--download-artifacts",
        action="store_true",
        help="Download all public artifacts after the job completes.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Write contact-sheet.jpg from source, QA, masked, and fused videos. Implies --download-artifacts.",
    )
    parser.add_argument(
        "--require-no-fallback",
        action="store_true",
        help="Fail if cv_run_manifest.json reports an identity-box mask fallback.",
    )
    parser.add_argument(
        "--require-sam-seed-source",
        default="",
        help="Fail unless SAM prompting reports this seed_source.",
    )
    parser.add_argument("--min-nonempty-frame-rate", type=float, default=0.0)
    parser.add_argument("--min-identity-accepted-rate", type=float, default=0.0)
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

    runner_prompt = _read_json_file(args.prompt_json) if args.prompt_json else None
    upload = _upload_clip(args.api_base_url, clip_path, runner_prompt=runner_prompt)
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

    if args.out_dir:
        run_out_dir = args.out_dir.expanduser().resolve() / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)
        (run_out_dir / "job.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        downloaded: dict[str, Path] = {}
        if args.download_artifacts or args.contact_sheet:
            downloaded = _download_artifacts(final, run_out_dir)
            print(
                "Downloaded artifacts:",
                json.dumps({name: str(path) for name, path in downloaded.items()}, indent=2),
                flush=True,
            )
        manifest_path = downloaded.get("cv_run_manifest.json")
        if manifest_path and manifest_path.is_file():
            manifest = _read_json_file(manifest_path)
            _assert_manifest_quality(
                manifest,
                require_no_fallback=args.require_no_fallback,
                require_seed_source=args.require_sam_seed_source or None,
                min_nonempty_frame_rate=args.min_nonempty_frame_rate,
                min_identity_accepted_rate=args.min_identity_accepted_rate,
            )
            print("Manifest validation passed.", flush=True)
        elif (
            args.require_no_fallback
            or args.require_sam_seed_source
            or args.min_nonempty_frame_rate > 0
            or args.min_identity_accepted_rate > 0
        ):
            print("cv_run_manifest.json was not downloaded.", file=sys.stderr)
            return 1
        if args.contact_sheet:
            _write_contact_sheet(
                source_video=clip_path,
                artifacts=downloaded,
                output_path=run_out_dir / "contact-sheet.jpg",
            )
            print(f"Contact sheet: {run_out_dir / 'contact-sheet.jpg'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
