#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whodoirunlike.curation import (
    DEFAULT_SAMPLE_FPS,
    DEFAULT_STEP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    propose_clip_windows,
    write_curation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propose ranked runner clip windows from long source videos."
    )
    parser.add_argument("videos", nargs="+", type=Path, help="Local source video path(s).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/curation/clip_windows.json"),
        help="Output JSON manifest.",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument(
        "--view-bucket",
        choices=["side", "diagonal", "front", "rear", "mixed", "unknown"],
        default="unknown",
        help="Known/reviewed coarse angle bucket, if available.",
    )
    parser.add_argument(
        "--no-scenedetect",
        action="store_true",
        help="Treat each input as one shot instead of running PySceneDetect.",
    )
    parser.add_argument(
        "--write-thumbnails",
        action="store_true",
        help="Write a midpoint thumbnail for each selected window.",
    )
    parser.add_argument(
        "--write-previews",
        action="store_true",
        help="Write small MP4 previews for each selected window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    output_dir = args.out.parent
    manifests = []

    for video_path in args.videos:
        manifest = propose_clip_windows(
            video_path,
            top_k=args.top_k,
            view_bucket=args.view_bucket,
            use_scenedetect=not args.no_scenedetect,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            sample_fps=args.sample_fps,
            output_dir=output_dir,
            write_thumbnails=args.write_thumbnails,
            write_previews=args.write_previews,
        )
        manifests.append(manifest)
        console.print(
            f"[green]ranked {len(manifest['windows'])} windows[/green] "
            f"from {video_path} ({manifest['windows_considered']} considered)"
        )

    payload = write_curation_manifest(args.out, manifests)
    console.print(f"[green]wrote[/green] {args.out} with {len(payload['windows'])} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
