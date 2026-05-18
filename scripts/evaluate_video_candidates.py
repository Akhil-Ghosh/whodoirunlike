#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whodoirunlike.video_eval import write_cv_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and CV-score top candidate videos.")
    parser.add_argument(
        "--scored-csv",
        type=Path,
        default=Path("artifacts/discovery/candidates.scored.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluation/video_candidates.cv.csv"),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=900,
        help="Skip longer sources for automatic CV scoring. Use 0 to disable.",
    )
    parser.add_argument(
        "--recommendation",
        action="append",
        dest="recommendations",
        default=["review_first"],
        help="Metadata recommendation bucket to include. Repeatable.",
    )
    parser.add_argument("--download-dir", type=Path, default=Path("clips/raw/candidates"))
    parser.add_argument(
        "--max-height",
        type=int,
        default=360,
        help="Maximum download height for CV triage videos. Use 720 for human review quality.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/mediapipe"))
    parser.add_argument("--model-variant", choices=["lite", "full", "heavy"], default="lite")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = write_cv_evaluation(
        args.scored_csv,
        args.out,
        limit=args.limit,
        sample_count=args.sample_count,
        recommendations=set(args.recommendations),
        download_dir=args.download_dir,
        model_dir=args.model_dir,
        model_variant=args.model_variant,
        max_duration_seconds=args.max_duration_seconds or None,
        max_height=args.max_height,
        force_download=args.force_download,
    )
    Console().print(f"[green]CV-scored {count} candidates[/green] into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
