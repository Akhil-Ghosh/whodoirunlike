#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from whodoirunlike.review_app import REPO_ROOT, ReviewAppConfig, load_review_clips
from whodoirunlike.video_eval import download_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download higher-quality videos for human review.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/evaluation/video_candidates.top30.json"),
        help="Evaluated candidate source JSON/CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluation/video_candidates.review20_720.json"),
        help="Output JSON used by the review UI.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Maximum download height. Use 0 for the highest available source format.",
    )
    parser.add_argument("--download-dir", type=Path, default=Path("clips/raw/review_720p"))
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def write_outputs(rows: list[dict[str, Any]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    console = Console()
    config = ReviewAppConfig(
        source_path=args.source,
        annotations_path=Path("artifacts/review/clip_reviews.json"),
        static_dir=Path("review_ui"),
        repo_root=REPO_ROOT,
        limit=args.limit,
    )
    clips = load_review_clips(config)
    output_rows: list[dict[str, Any]] = []
    max_height = args.max_height if args.max_height > 0 else None

    for index, clip in enumerate(clips, start=1):
        console.print(f"[bold]{index}/{len(clips)}[/bold] {clip['runner_name']}: {clip['title'][:80]}")
        row = dict(clip)
        try:
            video_path = download_candidate(
                row,
                args.download_dir,
                force=args.force_download,
                max_height=max_height,
            )
            row["video_path"] = str(video_path)
            row["review_video_max_height"] = max_height or 0
            row["review_video_quality"] = "best" if max_height is None else f"{max_height}p"
            row["review_file_size_mb"] = round(video_path.stat().st_size / 1_000_000, 2)
            row["review_download_error"] = ""
        except Exception as exc:
            row["review_download_error"] = str(exc)
        output_rows.append(row)

    write_outputs(output_rows, args.out)
    console.print(f"[green]Wrote {len(output_rows)} review videos[/green] to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
