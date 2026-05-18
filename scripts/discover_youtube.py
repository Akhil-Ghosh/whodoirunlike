#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

from whodoirunlike.discovery import build_queries, load_runners, normalize_yt_entry, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover YouTube candidate videos for review.")
    parser.add_argument("--runner-data", type=Path, default=Path("data/runners.yml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/discovery/candidates.jsonl"))
    parser.add_argument("--limit-per-query", type=int, default=5)
    parser.add_argument("--max-runners", type=int)
    parser.add_argument("--max-queries-per-runner", type=int, default=8)
    parser.add_argument("--runner", action="append", dest="runner_slugs", help="Runner slug to include.")
    parser.add_argument("--bucket", choices=["800_1500", "5k_10k", "marathon"])
    parser.add_argument(
        "--dry-run-queries",
        action="store_true",
        help="Print queries without calling YouTube.",
    )
    return parser.parse_args()


def import_ytdlp():
    try:
        import yt_dlp
    except ImportError:
        print(
            "Missing yt-dlp. Run: python -m pip install -e \".[dev]\"",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return yt_dlp


def main() -> int:
    args = parse_args()
    console = Console()
    runners = load_runners(args.runner_data)

    if args.runner_slugs:
        wanted = set(args.runner_slugs)
        runners = [runner for runner in runners if runner.slug in wanted]
    if args.bucket:
        runners = [runner for runner in runners if runner.primary_bucket == args.bucket]
    if args.max_runners:
        runners = runners[: args.max_runners]

    if args.dry_run_queries:
        for runner in runners:
            console.print(f"[bold]{runner.name}[/bold] ({runner.primary_bucket})")
            for query in build_queries(runner, args.max_queries_per_runner):
                console.print(f"  - {query}")
        return 0

    yt_dlp = import_ytdlp()
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "noplaylist": True,
    }

    rows = []
    seen_video_ids: set[tuple[str, str]] = set()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for runner in track(runners, description="Discovering candidates"):
            queries = build_queries(runner, args.max_queries_per_runner)
            for query in queries:
                search = f"ytsearch{args.limit_per_query}:{query}"
                try:
                    result = ydl.extract_info(search, download=False)
                except Exception as exc:
                    console.print(f"[yellow]Search failed[/yellow] {query}: {exc}")
                    continue

                for entry in result.get("entries") or []:
                    if not entry:
                        continue
                    video_id = entry.get("id") or entry.get("display_id") or ""
                    dedupe_key = (runner.slug, video_id)
                    if not video_id or dedupe_key in seen_video_ids:
                        continue
                    seen_video_ids.add(dedupe_key)
                    rows.append(normalize_yt_entry(runner, query, entry))

    count = write_jsonl(args.out, rows)
    console.print(f"[green]Wrote {count} candidates[/green] to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

