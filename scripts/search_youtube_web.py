#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whodoirunlike.web_search import (
    make_search_row,
    search_with_camoufox,
    search_with_scrapling,
    write_search_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search YouTube via Scrapling and/or Camoufox.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--runner-slug")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--backend", choices=["scrapling", "camoufox", "both"], default="both")
    parser.add_argument("--out", type=Path, default=Path("artifacts/discovery/web_search.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    if args.backend in {"scrapling", "both"}:
        rows.extend(search_with_scrapling(args.query, args.limit))
    if args.backend in {"camoufox", "both"}:
        rows.extend(search_with_camoufox(args.query, args.limit))

    normalized_rows = [make_search_row(row, args.runner_slug) for row in rows]
    count = write_search_jsonl(args.out, normalized_rows)
    Console().print(f"[green]Wrote {count} web-search rows[/green] to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

