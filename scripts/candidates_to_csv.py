#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whodoirunlike.discovery import write_candidates_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert candidate JSONL to a review CSV.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = write_candidates_csv(args.jsonl, args.csv)
    Console().print(f"[green]Wrote {count} rows[/green] to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

