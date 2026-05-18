#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whodoirunlike.evaluation import evaluate_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score discovered video candidates for review.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = evaluate_candidates(args.jsonl, args.csv)
    Console().print(f"[green]Scored {count} candidates[/green] into {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

