#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.active_learning import build_uncertainty_queue
from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an active-learning review queue from CV QC metrics.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=Path("artifacts/active_learning/uncertainty_queue.json"))
    args = parser.parse_args()
    result = build_uncertainty_queue(args.run_root, args.out)
    Console().print_json(json.dumps({"out": str(args.out), "entry_count": result["entry_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
