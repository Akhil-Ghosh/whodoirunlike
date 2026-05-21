#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.artifact_tables import export_cv_tables
from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Export JSONL CV artifacts to Parquet tables.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    args = parser.parse_args()
    result = export_cv_tables(args.run_root / args.candidate_id)
    Console().print_json(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
