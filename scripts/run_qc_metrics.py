#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.qc import run_qc_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute QC metrics for one CV run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    args = parser.parse_args()
    result = run_qc_metrics(args.run_root / args.candidate_id)
    Console().print_json(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
