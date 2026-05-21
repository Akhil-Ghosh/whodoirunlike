#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.multiview import write_cross_view_match


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a cross-view or cross-clip association.")
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-b", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=Path("artifacts/multiview/match.json"))
    parser.add_argument("--synchronized", action="store_true")
    parser.add_argument("--temporal-offset-seconds", type=float)
    parser.add_argument("--reprojection-error-px", type=float)
    args = parser.parse_args()
    result = write_cross_view_match(
        args.run_root / args.candidate_a,
        args.run_root / args.candidate_b,
        args.out,
        synchronized=args.synchronized,
        temporal_offset_seconds=args.temporal_offset_seconds,
        reprojection_error_px=args.reprojection_error_px,
    )
    Console().print_json(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
