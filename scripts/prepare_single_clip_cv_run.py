#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_CV_RUN_ROOT,
    DEFAULT_REVIEW_MANIFEST,
    prepare_single_clip_cv_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one reviewed clip for the CV pipeline.")
    parser.add_argument("--candidate-id", help="Reviewed candidate id. Defaults to first clip by quality.")
    parser.add_argument("--quality", default="good", help="Quality bucket to use when candidate id is omitted.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite generated segment/frame/prompt files.")
    args = parser.parse_args()

    manifest = prepare_single_clip_cv_run(
        candidate_id=args.candidate_id,
        quality=args.quality,
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        output_root=args.out_root,
        force=args.force,
    )
    run_dir = Path(manifest["paths"]["source_segment"]).parent
    Console().print_json(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "candidate_id": manifest["candidate_id"],
                "runner_name": manifest["runner_name"],
                "camera_angle": manifest["review"]["camera_angle"],
                "source_segment": manifest["paths"]["source_segment"],
                "prompt_frame": manifest["paths"]["prompt_frame"],
                "person_prompt": manifest["paths"]["person_prompt"],
                "track_seed": manifest["paths"]["track_seed"],
                "view_bucket": manifest["paths"]["view_bucket"],
                "manifest": str(run_dir / "cv_run_manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
