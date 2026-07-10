from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.running_clip_run import RunningClipRun


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("Parquet export needs pyarrow. Install with: python -m pip install pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)


def export_cv_tables(run_dir: Path) -> dict[str, Any]:
    run = RunningClipRun(run_dir)
    manifest = run.read_manifest()
    exports: dict[str, dict[str, Any]] = {}
    mappings = {
        "poses": ("pose_landmarks", "poses"),
        "densepose_parquet": ("densepose", "densepose_parquet"),
        "fused_form_parquet": ("fused_form", "fused_form_parquet"),
    }
    for output_key, (input_key, manifest_output_key) in mappings.items():
        input_path = run.artifact_path(input_key, manifest)
        output_path = run.artifact_path(manifest_output_key, manifest)
        if not input_path.exists():
            exports[output_key] = {
                "status": "missing_input",
                "input": str(input_path),
                "output": str(output_path),
                "row_count": 0,
            }
            continue
        rows = read_jsonl(input_path)
        row_count = write_parquet(output_path, rows)
        exports[output_key] = {
            "status": "complete",
            "input": str(input_path),
            "output": str(output_path),
            "row_count": row_count,
        }

    manifest["updated_at"] = utc_now_iso()
    run.update_stage(
        "artifact_tables",
        {
            "status": "complete",
            "exports": exports,
            "completed_at": utc_now_iso(),
        },
        manifest,
    )
    return {"candidate_id": manifest.get("candidate_id"), "exports": exports}
