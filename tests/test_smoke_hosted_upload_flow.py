from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_smoke_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/smoke_hosted_upload_flow.py"
    spec = importlib.util.spec_from_file_location("smoke_hosted_upload_flow", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_quality_accepts_track_seeded_sam_without_fallback() -> None:
    smoke = _load_smoke_module()
    manifest = {
        "video": {"frame_count": 100},
        "stages": {
            "whole_runner_mask": {
                "mask_summary": {"nonempty_frames": 96},
                "prompting": {
                    "sam31": {"seed_source": "target_track_first_visible_frame"},
                    "identity_filter": {
                        "enabled": True,
                        "accepted_frames": 94,
                        "rejected_frames": 0,
                        "unchecked_frames": 6,
                    },
                },
            }
        },
    }

    smoke._assert_manifest_quality(
        manifest,
        require_no_fallback=True,
        require_seed_source="target_track_first_visible_frame",
        min_nonempty_frame_rate=0.9,
        min_identity_accepted_rate=0.9,
    )


def test_manifest_quality_rejects_box_fallback() -> None:
    smoke = _load_smoke_module()
    manifest = {
        "video": {"frame_count": 100},
        "stages": {
            "whole_runner_mask": {
                "fallback": {"reason": "sam31_gpu_sparse_or_off_target_mask"},
                "mask_summary": {"nonempty_frames": 100},
            }
        },
    }

    with pytest.raises(RuntimeError, match="used fallback"):
        smoke._assert_manifest_quality(manifest, require_no_fallback=True)


def test_manifest_quality_requires_track_seed_source() -> None:
    smoke = _load_smoke_module()
    manifest = {
        "video": {"frame_count": 100},
        "stages": {
            "whole_runner_mask": {
                "mask_summary": {"nonempty_frames": 100},
                "prompting": {"sam31": {"seed_source": "user_prompt"}},
            }
        },
    }

    with pytest.raises(RuntimeError, match="Expected SAM seed source"):
        smoke._assert_manifest_quality(
            manifest,
            require_seed_source="target_track_first_visible_frame",
        )
