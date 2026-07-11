from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM31_REVISION = "5dd401d1c5c1d5c3eedff06d41b77af824517619"


def test_runpod_image_pins_and_patches_tested_sam31_revision() -> None:
    dockerfile = (ROOT / "Dockerfile.runpod").read_text(encoding="utf-8")
    patch = (
        ROOT / "patches/sam3/0001-preserve-refined-mask-on-uncached-frame.patch"
    ).read_text(encoding="utf-8")

    assert SAM31_REVISION in dockerfile
    assert "git -C /opt/sam3 apply --check /tmp/sam3.patch" in dockerfile
    assert "git+https://github.com/facebookresearch/sam3.git" not in dockerfile
    assert 'cached_frame_outputs"].get(frame_idx, {})' in patch
    assert "\n+            return {}" not in patch
