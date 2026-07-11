from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM31_REVISION = "5dd401d1c5c1d5c3eedff06d41b77af824517619"
BOXMOT_VERSION = "21.0.0"
RTMLIB_VERSION = "0.0.15"


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


def test_runpod_image_pins_compatible_boxmot_and_checks_tracker_import() -> None:
    dockerfile = (ROOT / "Dockerfile.runpod").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert f'"boxmot=={BOXMOT_VERSION}"' in dockerfile
    assert f'"boxmot=={BOXMOT_VERSION}"' in pyproject
    assert "identity_setup_status" in dockerfile


def test_runpod_image_preloads_verified_rtmlib_models() -> None:
    dockerfile = (ROOT / "Dockerfile.runpod").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    smoke_workflow = (ROOT / ".github/workflows/smoke-runpod-image.yml").read_text(
        encoding="utf-8"
    )

    assert "TORCH_HOME=/opt/rtmlib-cache" in dockerfile
    assert "python scripts/preload_rtmlib_models.py" in dockerfile
    assert f'"rtmlib=={RTMLIB_VERSION}"' in dockerfile
    assert f'"rtmlib=={RTMLIB_VERSION}"' in pyproject
    assert "--network none" in smoke_workflow
    assert "build_rtmlib_model" in smoke_workflow
