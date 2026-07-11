from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM31_REVISION = "5dd401d1c5c1d5c3eedff06d41b77af824517619"
BOXMOT_VERSION = "21.0.0"
RTMLIB_VERSION = "0.0.15"
ULTRALYTICS_VERSION = "8.4.92"
YOLO26N_SEG_SHA256 = "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
RUNPOD_BASE_DIGEST = "sha256:3e874356857adfa3e8faa3fd913b65bd127f77a0fe2e489513e7775e1c1e16b1"
DETECTRON2_REVISION = "02b5c4e295e990042a714712c21dc79b731e8833"


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


def test_runpod_image_pins_base_image_and_detectron2_revision() -> None:
    dockerfile = (ROOT / "Dockerfile.runpod").read_text(encoding="utf-8")

    assert RUNPOD_BASE_DIGEST in dockerfile.splitlines()[0]
    assert f"ARG DETECTRON2_COMMIT={DETECTRON2_REVISION}" in dockerfile
    assert "git -C /opt/detectron2 fetch --depth 1 origin ${DETECTRON2_COMMIT}" in dockerfile
    assert "git clone --depth 1 https://github.com/facebookresearch/detectron2.git" not in dockerfile


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


def test_runpod_image_runs_rtmlib_with_cuda_onnxruntime() -> None:
    dockerfile = (ROOT / "Dockerfile.runpod").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-runpod-processor.txt").read_text(
        encoding="utf-8"
    )

    assert "MMPOSE_DEVICE=cuda" in dockerfile
    assert "RTMW_RUNTIME_BACKEND=onnxruntime" in dockerfile
    assert "onnxruntime-gpu==1.27.0" in requirements
    assert "\nonnxruntime>=1.20" not in requirements
    assert "CUDAExecutionProvider" in dockerfile


def test_runpod_image_pins_and_preloads_yolo26_segmentation_weights() -> None:
    dockerfile = (ROOT / "Dockerfile.runpod").read_text(encoding="utf-8")

    assert f'"ultralytics=={ULTRALYTICS_VERSION}"' in dockerfile
    assert "yolo26n-seg.pt" in dockerfile
    assert YOLO26N_SEG_SHA256 in dockerfile


def test_manual_image_build_is_sha_only_and_cannot_move_latest() -> None:
    workflow = (ROOT / ".github/workflows/build-runpod-processor.yml").read_text(
        encoding="utf-8"
    )

    assert "${{ env.IMAGE_NAME }}:${{ github.sha }}" in workflow
    assert "publish_latest" not in workflow
    assert ":latest" not in workflow
