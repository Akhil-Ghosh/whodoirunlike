from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_SOURCES = {
    "src/whodoirunlike/sam31_parity.py",
    "src/whodoirunlike/sam31_benchmark.py",
    "src/whodoirunlike/pipeline_parity.py",
    "src/whodoirunlike/sam31_benchmark_serverless.py",
}


@pytest.mark.parametrize(
    ("dockerfile_name", "expected_from", "image_role", "base_commit"),
    [
        (
            "Dockerfile.runpod.benchmark",
            "ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor@"
            "sha256:52392c71b9a44d804f0ba7fcd894247988319dbe98e6160cb05d70a13894714a",
            "candidate",
            "657d36588f3ca073554bfd40071ff747e0e750bb",
        ),
        (
            "Dockerfile.runpod.benchmark.control",
            "ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor@"
            "sha256:a137045767045edb502f1fcce14f5e113ce3c92b2a376bfa2f04e906cea7e73b",
            "control",
            "8b33d07cd129cb6878f4630af133bf30c291914b",
        ),
    ],
)
def test_benchmark_images_are_exact_additive_overlays(
    dockerfile_name: str,
    expected_from: str,
    image_role: str,
    base_commit: str,
) -> None:
    text = (REPO_ROOT / dockerfile_name).read_text(encoding="utf-8")
    instructions = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    from_lines = [line for line in instructions if line.startswith("FROM ")]
    copy_lines = [line for line in instructions if line.startswith("COPY ")]

    assert from_lines == [f"FROM {expected_from}"]
    assert {line.split()[1] for line in copy_lines} == OVERLAY_SOURCES
    assert len(copy_lines) == len(OVERLAY_SOURCES)
    assert not any(line.startswith(("RUN ", "ADD ")) for line in instructions)
    assert "WHODOIRUNLIKE_PROCESSOR_VERSION" not in text
    assert f"WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE={image_role}" in text
    assert f"WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT={base_commit}" in text
    assert "WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT=true" in text


def test_overlay_uses_only_public_production_entrypoints() -> None:
    mask_source = (REPO_ROOT / "src/whodoirunlike/sam31_benchmark.py").read_text(encoding="utf-8")
    pipeline_source = (REPO_ROOT / "src/whodoirunlike/pipeline_parity.py").read_text(
        encoding="utf-8"
    )

    assert "from whodoirunlike.sam31_gpu_runner import run_sam31_gpu_mask" in mask_source
    assert "from whodoirunlike.full_pipeline import run_full_cv_pipeline" in pipeline_source
    for private_name in (
        "_collect_sam31_masks",
        "_configure_interactive_tracker_for_user_prompt",
        "_filter_masks_to_track_boxes",
        "_load_identity_track_boxes",
        "_patch_multiplex_init_state_kwargs",
        "_synchronize_cuda",
        "_densepose_runtime_kwargs",
    ):
        assert private_name not in mask_source
        assert private_name not in pipeline_source
