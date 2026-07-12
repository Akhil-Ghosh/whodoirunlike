from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_SOURCES = {
    "src/whodoirunlike/sam31_parity.py",
    "src/whodoirunlike/sam31_benchmark.py",
    "src/whodoirunlike/pipeline_parity.py",
    "src/whodoirunlike/sam31_benchmark_serverless.py",
}
def test_final_candidate_parity_image_is_a_dynamic_immutable_harness_only_overlay() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.runpod.benchmark.final-candidate").read_text(
        encoding="utf-8"
    )
    instructions = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    from_lines = [line for line in instructions if line.startswith("FROM ")]
    copy_lines = [line for line in instructions if line.startswith("COPY ")]

    assert instructions[0] == "ARG WHODOIRUNLIKE_FINAL_CANDIDATE_IMAGE"
    assert from_lines == ["FROM ${WHODOIRUNLIKE_FINAL_CANDIDATE_IMAGE}"]
    assert {line.split()[1] for line in copy_lines} == OVERLAY_SOURCES
    assert len(copy_lines) == len(OVERLAY_SOURCES)
    assert not any(line.startswith(("RUN ", "ADD ")) for line in instructions)
    assert "WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE=final_candidate" in dockerfile
    assert "WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT=true" in dockerfile
    assert "WHODOIRUNLIKE_CODE_OVERLAY_SOURCE=base_image" in dockerfile
    assert "WHODOIRUNLIKE_MASK_BACKEND=sam31_gpu" in dockerfile
    assert "MMPOSE_DEVICE=cpu" in dockerfile
    assert "MMPOSE_USE_DETECTOR=" not in dockerfile
    assert "DENSEPOSE_TARGET_CROP_ENABLED=" not in dockerfile


def test_final_candidate_workflow_is_manual_scratch_only_and_fails_closed() -> None:
    workflow = (REPO_ROOT / ".github/workflows/build-runpod-final-parity.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "candidate_commit:" in workflow
    assert "candidate_image_digest:" in workflow
    assert 're.fullmatch(r"[0-9a-f]{40}", commit)' in workflow
    assert 're.fullmatch(r"sha256:[0-9a-f]{64}", digest)' in workflow
    assert 'f"{image_name}@{digest}"' in workflow
    assert "FINAL_CANDIDATE_PRODUCTION_FILES" in workflow
    assert "WHODOIRUNLIKE_FINAL_PRODUCTION_SHA256_B64" in workflow
    assert "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER" in workflow
    assert "/opt/densepose-weights/model_final_162be9.pkl" in workflow
    assert "b8a7382001b16e453bad95ca9dbc68ae8f2b839b304cf90eaf5c27fbdb4dae91" in workflow
    assert "Dockerfile.runpod.benchmark.final-candidate" in workflow
    assert "sam31-final-parity-${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert "production_final_candidate" in workflow
    assert "base_contract\"][\"expected_sha256\"]" in workflow
    assert "deploy" not in workflow.lower()
    assert ":latest" not in workflow
    assert "runpodctl" not in workflow.lower()
    assert "api.runpod.ai" not in workflow.lower()


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


def test_parity_handoff_has_no_production_or_staging_sink_default() -> None:
    pipeline_source = (REPO_ROOT / "src/whodoirunlike/pipeline_parity.py").read_text(
        encoding="utf-8"
    )
    cli_source = (REPO_ROOT / "scripts/run_sam31_speed_lab.py").read_text(encoding="utf-8")

    assert "WHODOIRUNLIKE_PARITY_SINK_ORIGIN" in pipeline_source
    assert 'default="https://api.whodoirunlike.com"' not in cli_source
