from __future__ import annotations

from whodoirunlike.sam31_loader_config import sam31_exact_cv2_loader_settings


def test_exact_cv2_loader_settings_default_safely() -> None:
    settings = sam31_exact_cv2_loader_settings({})

    assert settings.to_dict() == {
        "mode": "upstream",
        "enabled": False,
        "chunk_frames": 8,
        "max_frames": 600,
        "max_destination_bytes": 8 * 1024**3,
        "required_concurrency": 1,
        "configured_concurrency": 1,
        "concurrency_ready": True,
    }


def test_exact_cv2_loader_settings_clamp_negative_and_invalid_values() -> None:
    settings = sam31_exact_cv2_loader_settings(
        {
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER": "true",
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES": "-3",
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_FRAMES": "-10",
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_DESTINATION_BYTES": "invalid",
            "WHODOIRUNLIKE_PROCESSOR_CONCURRENCY": "999",
        }
    )

    assert settings.enabled is True
    assert settings.chunk_frames == 1
    assert settings.max_frames == 1
    assert settings.max_destination_bytes == 8 * 1024**3
    assert settings.configured_concurrency == 64
    assert settings.concurrency_ready is False


def test_exact_cv2_loader_settings_clamp_values_to_absolute_caps() -> None:
    settings = sam31_exact_cv2_loader_settings(
        {
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES": "999999",
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_FRAMES": "999999",
            "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_DESTINATION_BYTES": str(
                999999 * 1024**3
            ),
        }
    )

    assert settings.chunk_frames == 64
    assert settings.max_frames == 3600
    assert settings.max_destination_bytes == 32 * 1024**3
