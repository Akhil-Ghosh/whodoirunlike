from __future__ import annotations

from typing import Any


def test_runpod_handler_health(monkeypatch: Any) -> None:
    from whodoirunlike import runpod_serverless

    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET", "secret")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("WHODOIRUNLIKE_MASK_BACKEND", "sam31_gpu")

    response = runpod_serverless.handler({"input": {"type": "health"}})

    assert response["status"] == "ok"
    assert response["health"]["has_processor_secret"] is True
    assert response["health"]["has_hf_token"] is True
    assert response["health"]["mask_backend"] == "sam31_gpu"


def test_runpod_handler_deep_health(monkeypatch: Any) -> None:
    from whodoirunlike import runpod_serverless

    monkeypatch.setattr(
        runpod_serverless,
        "processor_readiness",
        lambda: {"ready_for_full_pipeline": True, "mask_backend": "sam31_gpu"},
    )

    response = runpod_serverless.handler({"input": {"type": "health", "level": "deep"}})

    assert response["status"] == "ok"
    assert response["readiness"]["mask_backend"] == "sam31_gpu"


def test_runpod_handler_processes_worker_payload(monkeypatch: Any) -> None:
    from whodoirunlike import runpod_serverless

    captured: dict[str, Any] = {}

    def fake_process(payload: Any, *, raise_on_error: bool) -> dict[str, Any]:
        captured["run_id"] = payload.run_id
        captured["raise_on_error"] = raise_on_error
        return {"status": "complete", "run_id": payload.run_id}

    monkeypatch.setattr(runpod_serverless, "process_hosted_job", fake_process)

    response = runpod_serverless.handler(
        {
            "input": {
                "run_id": "12345678-1234-4234-9234-123456789abc",
                "callback_base_url": "https://api.whodoirunlike.com",
                "source": {
                    "url": "https://api.whodoirunlike.com/v1/jobs/12345678-1234-4234-9234-123456789abc/source",
                    "key": "uploads/12345678-1234-4234-9234-123456789abc/source.mp4",
                    "filename": "clip.mp4",
                    "content_type": "video/mp4",
                    "size_bytes": 123,
                },
            }
        }
    )

    assert response == {"status": "complete", "run_id": "12345678-1234-4234-9234-123456789abc"}
    assert captured == {
        "run_id": "12345678-1234-4234-9234-123456789abc",
        "raise_on_error": True,
    }
