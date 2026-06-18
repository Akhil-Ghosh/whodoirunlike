from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from whodoirunlike.cv_flow import utc_now_iso
from whodoirunlike.form_features import compile_form_features
from whodoirunlike.hosted_processor import router as hosted_processor_router
from whodoirunlike.pose_runner import process_pose_video, update_manifest_after_pose
from whodoirunlike.sam2_runner import inspect_video, write_json
from whodoirunlike.video_eval import POSE_MODEL_URLS, ensure_pose_model


DEFAULT_ARTIFACT_ROOT = Path(os.getenv("WHODOIRUNLIKE_API_ARTIFACT_ROOT", "artifacts/api_runs"))
DEFAULT_MODEL_DIR = Path(os.getenv("WHODOIRUNLIKE_MODEL_DIR", "models/mediapipe"))
DEFAULT_MAX_BYTES = int(os.getenv("WHODOIRUNLIKE_MAX_UPLOAD_BYTES", str(75 * 1024 * 1024)))
DEFAULT_MAX_SECONDS = float(os.getenv("WHODOIRUNLIKE_MAX_DURATION_SECONDS", "20"))
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:4173",
    "http://localhost:4173",
)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v"}


class ArtifactLinks(BaseModel):
    source: str
    skeleton_render: str
    qa_overlay: str
    pose_landmarks: str
    features: str
    form_features: str


class ClipProcessResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    elapsed_seconds: float
    model_variant: str
    video: dict[str, Any]
    quality: dict[str, Any]
    explainability_metrics: dict[str, Any]
    summary_features: dict[str, Any]
    artifacts: ArtifactLinks


def _artifact_root() -> Path:
    root = DEFAULT_ARTIFACT_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cors_origins() -> list[str]:
    raw = os.getenv("WHODOIRUNLIKE_CORS_ORIGINS")
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    if raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _extension_for_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in ALLOWED_EXTENSIONS:
        return suffix
    content_type = (file.content_type or "").lower()
    if content_type == "video/mp4":
        return ".mp4"
    if content_type in {"video/quicktime", "video/mov"}:
        return ".mov"
    if content_type == "video/webm":
        return ".webm"
    raise HTTPException(
        status_code=415,
        detail="Upload an MP4, MOV, M4V, or WebM running clip.",
    )


async def _save_upload(upload: UploadFile, target: Path, max_bytes: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with target.open("wb") as f:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Clip is too large. Max upload size is {max_bytes // (1024 * 1024)} MB.",
                    )
                f.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return written


def _blank_prompt() -> dict[str, Any]:
    return {
        "version": 1,
        "selection": {
            "type": "auto",
            "positive_points": [],
            "negative_points": [],
            "box": None,
        },
        "frame": {"frame_index": 0},
    }


def _artifact_url(request: Request, artifact_root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(artifact_root).as_posix()
    return str(request.url_for("artifacts", path=relative))


def _write_initial_manifest(
    *,
    run_dir: Path,
    run_id: str,
    source_path: Path,
    prompt_path: Path,
    pose_landmarks_path: Path,
    skeleton_render_path: Path,
    qa_overlay_path: Path,
    features_path: Path,
    form_features_path: Path,
    form_feature_arrays_path: Path,
    upload: UploadFile,
    size_bytes: int,
    video_meta: dict[str, Any],
) -> Path:
    fps = float(video_meta.get("fps") or 0.0)
    frame_count = int(video_meta.get("frame_count") or 0)
    duration_seconds = round(frame_count / fps, 3) if fps else None
    manifest_path = run_dir / "cv_run_manifest.json"
    write_json(
        manifest_path,
        {
            "version": 1,
            "candidate_id": run_id,
            "runner_name": "Uploaded clip",
            "runner_slug": "uploaded-clip",
            "created_at": utc_now_iso(),
            "source": {
                "platform": "api_upload",
                "filename": upload.filename,
                "content_type": upload.content_type,
                "size_bytes": size_bytes,
                "video_path": str(source_path),
            },
            "review": {
                "quality": "api_upload",
                "camera_angle": "unknown",
                "primary_bucket": "running",
                "duration_seconds": duration_seconds,
            },
            "paths": {
                "source_segment": str(source_path),
                "person_prompt": str(prompt_path),
                "pose_landmarks": str(pose_landmarks_path),
                "skeleton_render": str(skeleton_render_path),
                "qa_overlay": str(qa_overlay_path),
                "features": str(features_path),
                "form_features": str(form_features_path),
                "form_feature_arrays": str(form_feature_arrays_path),
            },
            "stages": {
                "upload": {"status": "complete", "output": str(source_path)},
                "pose": {"status": "pending"},
                "renders": {"status": "pending"},
                "features": {"status": "pending"},
                "form_features": {"status": "pending"},
            },
        },
    )
    return manifest_path


def _process_clip(
    *,
    run_dir: Path,
    model_variant: str,
    manifest_path: Path,
    source_path: Path,
    prompt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = ensure_pose_model(DEFAULT_MODEL_DIR, model_variant)
    paths = {
        "pose_landmarks": run_dir / "pose_landmarks.jsonl",
        "skeleton_render": run_dir / "skeleton_render.mp4",
        "qa_overlay": run_dir / "qa_overlay.mp4",
        "features": run_dir / "features.json",
    }
    summary = process_pose_video(
        input_video=source_path,
        source_video=source_path,
        mask_video=None,
        prompt=prompt,
        pose_landmarks_path=paths["pose_landmarks"],
        skeleton_render_path=paths["skeleton_render"],
        qa_overlay_path=paths["qa_overlay"],
        features_path=paths["features"],
        model_path=model_path,
        model_variant=model_variant,
    )
    update_manifest_after_pose(
        manifest_path,
        pose_landmarks_path=paths["pose_landmarks"],
        skeleton_render_path=paths["skeleton_render"],
        features_path=paths["features"],
        result=summary,
    )
    form_feature_result = compile_form_features(run_dir=run_dir)
    return summary, form_feature_result


def create_app() -> FastAPI:
    artifact_root = _artifact_root()
    app = FastAPI(
        title="Who Do I Run Like API",
        version="0.1.0",
        description="Clip-processing API for running-form pose extraction and feature artifacts.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.mount("/artifacts", StaticFiles(directory=artifact_root), name="artifacts")
    app.include_router(hosted_processor_router)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "whodoirunlike-api",
            "model_variants": sorted(POSE_MODEL_URLS),
            "artifact_root": str(artifact_root),
        }

    @app.post("/v1/clips", response_model=ClipProcessResponse)
    async def process_clip(
        request: Request,
        file: UploadFile = File(...),
        model_variant: str = Form("lite"),
    ) -> ClipProcessResponse:
        if model_variant not in POSE_MODEL_URLS:
            raise HTTPException(
                status_code=400,
                detail=f"model_variant must be one of: {', '.join(sorted(POSE_MODEL_URLS))}",
            )

        run_id = uuid4().hex[:16]
        run_dir = artifact_root / run_id
        extension = _extension_for_upload(file)
        source_path = run_dir / f"source_segment{extension}"
        created_at = utc_now_iso()
        started_at = time.monotonic()

        size_bytes = await _save_upload(file, source_path, DEFAULT_MAX_BYTES)
        try:
            video_meta = inspect_video(source_path)
        except Exception as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Could not read uploaded video: {exc}") from exc

        fps = float(video_meta.get("fps") or 0.0)
        frame_count = int(video_meta.get("frame_count") or 0)
        duration_seconds = frame_count / fps if fps else 0.0
        if duration_seconds > DEFAULT_MAX_SECONDS:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(
                status_code=413,
                detail=f"Clip is {duration_seconds:.1f}s. Submit a clip under {DEFAULT_MAX_SECONDS:.0f}s.",
            )

        prompt = _blank_prompt()
        prompt_path = run_dir / "person_prompt.json"
        write_json(prompt_path, prompt)
        manifest_path = _write_initial_manifest(
            run_dir=run_dir,
            run_id=run_id,
            source_path=source_path,
            prompt_path=prompt_path,
            pose_landmarks_path=run_dir / "pose_landmarks.jsonl",
            skeleton_render_path=run_dir / "skeleton_render.mp4",
            qa_overlay_path=run_dir / "qa_overlay.mp4",
            features_path=run_dir / "features.json",
            form_features_path=run_dir / "form_features.json",
            form_feature_arrays_path=run_dir / "form_features.npz",
            upload=file,
            size_bytes=size_bytes,
            video_meta=video_meta,
        )

        try:
            summary, form_features = await run_in_threadpool(
                _process_clip,
                run_dir=run_dir,
                model_variant=model_variant,
                manifest_path=manifest_path,
                source_path=source_path,
                prompt=prompt,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Clip processing failed: {exc}") from exc

        return ClipProcessResponse(
            run_id=run_id,
            status="complete",
            created_at=created_at,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
            model_variant=model_variant,
            video={
                **video_meta,
                "duration_seconds": round(duration_seconds, 3),
                "size_bytes": size_bytes,
            },
            quality=summary.get("quality", {}),
            explainability_metrics=summary.get("explainability_metrics", {}),
            summary_features=form_features.get("summary_features", {}),
            artifacts=ArtifactLinks(
                source=_artifact_url(request, artifact_root, source_path),
                skeleton_render=_artifact_url(request, artifact_root, run_dir / "skeleton_render.mp4"),
                qa_overlay=_artifact_url(request, artifact_root, run_dir / "qa_overlay.mp4"),
                pose_landmarks=_artifact_url(request, artifact_root, run_dir / "pose_landmarks.jsonl"),
                features=_artifact_url(request, artifact_root, run_dir / "features.json"),
                form_features=_artifact_url(request, artifact_root, run_dir / "form_features.json"),
            ),
        )

    return app


app = create_app()
