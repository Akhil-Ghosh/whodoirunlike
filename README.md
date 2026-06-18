# Who Do I Run Like

Running-form computer vision pipeline with a deployed-style API and web demo.

This repo turns a short running clip into reviewable motion artifacts:

- MediaPipe pose landmarks per frame
- skeleton render and QA overlay videos
- pose-quality metrics
- normalized form features for later similarity search

It is intentionally scoped as an engineering project, not a coaching product. The matching layer is in progress; the current shipped path is clip ingestion, model inference, artifact generation, and feature compilation.

## Why This Exists

The product idea is simple: upload a running clip and eventually compare your form to a reference library of elite runners.

The engineering work behind that is less simple:

1. find usable running footage
2. isolate the target runner
3. extract pose consistently across frames
4. score artifact quality
5. compile features that can support retrieval and explanation

This repo focuses on those backend and data-pipeline pieces.

## What It Demonstrates

- **Python backend**: FastAPI service that accepts video uploads and returns JSON plus generated artifacts.
- **ML inference**: MediaPipe Pose Landmarker over uploaded running clips.
- **Data pipeline design**: staged artifacts, manifests, JSONL/NPZ feature contracts, review queues, and QC metrics.
- **Computer vision ops**: browser-playable render outputs for debugging model quality.
- **Frontend integration**: Next.js technical-preview page that can call the API.
- **Deployment readiness**: Dockerfile, environment config, and clear local/cloud run commands.
- **Testing**: unit coverage for candidate scoring, pose selection, feature compilation, pipeline contracts, and API behavior.

## Architecture

```text
site/                         Next.js technical-preview UI
src/whodoirunlike/api.py      FastAPI clip-processing API
src/whodoirunlike/pose_runner.py
                              MediaPipe pose inference + skeleton/QA renders
src/whodoirunlike/form_features.py
                              normalized pose arrays and summary features
src/whodoirunlike/*_runner.py optional CV stages for masks, identity, DensePose
scripts/                      CLI entrypoints for offline ingestion/review
schemas/                      JSON contracts for candidates, reviews, artifacts
docs/                         design notes and implementation plans
```

The web API is deliberately lean. It runs the pose path synchronously for short clips and writes artifacts under `artifacts/api_runs/<run_id>/`.

For longer or production workloads, the same contract can move to async jobs: upload object storage, enqueue processing, run workers, then serve artifacts once complete.

## API Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn whodoirunlike.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Process a short clip:

```bash
curl -X POST http://127.0.0.1:8000/v1/clips \
  -F "model_variant=lite" \
  -F "file=@/path/to/running-clip.mp4"
```

Response shape:

```json
{
  "run_id": "abc123...",
  "status": "complete",
  "quality": {
    "pose_hit_rate": 0.92,
    "usable_rate": 0.84
  },
  "summary_features": {
    "stride_rhythm_proxy": 1.5,
    "arm_swing_amplitude": 0.42
  },
  "artifacts": {
    "skeleton_render": "http://127.0.0.1:8000/artifacts/...",
    "qa_overlay": "http://127.0.0.1:8000/artifacts/..."
  }
}
```

The first API call downloads the selected MediaPipe model into `models/mediapipe/`.

## Docker

```bash
docker build -t whodoirunlike-api .
docker run --rm -p 8000:8000 \
  -e WHODOIRUNLIKE_CORS_ORIGINS=http://127.0.0.1:4173 \
  whodoirunlike-api
```

Important environment variables:

```text
WHODOIRUNLIKE_API_ARTIFACT_ROOT=artifacts/api_runs
WHODOIRUNLIKE_MODEL_DIR=models/mediapipe
WHODOIRUNLIKE_MAX_UPLOAD_BYTES=78643200
WHODOIRUNLIKE_MAX_DURATION_SECONDS=20
WHODOIRUNLIKE_CORS_ORIGINS=http://127.0.0.1:4173,http://localhost:4173
```

See `.env.example`.

## Frontend Quickstart

In one terminal, run the API. In another:

```bash
cd site
npm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://127.0.0.1:4173`.

The page is a technical preview:

- hero comparison visual
- featured four-stage demo from an existing processed clip
- volunteer upload card wired to the FastAPI service

## Offline Pipeline

The API path is the portfolio-facing service. The broader offline pipeline is still available for corpus building and artifact review.

Useful commands:

```bash
python scripts/discover_youtube.py \
  --runner-data data/runners.yml \
  --out artifacts/discovery/candidates.jsonl \
  --limit-per-query 5

python scripts/evaluate_video_candidates.py \
  --limit 30 \
  --sample-count 16 \
  --out artifacts/evaluation/video_candidates.top30.csv

python scripts/serve_review_ui.py --limit 20 --port 8765

python scripts/prepare_single_clip_cv_run.py --candidate-id <candidate_id>
python scripts/run_pose_landmarks.py --candidate-id <candidate_id> --model-variant heavy
python scripts/compile_form_features.py --candidate-id <candidate_id>
```

More detail:

- `docs/single-clip-cv-flow.md`
- `docs/identity-stable-runner-pipeline-plan.md`
- `docs/similarity-search-implementation-plan.md`

## Tests

```bash
python -m pytest
cd site && npm run typecheck && npm run build
```

The test suite covers:

- runner seed data contracts
- YouTube candidate scoring
- clip curation
- pose target selection
- form-feature compilation
- pipeline manifest updates
- API validation and response shape

## Current Limits

- The API is optimized for short clips, not full races.
- The public matching layer is not complete yet.
- DensePose, identity tracking, and SAM 3.1 masking are available as offline stages, not part of the default web request.
- Uploaded API artifacts are local by default. A production deployment should move source clips and generated artifacts to object storage and run inference in background workers.

## Design Notes

Primary matching will use pose-sequence similarity. Masks and DensePose are treated as confidence/QA layers, not as the source of truth.

See `CONTEXT.md` for project language and `docs/adr/0001-pose-sequence-primary-densepose-confidence-layer.md` for the core architecture decision.
