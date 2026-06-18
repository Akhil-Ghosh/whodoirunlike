# Who Do I Run Like

Small computer vision project for running clips.

You can upload a short clip, run pose extraction, and get back review artifacts:

- a skeleton video
- a QA overlay
- pose quality metrics
- a small feature summary

The matching part is still a work in progress. Right now the repo is mostly about getting from raw video to usable pose artifacts without hiding the messy parts.

## Current state

- FastAPI endpoint for short video uploads
- MediaPipe pose pass over each frame
- generated skeleton and QA videos
- Next.js preview site with demo, gallery, and about pages
- Cloudflare Worker for R2-backed uploads, job status, and artifacts
- RunPod Serverless processor for identity tracking, SAM 3.1 GPU masks, pose, DensePose, fusion, features, and QC
- offline scripts for finding and reviewing candidate running footage

No coaching claims. The metrics are rough signals for checking whether a clip is readable.

## What happens to a clip locally

1. Save the upload under `artifacts/api_runs/<run_id>/`.
2. Run MediaPipe pose landmarks.
3. Write the skeleton render, QA overlay, landmarks, and feature summary.
4. Return JSON with artifact URLs and basic quality metrics.

That local API path is synchronous because it is meant for quick short-clip checks.

## Hosted flow

The Cloudflare path is async:

1. The site uploads the clip to the Worker at `api.whodoirunlike.com`.
2. The Worker stores the source clip and job record in R2.
3. Once `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` are set, the Worker queues a RunPod Serverless job.
4. The RunPod processor downloads the clip, seeds a center-runner prompt, runs identity tracking, SAM 3.1 GPU, pose, DensePose, fusion, features, and QC, then uploads artifacts back through the Worker.

The production Worker is connected to RunPod. The endpoint scales to zero when idle, so the first request after a quiet period can take a few minutes to start.

## Run the API

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

Process a clip:

```bash
curl -X POST http://127.0.0.1:8000/v1/clips \
  -F "model_variant=lite" \
  -F "file=@/path/to/running-clip.mp4"
```

Artifacts are written locally by default. The first API call downloads the selected MediaPipe model into `models/mediapipe/`.

Useful environment variables:

```text
WHODOIRUNLIKE_API_ARTIFACT_ROOT=artifacts/api_runs
WHODOIRUNLIKE_MODEL_DIR=models/mediapipe
WHODOIRUNLIKE_MAX_UPLOAD_BYTES=78643200
WHODOIRUNLIKE_MAX_DURATION_SECONDS=20
WHODOIRUNLIKE_CORS_ORIGINS=http://127.0.0.1:4173,http://localhost:4173
WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET=<same secret as the Worker>
WHODOIRUNLIKE_HOSTED_RUN_ROOT=artifacts/hosted_runs
WHODOIRUNLIKE_IDENTITY_BACKEND=boxmot_bytetrack
WHODOIRUNLIKE_POSE_BACKEND=mmpose_rtmpose_l_384
WHODOIRUNLIKE_MASK_BACKEND=sam31_gpu
WHODOIRUNLIKE_MASK_QUALITY_MODE=native
HF_TOKEN=<token with facebook/sam3.1 access>
WHODOIRUNLIKE_SKIP_DENSEPOSE=false
DENSEPOSE_CONFIG=models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml
DENSEPOSE_WEIGHTS=models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl
```

See `.env.example` for the same defaults.

Hosted processor readiness:

```bash
curl http://127.0.0.1:8000/v1/processor/health
```

For the full hosted pipeline, `readiness.ready_for_full_pipeline` should be `true`. If it is false, the response names the missing identity, SAM 3.1 GPU, pose, or DensePose dependency.

RunPod setup lives in [docs/runpod-serverless.md](docs/runpod-serverless.md). After the Worker and cloud processor are connected, run an end-to-end hosted smoke test:

```bash
.venv/bin/python scripts/smoke_hosted_upload_flow.py \
  --api-base-url https://api.whodoirunlike.com \
  --clip /path/to/short-running-clip.mp4
```

## Run the site

In one terminal, run the API. In another:

```bash
cd site
npm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://127.0.0.1:4173`.

The site has a demo walkthrough, a gallery of processed runners, and an upload card. In local dev it calls the synchronous FastAPI endpoint. In production it uses the Cloudflare Worker upload/job flow.

For Cloudflare Pages, use `site` as the root directory, `npm ci && npm run build:pages` as the build command, and `out` as the output directory. The `whodoirunlike` Pages project is Git-backed and `whodoirunlike.com` is attached. If Cloudflare shows a Git disconnect warning, reconnect the Pages GitHub App to `Akhil-Ghosh/whodoirunlike` before relying on automatic deploys.

## Docker

```bash
docker build -t whodoirunlike-api .
docker run --rm -p 8000:8000 \
  -e WHODOIRUNLIKE_CORS_ORIGINS=http://127.0.0.1:4173 \
  whodoirunlike-api
```

## Repo map

```text
site/                         Next.js preview site
worker/                       Cloudflare Worker for uploads, jobs, and R2 artifacts
src/whodoirunlike/api.py      FastAPI upload endpoint
src/whodoirunlike/hosted_processor.py
                              hosted Worker-to-pipeline bridge
src/whodoirunlike/runpod_serverless.py
                              RunPod Serverless entrypoint
src/whodoirunlike/full_pipeline.py
                              identity, SAM, pose, DensePose, fusion, features, QC
src/whodoirunlike/pose_runner.py
                              pose inference and render outputs
src/whodoirunlike/form_features.py
                              normalized pose arrays and summary features
scripts/                      offline ingestion and review commands
schemas/                      JSON contracts
docs/                         plans and design notes
```

DensePose, SAM, identity tracking, and curation tools are still heavyweight. They run in the processor path, not inside Cloudflare Pages or the Worker runtime.

## Tests

```bash
.venv/bin/python -m pytest
cd site && npm run typecheck && npm run build
cd worker && npm run check
```

## Known limits

- Short clips only by default.
- Matching is not live yet.
- Hosted processing depends on Cloudflare R2, Worker secrets, and the current RunPod endpoint.
- The auto target prompt works best when the uploaded runner is centered.
- The gallery uses a small set of hand-reviewed examples, not a full reference corpus.
