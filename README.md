<p align="center">
  <img src="site/public/assets/brand/logo-lockup.svg" alt="Who Do I Run Like" width="360" />
</p>

# Who Do I Run Like

Technical preview for a running-form computer vision pipeline.

Upload a short running clip, choose the target runner, and the system returns visual review artifacts: target-runner isolation, pose render, QA overlay, fused overlay, and basic quality metrics. The matching/reference-library side is still early; the current product is about making the clip-processing path reliable and inspectable.

Live site: [whodoirunlike.com](https://whodoirunlike.com)

## What is working

- Next.js demo site with a featured walkthrough, gallery, about page, and volunteer upload flow.
- Cloudflare Pages for the site.
- Cloudflare Worker API for uploads, job status, and artifact serving.
- R2-backed storage for source clips, job records, and processed outputs.
- RunPod Serverless GPU processor for identity tracking, SAM 3.1 runner masks, pose, DensePose, fusion, and QC.
- Attempt, stage, span, resource, and five-second progress telemetry with an R2 operational timeline.
- Private AWS analytics adapter using API Gateway, FIFO SQS, Lambda, S3, Glue, and Athena.
- Private processing dashboard at [analytics.whodoirunlike.com](https://analytics.whodoirunlike.com) for stage tails, attempt waterfalls, stalls, and sanitized failures.
- Local FastAPI path for quick short-clip pose artifact checks.
- Offline scripts for candidate clip discovery, review, and curation.

## Hosted flow

1. The browser uploads a clip and target-runner box to the Worker at `api.whodoirunlike.com`.
2. The Worker stores the source clip and job record in R2.
3. RunPod Serverless runs the full CV processor.
4. The processor writes artifacts back through the Worker.
5. The Worker stores the attempt timeline in R2 and asynchronously exports metadata events to AWS when configured.
6. The site polls job status and links to the finished overlay.

The RunPod endpoint scales down when idle, so the first run after a quiet period can take a few minutes to start.

## Run locally

API:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn whodoirunlike.api:app --host 127.0.0.1 --port 8000
```

Site:

```bash
cd site
npm ci
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://127.0.0.1:4173`.

## Checks

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
cd site && npm run typecheck && npm run build:pages
cd worker && npm run check
cd infra/analytics && npm test && npm run synth
cd analytics-dashboard && npm run check
```

## Deploy notes

Cloudflare Pages:

```text
Root directory: site
Build command: npm ci && npm run build:pages
Build output directory: out
Production branch: main
```

Hosted-service setup references:

- [worker/README.md](worker/README.md)
- [site/README.md](site/README.md)
- [infra/analytics/README.md](infra/analytics/README.md)
- [analytics-dashboard/README.md](analytics-dashboard/README.md)

## Repo map

```text
site/                         Next.js technical-preview site
worker/                       Cloudflare Worker for uploads, jobs, and R2 artifacts
infra/analytics/              private AWS processing-metadata analytics adapter
analytics-dashboard/          private Cloudflare-hosted Athena analytics UI and query proxy
src/whodoirunlike/api.py      local FastAPI upload endpoint
src/whodoirunlike/full_pipeline.py
                              identity, SAM, pose, DensePose, fusion, features, QC
src/whodoirunlike/running_clip_run.py
                              canonical run manifest, stage state, and artifact paths
src/whodoirunlike/runpod_serverless.py
                              RunPod Serverless entrypoint
scripts/                      ingestion, review, smoke-test, and curation commands
schemas/                      JSON contracts for clips, runners, artifacts, and QC
docs/adr/                     public architectural decision records
```

## Limits

- Short clips only.
- Matching is not production-ready yet.
- Hosted processing depends on Cloudflare R2, Worker secrets, and the current RunPod endpoint.
- This is an entertainment/research prototype, not coaching, medical, or biometric-identification software.
