# Cloudflare hosting plan

This splits the deploy into two pieces:

1. Cloudflare Pages serves the static Next.js preview site.
2. RunPod Serverless runs the CV pipeline and stores artifacts through the Worker.

Pages should not run the full pipeline. The heavy path uses Python, OpenCV, YOLO/BoxMOT, SAM 3.1 GPU, DensePose, model files, and CUDA. Pages Functions run on the Workers runtime, which is a good fit for orchestration and storage access, not this kind of video processing.

## What is implemented now

The site is configured for static export:

- `site/next.config.ts` uses `output: "export"`.
- Next images are unoptimized so they work without the Next image server.
- `site/wrangler.jsonc` points Pages at `site/out`.
- `site/package.json` has `pages:preview` and `pages:deploy` scripts.
- The upload card uses the async Worker flow in production and the sync FastAPI path in local dev.

The Worker API is scaffolded under `worker/`:

- `POST /v1/uploads` stores the source clip in R2 and creates a job record.
- `POST /v1/jobs/:run_id/start` queues the processor job.
- `GET /v1/jobs/:run_id` returns job status.
- `GET /v1/artifacts/:run_id/:name` streams completed artifacts from R2.
- `GET /v1/jobs/:run_id/source` is token-protected for the processor.
- `PUT /v1/jobs/:run_id/artifacts/:name` is token-protected for processor artifact uploads.
- `POST /v1/jobs/:run_id/report` is token-protected for processor status updates.

Current production safety state:

- `api.whodoirunlike.com` is deployed and stores uploads in R2.
- `RUNPOD_ENDPOINT_ID` is blank in production, so the Worker does not queue RunPod jobs yet.
- A temporary local Cloudflare Tunnel was tested and removed. Do not use a laptop tunnel for the public demo.
- The existing Pages deployment was created by direct upload. Cloudflare cannot convert a Direct Upload project to Git-backed; recreate the Pages project from GitHub for the real production site.

The FastAPI app now exposes:

- `POST /v1/processor/jobs`, which still works for pod/debug deployments.
- `GET /v1/processor/health`, which reports whether the hosted machine has the shared secret, identity backend, SAM 3.1 GPU, pose backend, and DensePose config/weights/dependencies ready for the full pipeline.

RunPod Serverless uses [src/whodoirunlike/runpod_serverless.py](../src/whodoirunlike/runpod_serverless.py) instead of the FastAPI route, but both paths call the same `process_hosted_job` implementation.

Cloudflare Pages settings:

```text
Root directory: site
Build command: npm run build:pages
Build output directory: out
Production branch: main
Custom domain: whodoirunlike.com
Production env var: NEXT_PUBLIC_API_BASE_URL=https://api.whodoirunlike.com
Production env var: NEXT_PUBLIC_UPLOAD_API_MODE=async
Preview env var: NEXT_PUBLIC_API_BASE_URL=https://staging-api.whodoirunlike.com
```

Direct deploy from the repo:

```bash
cd site
npm ci
NEXT_PUBLIC_API_BASE_URL=https://api.whodoirunlike.com NEXT_PUBLIC_UPLOAD_API_MODE=async npm run build:pages
npx wrangler pages deploy out --project-name whodoirunlike
```

Worker deploy from the repo:

```bash
cd worker
npm ci
npx wrangler r2 bucket create whodoirunlike-clips
npx wrangler r2 bucket create whodoirunlike-clips-preview
npx wrangler secret put PROCESSOR_SHARED_SECRET
npx wrangler secret put RUNPOD_API_KEY
npm run deploy
```

`RUNPOD_ENDPOINT_ID` is intentionally empty in `worker/wrangler.jsonc`; set it only after the RunPod endpoint exists.

Processor readiness check:

```bash
curl -sS -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"type":"health"}}'
```

Do not route public uploads to the processor until `readiness.ready_for_full_pipeline` is `true`. The response breaks failures down by `processor_secret`, `identity`, `mask`, `pose`, and `densepose`.

## Processing service options

### RunPod Serverless

Use RunPod Serverless with a CUDA image built from [Dockerfile.runpod](../Dockerfile.runpod). This is the production path.

The mask backend is `sam31_gpu`, using the official `facebookresearch/sam3` SAM 3.1 video predictor. SAM2 is not part of this hosted plan. SAM 3.1 MLX remains local-only for Apple Silicon experiments.

## Recommended v1 architecture

```text
Cloudflare Pages
  static site
  NEXT_PUBLIC_API_BASE_URL points at the public API

Cloudflare Worker
  accepts upload bodies
  records job metadata
  queues jobs on RunPod Serverless
  serves private R2 artifacts by run ID

Cloudflare R2
  uploads/<run_id>/source.mp4
  jobs/<run_id>.json
  artifacts/<run_id>/...

RunPod Serverless
  downloads the source clip from the Worker
  runs identity tracking, SAM 3.1 GPU, pose, DensePose, fusion, features, QC
  uploads generated artifacts through the Worker
  reports job status through the Worker
```

For metadata, start with a small JSON job record in R2 or KV. Move to D1 once the status model needs filtering, admin views, or history.

## What I need from you

Cloudflare:

- Cloudflare account ID.
- Whether the Pages project should be named `whodoirunlike`.
- Confirmation that `whodoirunlike.com` is in the same Cloudflare account where Pages will live.
- The RunPod endpoint ID once the serverless endpoint is created.
- A Cloudflare API token with Pages edit access if you want me to deploy from this machine.
- R2 bucket names, or permission for me to create them.
- The shared `PROCESSOR_SHARED_SECRET` value, or permission for me to generate and set one.
- A RunPod API key for the Worker secret `RUNPOD_API_KEY`.

Pipeline hosting:

- RunPod Serverless is selected for v1.
- Budget comfort for compute while people try uploads.
- Whether uploaded clips should be deleted automatically, and after how long.
- Whether volunteer clips can be retained for improving the reference set, or only processed once.

Model/runtime:

- DensePose config and weights path, or permission to use the public Detectron2 DensePose model URL.
- Hugging Face token with access to `facebook/sam3.1`.
- GHCR registry access if the RunPod image stays private.
- Confirmation that `/v1/processor/health` is green on the processing machine before we connect public uploads.

Product:

- Max upload length for the public demo.
- Whether uploads should require email, or stay anonymous with a run link.
- Whether failed jobs should expose debug details to users or only to the admin logs.

## Next implementation step

Deploy order:

1. Log in to Cloudflare or provide a scoped API token.
2. Create the R2 buckets.
3. Deploy the Worker and bind `api.whodoirunlike.com`.
4. Build and publish the RunPod processor image.
5. Create the RunPod Serverless template and endpoint.
6. Confirm the RunPod health job reports `readiness.ready_for_full_pipeline: true`.
7. Set `RUNPOD_ENDPOINT_ID` in the Worker and redeploy.
8. Deploy Pages to `whodoirunlike.com`.

After that, submit one short centered running clip and verify that R2 contains the source, job record, and returned artifacts.

Smoke test command:

```bash
.venv/bin/python scripts/smoke_hosted_upload_flow.py \
  --api-base-url https://api.whodoirunlike.com \
  --clip /path/to/short-running-clip.mp4
```

Use the smoke test only after `RUNPOD_ENDPOINT_ID` points at the RunPod endpoint.
