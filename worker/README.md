# Who Do I Run Like Worker

Cloudflare Worker API for the public upload flow.

The Worker does three small jobs:

- stores volunteer clips in R2
- keeps a JSON job record in R2
- hands the job to RunPod Serverless when `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` are configured
- serves completed artifacts

It does not run SAM, DensePose, identity tracking, or video encoding. Those stay in the RunPod processor.

Production currently leaves `RUNPOD_ENDPOINT_ID` blank. Uploads are stored in R2, but `POST /v1/jobs/:run_id/start` returns `processor_configured: false` until the RunPod endpoint is created and configured.

## Local setup

```bash
cd worker
npm ci
cp .dev.vars.example .dev.vars
npm run dev
```

Set the same `PROCESSOR_SHARED_SECRET` in both `worker/.dev.vars` and the RunPod processor environment.

## Deploy

Create the R2 buckets once:

```bash
npx wrangler r2 bucket create whodoirunlike-clips
npx wrangler r2 bucket create whodoirunlike-clips-preview
```

Set the processor secret:

```bash
npx wrangler secret put PROCESSOR_SHARED_SECRET
npx wrangler secret put RUNPOD_API_KEY
```

After the RunPod endpoint exists, set `RUNPOD_ENDPOINT_ID` in `wrangler.jsonc`.

Deploy:

```bash
npm run deploy
```

The production route is `api.whodoirunlike.com`.
