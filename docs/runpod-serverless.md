# RunPod Serverless processor

This is the cloud processor path for public uploads.

Cloudflare keeps the public API and R2 storage. RunPod runs the GPU job. The Worker sends a RunPod Serverless job with the source clip URL; the RunPod worker downloads the clip through the protected Worker endpoint, runs the full CV pipeline, uploads artifacts back to R2 through the Worker, and reports status.

## Runtime target

- RunPod Serverless GPU endpoint
- CUDA 12.6+ and PyTorch 2.7+ for official SAM 3.1
- `WHODOIRUNLIKE_MASK_BACKEND=sam31_gpu`
- DensePose on CUDA with the Detectron2 DensePose project
- MMPose RTMPose for pose landmarks
- BoxMOT/YOLO identity tracking

Do not use SAM2 for this hosted path. Do not use SAM 3.1 MLX for this hosted path. MLX is only useful for local Apple Silicon runs.

## Image

The serverless image is defined in [Dockerfile.runpod](../Dockerfile.runpod).

The image keeps the RunPod base image's CUDA/Torch stack intact. [requirements-runpod-processor.txt](../requirements-runpod-processor.txt) installs the non-Torch Python dependencies, then the Dockerfile installs SAM 3.1, Ultralytics, BoxMOT, RTMLib, Detectron2, and DensePose with `--no-deps` so pip does not replace Torch/Torchvision during the build.

It starts:

```bash
python -m whodoirunlike.runpod_serverless
```

The manual GitHub Actions workflow builds:

```text
ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor:latest
```

Run it from GitHub Actions:

```bash
gh workflow run "Build RunPod Processor"
```

Watch it:

```bash
gh run list --workflow "Build RunPod Processor" --limit 5
```

This repository is public now, but confirm the GHCR package visibility after the first successful image push. If the package stays private, RunPod still needs registry auth.

## Pod-first debug path

Use this before Serverless. It is faster to debug the CUDA/SAM/DensePose runtime on an interactive pod than to keep waiting on full image builds.

Create a pod from the RunPod UI or CLI with:

```bash
runpodctl pod create \
  --name whodoirunlike-processor-debug \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --min-cuda-version 12.6 \
  --container-disk-in-gb 100 \
  --volume-in-gb 100 \
  --ports "8000/http,22/tcp"
```

Inside the pod, set the required secrets and run:

```bash
export WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET=<shared-secret>
export HF_TOKEN=<hf-token-with-facebook-sam3.1-access>
curl -fsSL https://raw.githubusercontent.com/Akhil-Ghosh/whodoirunlike/main/scripts/runpod_bootstrap_processor.sh | bash
```

When the script prints `ready_for_full_pipeline: true`, the FastAPI processor listens on port `8000`. For a temporary cloud-only smoke test, set the Worker `PROCESSOR_URL` to the pod's RunPod HTTP proxy URL and redeploy the Worker. Clear `PROCESSOR_URL` again before switching to Serverless.

## Required secrets

RunPod endpoint env:

```text
WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET=<same value as Cloudflare Worker>
HF_TOKEN=<Hugging Face token with access to facebook/sam3.1>
WHODOIRUNLIKE_MASK_BACKEND=sam31_gpu
DENSEPOSE_DEVICE=cuda
```

Cloudflare Worker secrets:

```bash
cd worker
npx wrangler secret put PROCESSOR_SHARED_SECRET
npx wrangler secret put RUNPOD_API_KEY
```

Checked-in Worker config gets the non-secret endpoint ID:

```json
"RUNPOD_ENDPOINT_ID": "<endpoint-id>"
```

## Create the RunPod endpoint

Create a GHCR registry auth in RunPod if the image stays private. Use a GitHub PAT with `read:packages`:

```bash
runpodctl registry create \
  --name ghcr-whodoirunlike \
  --username Akhil-Ghosh \
  --password '<github_pat_with_read_packages>'
```

Create the serverless template after the GHCR image exists:

```bash
runpodctl template create \
  --serverless \
  --name whodoirunlike-processor \
  --image ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor:latest \
  --container-disk-in-gb 100 \
  --env '{
    "WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET":"<shared-secret>",
    "HF_TOKEN":"<hf-token>",
    "WHODOIRUNLIKE_MASK_BACKEND":"sam31_gpu",
    "WHODOIRUNLIKE_IDENTITY_BACKEND":"boxmot_botsort",
    "WHODOIRUNLIKE_POSE_BACKEND":"mmpose_rtmpose_l_384",
    "WHODOIRUNLIKE_SKIP_DENSEPOSE":"false",
    "DENSEPOSE_DEVICE":"cuda"
  }'
```

Create the endpoint from that template. RTX 4090 is the first cost-conscious target; switch to A40/H100 if SAM 3.1 or DensePose needs more memory.

```bash
runpodctl serverless create \
  --name whodoirunlike-processor \
  --template-id <template-id> \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --min-cuda-version 12.6 \
  --workers-min 0 \
  --workers-max 2 \
  --idle-timeout 60 \
  --execution-timeout 900
```

Then set `RUNPOD_ENDPOINT_ID` in [worker/wrangler.jsonc](../worker/wrangler.jsonc) and deploy the Worker:

```bash
cd worker
npm run deploy
```

## Health check

RunPod Serverless uses a queue endpoint, so health is a small job:

```bash
curl -sS -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"type":"health"}}'
```

The response should include:

```json
{
  "status": "ok",
  "readiness": {
    "mask_backend": "sam31_gpu",
    "ready_for_full_pipeline": true
  }
}
```

## Public upload flow

Once Cloudflare has `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`, and `PROCESSOR_SHARED_SECRET`, this starts a job:

```bash
curl -X POST "https://api.whodoirunlike.com/v1/jobs/<run-id>/start"
```

The Worker stores `queued_on_runpod` progress with the RunPod job ID. RunPod then reports `running`, uploads artifacts, and reports `complete` or `failed` through the Worker.
