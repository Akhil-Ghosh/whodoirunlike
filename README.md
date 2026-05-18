# Who Do I Run Like

Offline-first CV ingestion pipeline for `whodoirunlike.com`.

The first product goal is an entertainment similarity experience: upload a running clip, generate a form/motion representation, and return the elite runner whose motion looks closest. Sprinting stays in the product model, but the first working pipeline targets distance running.

## MVP Decisions

- Product mode: entertainment-first, not coaching or injury analysis.
- First supported mode: `running`; `sprinting` is planned but "coming soon" in UI.
- User input: video-only, anonymous upload, temporary storage by default.
- Corpus target: 30 elite distance runners, split 10 / 10 / 10 across `800_1500`, `5k_10k`, and `marathon`.
- Corpus clips: 3-5 approved reference segments per runner.
- Ingestion starts with YouTube discovery, then human review of URL + timestamps.
- Matching representation: pose-sequence similarity as the primary signal.
- Experiments: Gemini Embedding 2 on normalized skeleton render video and masked runner video.
- Visual output: SAM/ZIM-style mask plus pose/body map for private alpha, treated as visual evidence rather than the matching source of truth.

## Repo Layout

```text
data/
  runners.yml                    # Opinionated seed corpus list
  approved_segments.example.yml  # Human-approved segment shape
schemas/
  *.schema.json                  # Contracts for review + future CV artifacts
scripts/
  discover_youtube.py            # Create candidate video queue with yt-dlp search
  candidates_to_csv.py           # Convert JSONL candidates to review CSV
  serve_review_ui.py             # Local clip review/annotation UI
src/whodoirunlike/
  discovery.py                   # Shared discovery helpers and data contracts
  review_app.py                  # Lightweight local review server
artifacts/
  discovery/                     # Generated candidate queues
  review/                        # Local human labels, ignored by git
clips/
  raw/                           # Downloaded/source videos later
  segments/                      # Approved local segments later
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m camoufox fetch
```

## Discover Candidate YouTube Videos

Start small while we tune queries:

```bash
python scripts/discover_youtube.py \
  --runner-data data/runners.yml \
  --out artifacts/discovery/candidates.jsonl \
  --limit-per-query 5 \
  --max-runners 2
```

Export the review queue:

```bash
python scripts/candidates_to_csv.py \
  artifacts/discovery/candidates.jsonl \
  artifacts/discovery/candidates.csv
```

The discovery output is not an approved corpus. It is a review queue: you choose useful videos and timestamps, then add them to `data/approved_segments.yml` using the example file as a template.

## Score Candidate Videos

Metadata score first:

```bash
python scripts/evaluate_candidates.py \
  artifacts/discovery/candidates.jsonl \
  artifacts/discovery/candidates.scored.csv
```

Browser-backed search smoke tests:

```bash
python scripts/search_youtube_web.py \
  --query "Faith Kipyegon running form" \
  --runner-slug faith-kipyegon \
  --limit 3 \
  --backend both \
  --out artifacts/discovery/web_search.smoke.jsonl
```

CV score the strongest short/medium candidates:

```bash
python scripts/evaluate_video_candidates.py \
  --limit 30 \
  --sample-count 16 \
  --max-duration-seconds 600 \
  --out artifacts/evaluation/video_candidates.top30.csv
```

The CV pass downloads low-resolution source videos into `clips/raw/candidates`, samples frames, runs MediaPipe Pose Landmarker, and scores whether the clip appears to have usable full-body running footage. Treat this as triage, not final approval.

Download higher-quality copies for human review:

```bash
python scripts/download_review_videos.py \
  --limit 20 \
  --max-height 720 \
  --out artifacts/evaluation/video_candidates.review20_720.json
```

Use `--max-height 0` to fetch the highest available source format for each clip:

```bash
python scripts/download_review_videos.py \
  --limit 20 \
  --max-height 0 \
  --download-dir clips/raw/review_best \
  --out artifacts/evaluation/video_candidates.review20_best.json
```

## Review Candidate Clips

Run the local review UI against the top evaluated clips:

```bash
python scripts/serve_review_ui.py --limit 20 --port 8765
```

Open `http://127.0.0.1:8765`. The UI serves local candidate videos with byte-range support for scrubbing, lets you set start/end timestamps, previews the saved segment, records camera angle, and saves `good`, `mid`, or `bad` labels to `artifacts/review/clip_reviews.json`.

After preparing a CV run, open `http://127.0.0.1:8765/subject.html` to select the target runner on the prompt frame and inspect the run artifact slots.

Run the first SAM 2 whole-runner mask after saving a subject prompt:

```bash
python -m pip install torch torchvision
SAM2_BUILD_CUDA=0 python -m pip install git+https://github.com/facebookresearch/sam2.git
mkdir -p models/sam2
curl -L -o models/sam2/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

python scripts/run_sam2_mask.py \
  --candidate-id d6ee6cd75cd04b95 \
  --checkpoint models/sam2/sam2.1_hiera_tiny.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml
```

This writes `runner_mask.mp4`, `masked_runner.mp4`, `qa_overlay.mp4`, and `runner_mask_metadata.jsonl` inside the CV run folder.

Experimental SAM 3.1 on Apple Silicon via MLX:

```bash
python -m pip install -e ".[sam31]"

python scripts/run_sam31_mlx_mask.py \
  --candidate-id d6ee6cd75cd04b95 \
  --model mlx-community/sam3.1-bf16 \
  --quality-mode native \
  --prompt "a runner" \
  --prompt "a person"
```

The SAM 3.1 MLX path keeps the same artifact contract as SAM 2. It uses text prompts to
detect runners on each frame, then uses the saved prompt box/points to choose the target
runner identity and write the mask videos. The first run downloads the MLX model from
Hugging Face. Quality modes are `max` (source-sized square resolution, capped for sanity),
`native` (`1008`, the SAM 3.1 default), and `fast` (`224`).

Run MediaPipe pose extraction after a mask pass:

```bash
python scripts/run_pose_landmarks.py \
  --candidate-id d6ee6cd75cd04b95 \
  --model-variant heavy
```

This prefers `masked_runner.mp4` when present, writes `pose_landmarks.jsonl`,
`skeleton_render.mp4`, `features.json`, and refreshes `qa_overlay.mp4` with the runner mask
plus skeleton.

DensePose is optional and stays off the critical path. The local default uses the official
Detectron2 DensePose R50-FPN config and weights under `models/densepose/` when present:

```text
models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml
models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl
```

Manual setup:

```bash
PIP_NO_BUILD_ISOLATION=1 CC=clang CXX=clang++ \
  python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'

PIP_NO_BUILD_ISOLATION=1 CC=clang CXX=clang++ \
  python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose'

mkdir -p models/densepose/weights
git clone --depth 1 https://github.com/facebookresearch/detectron2.git models/densepose/detectron2
curl -L -o models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl \
  https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
```

Run:

```bash
python scripts/run_densepose.py \
  --candidate-id d6ee6cd75cd04b95 \
  --config models/densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  --weights models/densepose/weights/densepose_rcnn_R_50_FPN_s1x_model_final_162be9.pkl \
  --device cpu
```

The local review UI also uses those default files automatically. CPU inference is slow:
expect roughly a few minutes for an 8-10 second 1080p clip.

Fuse pose, mask, and DensePose into the confidence-weighted form artifact:

```bash
python scripts/run_fused_form.py \
  --candidate-id d6ee6cd75cd04b95
```

This writes `fused_form.jsonl` and `fused_overlay.mp4`. Matching should still use the
pose sequence as the motion truth; the fused output supplies per-frame/per-joint weights,
DensePose body-region coverage, occlusion/identity-risk states, and the alpha QA overlay.

## Next Pipeline Milestones

The detailed similarity-search tracker lives in
[`docs/similarity-search-implementation-plan.md`](docs/similarity-search-implementation-plan.md).

Prepare one reviewed clip for the single-clip CV loop:

```bash
python scripts/prepare_single_clip_cv_run.py --candidate-id <candidate_id>
```

This creates `artifacts/cv_runs/<candidate_id>/source_segment.mp4`, a prompt frame, a `person_prompt.json` selection stub, and a `cv_run_manifest.json` describing the segmentation, pose, DensePose, render, and feature stages.

1. Upgrade candidate CV scoring from uniform sampling to best contiguous pose-window detection.
2. Add the prompt-frame selection UI for `person_prompt.json`.
3. Add SAM 2.1 whole-runner mask generation.
4. Scale pose extraction over approved segments.
5. Add Detectron2 DensePose as a secondary body-surface layer after target tracking is stable.
6. Generate fused confidence-weighted form artifacts.
7. Generate canonical body-map render videos.
8. Compute pose-sequence similarities and Gemini render embeddings.
