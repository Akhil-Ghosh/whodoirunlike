#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${WHODOIRUNLIKE_APP_DIR:-/workspace/whodoirunlike}"
REPO_URL="${WHODOIRUNLIKE_REPO_URL:-https://github.com/Akhil-Ghosh/whodoirunlike.git}"
REPO_REF="${WHODOIRUNLIKE_REPO_REF:-main}"
DETECTRON2_DIR="${DETECTRON2_DIR:-/opt/detectron2}"
PORT="${PORT:-8000}"

export HF_HOME="${HF_HOME:-/runpod-volume/huggingface}"
export WHODOIRUNLIKE_HOSTED_RUN_ROOT="${WHODOIRUNLIKE_HOSTED_RUN_ROOT:-/runpod-volume/whodoirunlike/runs}"
export WHODOIRUNLIKE_IDENTITY_BACKEND="${WHODOIRUNLIKE_IDENTITY_BACKEND:-boxmot_botsort}"
export WHODOIRUNLIKE_POSE_BACKEND="${WHODOIRUNLIKE_POSE_BACKEND:-mmpose_rtmpose_l_384}"
export WHODOIRUNLIKE_MASK_BACKEND="${WHODOIRUNLIKE_MASK_BACKEND:-sam31_gpu}"
export WHODOIRUNLIKE_SAM31_GPU_USE_FA3="${WHODOIRUNLIKE_SAM31_GPU_USE_FA3:-false}"
export WHODOIRUNLIKE_SKIP_DENSEPOSE="${WHODOIRUNLIKE_SKIP_DENSEPOSE:-false}"
export DENSEPOSE_CONFIG="${DENSEPOSE_CONFIG:-$DETECTRON2_DIR/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml}"
export DENSEPOSE_WEIGHTS="${DENSEPOSE_WEIGHTS:-https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl}"
export DENSEPOSE_DEVICE="${DENSEPOSE_DEVICE:-cuda}"

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    ninja-build
  rm -rf /var/lib/apt/lists/*
fi

if [ ! -d "$APP_DIR/.git" ]; then
  mkdir -p "$(dirname "$APP_DIR")"
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
git fetch origin "$REPO_REF"
git checkout "$REPO_REF"
git pull --ff-only origin "$REPO_REF" || true

if [ ! -d "$DETECTRON2_DIR/.git" ]; then
  rm -rf "$DETECTRON2_DIR"
  git clone --depth 1 https://github.com/facebookresearch/detectron2.git "$DETECTRON2_DIR"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements-runpod-processor.txt
python -m pip install --no-cache-dir --no-deps -e .
python -m pip install --no-cache-dir --no-deps \
  git+https://github.com/facebookresearch/sam3.git \
  "ultralytics>=8.3" \
  "ultralytics-thop>=2.0.18" \
  "boxmot>=17" \
  "rtmlib>=0.0.15"
python -m pip install --no-cache-dir --no-deps "$DETECTRON2_DIR"
python -m pip install --no-cache-dir --no-deps "$DETECTRON2_DIR/projects/DensePose"

python - <<'PY'
from whodoirunlike.hosted_processor import processor_readiness
import json

print(json.dumps(processor_readiness(), indent=2, sort_keys=True))
PY

if [ "${WHODOIRUNLIKE_START_API:-1}" = "1" ]; then
  exec uvicorn whodoirunlike.api:app --host 0.0.0.0 --port "$PORT"
fi
