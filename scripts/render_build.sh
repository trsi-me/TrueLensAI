#!/usr/bin/env bash
# Render build: LFS → slim deps → fetch model URLs → export .h5→.tflite if still missing → verify.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/render_git_lfs_pull.sh

pip install --no-cache-dir -r requirements-app.txt
python scripts/fetch_pretrained_models.py

H5="ml_models/saved_models/image_model.h5"
TFL="ml_models/saved_models/image_model.tflite"

tfl_ok=0
if [[ -f "$TFL" ]]; then
  tsz=$(wc -c <"$TFL" | tr -d ' ')
  if [[ "$tsz" -ge 10000 ]]; then
    echo "render_build: image_model.tflite OK (${tsz} bytes)" >&2
    tfl_ok=1
  fi
fi

if [[ "$tfl_ok" -eq 1 ]]; then
  exit 0
fi

if [[ ! -f "$H5" ]]; then
  echo "render_build: no usable image_model.tflite. Add .h5 to repo/LFS, or set IMAGE_MODEL_TFLITE_DOWNLOAD_URL / PRETRAINED_MODELS_BASE_URL." >&2
  exit 1
fi

hsz=$(wc -c <"$H5" | tr -d ' ')
if [[ "$hsz" -lt 4096 ]]; then
  echo "render_build: image_model.h5 too small (LFS pointer?). Fix LFS or set IMAGE_MODEL_TFLITE_DOWNLOAD_URL." >&2
  exit 1
fi

echo "render_build: exporting image_model.tflite from .h5 (build step)..." >&2
export TF_CPP_MIN_LOG_LEVEL=2
pip install --no-cache-dir "tensorflow==2.13.0"
python scripts/export_image_model_tflite.py
pip uninstall -y tensorflow keras keras-nightly 2>/dev/null || true
pip install --no-cache-dir -r requirements-app.txt

tsz=$(wc -c <"$TFL" | tr -d ' ')
if [[ ! -f "$TFL" ]] || [[ "$tsz" -lt 10000 ]]; then
  echo "render_build: TFLite export failed (${tsz:-0} bytes). Check build logs above." >&2
  exit 1
fi
echo "render_build: TFLite created (${tsz} bytes)" >&2
