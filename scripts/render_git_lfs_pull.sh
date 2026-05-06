#!/usr/bin/env bash
# Render: native images have git but not git-lfs; shallow clones can leave *.h5/*.pkl as LFS pointers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "render_git_lfs_pull: not a git checkout; skipping LFS." >&2
  exit 0
fi

_ensure_git_lfs() {
  if git lfs version >/dev/null 2>&1; then
    return 0
  fi
  local ver arch url tmp dir
  ver="${GIT_LFS_VERSION:-3.6.1}"
  arch="amd64"
  url="https://github.com/git-lfs/git-lfs/releases/download/v${ver}/git-lfs-linux-${arch}-v${ver}.tar.gz"
  tmp="${TMPDIR:-/tmp}/truelens-git-lfs-$$"
  mkdir -p "$tmp"
  curl -fsSL "$url" | tar -xz -C "$tmp"
  dir="$(find "$tmp" -maxdepth 1 -type d -name 'git-lfs-*' | head -1)"
  if [[ -z "$dir" || ! -x "$dir/git-lfs" ]]; then
    echo "render_git_lfs_pull: failed to unpack git-lfs from $url" >&2
    exit 1
  fi
  export PATH="$dir:$PATH"
  git lfs version
}

_ensure_git_lfs
git lfs install

# Shallow clones (common on CI) sometimes need more history / explicit fetch for LFS blobs.
if [[ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" == "true" ]]; then
  echo "render_git_lfs_pull: shallow repo; widening fetch for LFS..." >&2
  git fetch --unshallow 2>/dev/null || git fetch --depth=500 origin "$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null || true
fi

echo "render_git_lfs_pull: fetching LFS objects for HEAD..." >&2
git lfs fetch origin "$(git rev-parse HEAD)" 2>/dev/null || git lfs fetch origin 2>/dev/null || true
git lfs checkout || true
git lfs pull

echo "render_git_lfs_pull: ml_models/saved_models sizes:" >&2
ls -la ml_models/saved_models 2>/dev/null || true

IMG="ml_models/saved_models/image_model.h5"
if [[ -f "$IMG" ]]; then
  sz=$(wc -c <"$IMG" | tr -d ' ')
  echo "render_git_lfs_pull: image_model.h5 is ${sz} bytes" >&2
  # Pointers are tiny; real Keras .h5 here is tens of MB.
  if [[ "$sz" -lt 4096 ]]; then
    if [[ -n "${PRETRAINED_MODELS_BASE_URL:-}" || -n "${IMAGE_MODEL_DOWNLOAD_URL:-}" ]]; then
      echo "render_git_lfs_pull: small file but PRETRAINED / IMAGE download URL set; fetch_pretrained_models may fix it." >&2
    else
      echo "render_git_lfs_pull: image_model.h5 looks like an LFS pointer (not hydrated). Add this script to buildCommand before pip, or set PRETRAINED_MODELS_BASE_URL." >&2
      exit 1
    fi
  fi
fi
