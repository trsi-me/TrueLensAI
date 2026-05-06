#!/usr/bin/env bash
# Render native builds include git but not git-lfs. LFS-tracked files would stay as
# tiny pointer files unless we install git-lfs and pull blobs before pip/install.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "render_git_lfs_pull: not a git checkout; skipping LFS pull." >&2
  exit 0
fi

if git lfs version >/dev/null 2>&1; then
  git lfs install
  git lfs pull
  exit 0
fi

LFS_VER="${GIT_LFS_VERSION:-3.6.1}"
ARCH="amd64"
URL="https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VER}/git-lfs-linux-${ARCH}-v${LFS_VER}.tar.gz"
TMP="${TMPDIR:-/tmp}/truelens-git-lfs-$$"
mkdir -p "$TMP"
curl -fsSL "$URL" | tar -xz -C "$TMP"
LFS_DIR="$(find "$TMP" -maxdepth 1 -type d -name 'git-lfs-*' | head -1)"
if [[ -z "$LFS_DIR" || ! -x "$LFS_DIR/git-lfs" ]]; then
  echo "render_git_lfs_pull: failed to unpack git-lfs from $URL" >&2
  exit 1
fi
export PATH="$LFS_DIR:$PATH"
git lfs version
git lfs install
git lfs pull
