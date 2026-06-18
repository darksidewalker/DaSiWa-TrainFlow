#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/home/darksidewalker/GitHub/musubi-tuner}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/training/musubi-tuner"

if [[ ! -d "$SRC/src/musubi_tuner" ]]; then
  echo "Source does not look like musubi-tuner: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'datasets/' \
  --exclude 'output/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude '*.safetensors' \
  --exclude '*.ckpt' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  "$SRC/" "$DEST/"

echo "Synced musubi-tuner source to $DEST"
