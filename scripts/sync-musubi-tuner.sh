#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/home/darksidewalker/GitHub/musubi-tuner}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/training/musubi-tuner"

# TrainFlow uses upstream Musubi for current scripts such as Krea2, but the
# LTX 2.3 integration still depends on LTX2 entrypoints that may live in an
# LTX-capable Musubi fork/local source. Do not blindly sync from an upstream
# tree that lacks ltx2_* files unless you restore/update the LTX2 files from
# that fork before testing.

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
