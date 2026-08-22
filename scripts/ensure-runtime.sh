#!/bin/sh
# One-time embedded runtime bootstrap for the TrainFlow container.
# Mirrors internal/runtimeops: installLinuxLocalPython + installTrainerDeps
# (runtimeops.go:933, runtimeops.go:323). Writes under /app/python_embeded,
# which persists in the trainflow_tf named volume.
set -eu
cd /app

UV=/usr/local/bin/uv
PYDIR=/app/python_embeded/linux

# 1) Python 3.12 via uv (mirrors installLinuxLocalPython, incl. flatten)
if [ ! -x "$PYDIR/bin/python" ]; then
  echo "[runtime] installing Python 3.12 (uv managed)..."
  UV_PYTHON_INSTALL_DIR="$PYDIR" "$UV" python install 3.12 --python-preference only-managed
  # uv lays out $PYDIR/<distro-dir>/... ; flatten into $PYDIR/ exactly like
  # the Go code (runtimeops.go installLinuxLocalPython): move every entry of
  # each immediate subdirectory up one level, then remove the empty dir.
  for entry in "$PYDIR"/*; do
    [ -d "$entry" ] || continue
    (cd "$entry" && find . -maxdepth 1 -mindepth 1 -exec mv -f {} "$PYDIR/" \;)
    rmdir "$entry" 2>/dev/null || true
  done
fi
rm -f "$PYDIR/lib/python3.12/EXTERNALLY-MANAGED"
PY="$PYDIR/bin/python"

# 2) Torch backend (mirrors torchInstallPlan, runtimeops.go:115)
case "${TORCH_BACKEND:-cuda13}" in
  cuda13|"") IDX=https://download.pytorch.org/whl/cu130 ;;
  rocm)      IDX=https://download.pytorch.org/whl/rocm6.4 ;;
  skip)      IDX="" ;;
  *) echo "unknown TORCH_BACKEND=${TORCH_BACKEND}, defaulting to cuda13"; IDX=https://download.pytorch.org/whl/cu130 ;;
esac

echo "[runtime] upgrading pip/setuptools/wheel..."
"$UV" pip install --python "$PY" --upgrade pip "setuptools<82" wheel

if [ -n "$IDX" ]; then
  echo "[runtime] installing PyTorch ($IDX)..."
  "$UV" pip install --python "$PY" --upgrade torch torchvision torchaudio --index-url "$IDX"
else
  echo "[runtime] TORCH_BACKEND=skip: verifying existing torch..."
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

# uv >= 0.12 resolves requirements/edits against the project discovered at
# CWD, so each trainer install runs from its own project directory —
# exactly what the Go installer's installIn does (runtimeops.go installTrainerDeps).
echo "[runtime] installing sd-scripts requirements + editable..."
(cd training/sd-scripts && "$UV" pip install --python "$PY" -r requirements.txt -e .)
echo "[runtime] installing musubi-tuner overlay + editable..."
(cd training/musubi-tuner && "$UV" pip install --python "$PY" -r ../requirements-musubi-overlay.txt -e .) || echo "[runtime] editable musubi install failed; PYTHONPATH runtime mode will be used"
echo "[runtime] installing UI/prep dependencies..."
"$UV" pip install --python "$PY" psutil toml pillow onnx onnxruntime-gpu pandas opencv-python

echo "[runtime] verifying..."
"$PY" -c "import torch, accelerate, transformers, diffusers, cv2; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
# Marker so the entrypoint can skip the bootstrap on later starts.
: > /app/python_embeded/linux/.trainflow-ready
echo "[runtime] bootstrap complete"
