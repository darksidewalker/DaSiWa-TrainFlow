#!/bin/sh
# TrainFlow container entrypoint.
# - Syncs image-owned files (binaries, training tree, scripts) into the /app
#   named volume on every start. Podman/Docker seed a named volume with image
#   contents only on FIRST mount, so without this sync, image rebuilds would
#   keep running the stale copy in the volume.
# - Bootstraps the embedded python runtime exactly once (marker-gated).
# - Then execs the app binary from /app (root=/app, binary beside training/).
set -eu
cd /app

TF=/opt/trainflow
PY=/app/python_embeded/linux/bin/python
READY=/app/python_embeded/linux/.trainflow-ready

# 1) Sync image-owned files into the /app volume (merge, never deletes user
# data). Binaries + scripts must track the current image; the training tree
# and all mutable state (settings, outputs, models, python_embeded) stay in
# the volume where the user/app writes them.
mkdir -p /app/python_embeded /app/datasets /app/models /app/scripts /app/training
cp -f "$TF/TrainFlow" "$TF/TrainFlow_Runtime_Tool" /app/
cp -f "$TF/entrypoint.sh" "$TF/ensure-runtime.sh" /app/scripts/
# Merge the image-owned training tree into the volume on every start so the
# code tracks the current image. Only tracked source files are in the image
# (settings.json, outputs and caches are .dockerignored), so app-written
# state in the volume is never clobbered.
if [ -d "$TF/training" ]; then
  cp -a "$TF/training/." /app/training/
fi

# 2) One-time runtime bootstrap gate
if [ ! -f "$READY" ] && [ -x "$PY" ] && "$PY" -c "import torch, accelerate, transformers, diffusers, cv2" >/dev/null 2>&1; then
  echo "[entrypoint] full runtime present; marking ready"
  : > "$READY"
fi

if [ "${TRAINFLOW_BOOTSTRAP:-auto}" = "off" ]; then
  echo "[entrypoint] bootstrap disabled; expecting pre-baked runtime"
elif [ ! -f "$READY" ]; then
  echo "[entrypoint] embedded python runtime missing/incomplete; running one-time bootstrap..."
  # flock so two services sharing the volume (app + runtime tool) cannot
  # bootstrap the same python_embeded concurrently.
  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 9
      sh /app/scripts/ensure-runtime.sh
    ) 9>/app/python_embeded/.bootstrap.lock
  else
    sh /app/scripts/ensure-runtime.sh
  fi
else
  echo "[entrypoint] embedded python runtime OK"
fi

export TRAINFLOW_NO_BROWSER="${TRAINFLOW_NO_BROWSER:-1}"
exec /app/"${TRAINFLOW_BIN:-TrainFlow}"
