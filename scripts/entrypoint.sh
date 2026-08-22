#!/bin/sh
# TrainFlow container entrypoint.
# - Bootstraps the embedded python runtime exactly once (skipped when torch imports).
# - Then execs the TrainFlow app (root=/app, binary beside training/).
set -eu
cd /app

PY=/app/python_embeded/linux/bin/python

need_bootstrap=0
if [ ! -x "$PY" ] || ! "$PY" -c "import torch" >/dev/null 2>&1; then
  need_bootstrap=1
fi

if [ "${TRAINFLOW_BOOTSTRAP:-auto}" = "off" ]; then
  echo "[entrypoint] bootstrap disabled; expecting pre-baked runtime"
elif [ "$need_bootstrap" -eq 1 ]; then
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
exec /app/"${TRAINFLOW_BIN:-TrainFlow}" "$@"
