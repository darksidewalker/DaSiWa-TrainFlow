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
  sh /app/scripts/ensure-runtime.sh
else
  echo "[entrypoint] embedded python runtime OK"
fi

export TRAINFLOW_NO_BROWSER="${TRAINFLOW_NO_BROWSER:-1}"
exec /app/TrainFlow
