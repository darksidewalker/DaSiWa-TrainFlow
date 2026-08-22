#!/usr/bin/env bash
# Build (optional) and start TrainFlow with GPU passthrough on docker or podman.
# Engine detection per devops/container-engine-compat; override with CONTAINER_ENGINE.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-dasiwa/trainflow:latest}
DATASETS_DIR=${DATASETS_DIR:-$PWD/datasets}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
mkdir -p "$DATASETS_DIR" "$MODELS_DIR"

detect_engine() {
  local force="${CONTAINER_ENGINE:-}"
  if [[ -n "$force" ]]; then
    command -v "$force" >/dev/null 2>&1 && { echo "$force"; return 0; }
    return 1
  fi
  command -v docker >/dev/null 2>&1 && { echo docker; return 0; }
  command -v podman >/dev/null 2>&1 && { echo podman; return 0; }
  return 1
}

CE=$(detect_engine) || { echo "No container engine found (install docker or podman)"; exit 1; }
echo "Using container engine: $CE"

if [[ "${TRAINFLOW_BUILD:-0}" == "1" ]]; then
  "$CE" build -t "$IMAGE" .
fi

GPU_ARGS=()
SMI_ARG=()
if [[ "${TRAINFLOW_GPU:-1}" == "1" ]]; then
  case "$CE" in
    podman)
      # Rootless + CDI (this host: /etc/cdi/nvidia.yaml). The classic
      # nvidia.com/gpu device only exists for podman's built-in compose.
      GPU_ARGS=(--device nvidia.com/gpu=all)
      # nvidia-smi for the UI GPU tiles (host tool, bind is engine-agnostic)
      if [[ -x /usr/bin/nvidia-smi ]]; then
        SMI_ARG=(-v /usr/bin/nvidia-smi:/usr/local/bin/nvidia-smi:ro)
      fi
      ;;
    docker)
      GPU_ARGS=(--gpus all)
      ;;
  esac
fi

# First start: if the one-time bootstrap hasn't run yet, start in the
# foreground so you can watch the ~5-10 min runtime install; it is cached
# in the volume afterwards.
if "$CE" volume inspect trainflow_tf >/dev/null 2>&1; then
  echo "Volume trainflow_tf exists; detaching."
  DETACH=(--detach)
else
  echo "First start: running in foreground for the one-time runtime bootstrap (Ctrl-C is safe, re-run afterwards)."
  DETACH=()
fi

"$CE" run \
  --name trainflow --replace \
  "${GPU_ARGS[@]}" \
  --ipc host \
  -p 7860:7860 -p 7870:7870 \
  -v trainflow_tf:/app \
  -v "$DATASETS_DIR":/app/datasets \
  -v "$MODELS_DIR":/app/models \
  "${SMI_ARG[@]}" \
  -e TRAINFLOW_NO_BROWSER=1 \
  -e "TORCH_BACKEND=${TORCH_BACKEND:-cuda13}" \
  "${DETACH[@]}" \
  "$IMAGE"

echo "UI:  http://127.0.0.1:7860/"
echo "Runtime tool: start via compose (shares the same app volume):"
echo "  $CE compose up -d trainflow-tool   # UI on http://127.0.0.1:7871/"
