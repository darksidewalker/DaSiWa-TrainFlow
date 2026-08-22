#!/usr/bin/env sh
# Push the local TrainFlow image to Docker Hub.
#
# Token handling (the token is never committed and never persists on disk):
#   1. $DOCKERHUB_TOKEN environment variable, or
#   2. the gitignored .env.docker file in the repo root
#      (format: export DOCKERHUB_TOKEN=<your-pat>)
#
# The script logs in, pushes, and immediately logs back out, so the token
# is only held in memory for the duration of the push.
set -eu
cd "$(dirname "$0")/.."

REGISTRY=${REGISTRY:-docker.io}
REGISTRY_USER=${REGISTRY_USER:-darksidewalker}
IMAGE=${IMAGE:-dasiwa/trainflow:latest}
REMOTE="${REGISTRY}/${REGISTRY_USER}/dasiwa-trainflow:latest"

# 1) Find the token
if [ -z "${DOCKERHUB_TOKEN:-}" ] && [ -f .env.docker ]; then
  # shellcheck disable=SC1091
  . ./.env.docker
fi
if [ -z "${DOCKERHUB_TOKEN:-}" ]; then
  echo "error: no registry token found." >&2
  echo "  set DOCKERHUB_TOKEN, or add it to the gitignored .env.docker:" >&2
  echo "    export DOCKERHUB_TOKEN=<your-pat>" >&2
  exit 1
fi

# 2) Engine detection (same logic as scripts/run.sh)
if [ -n "${CONTAINER_ENGINE:-}" ]; then
  CE=$CONTAINER_ENGINE
elif command -v docker >/dev/null 2>&1; then
  CE=docker
elif command -v podman >/dev/null 2>&1; then
  CE=podman
else
  echo "error: no container engine found (install docker or podman)" >&2
  exit 1
fi

# 3) One-shot login; always log out on exit so nothing persists
login_cmd() {
  printf '%s' "$DOCKERHUB_TOKEN" | "$CE" login "$REGISTRY" -u "$REGISTRY_USER" --password-stdin
}
logout_cmd() {
  "$CE" logout "$REGISTRY" >/dev/null 2>&1 || true
}
trap logout_cmd EXIT
login_cmd
echo "logged in to $REGISTRY as $REGISTRY_USER"

# 4) Tag + push
"$CE" tag "$IMAGE" "$REMOTE"
"$CE" push "$REMOTE"
echo "pushed $REMOTE (logged out; token not stored)"
