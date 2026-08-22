# syntax=docker/dockerfile:1

########## Stage 1: Go binaries ##########
FROM docker.io/golang:1.22 AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/TrainFlow ./cmd/trainflow \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/TrainFlow_Runtime_Tool ./cmd/runtime-tool

########## Stage 2: Runtime image ##########
FROM docker.io/python:3.12-bookworm
# git: runtime-tool app-binary update path; ca-certs: model downloads;
# libgl1/libglib2.0-0: opencv-python (non-headless) import at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# uv for the one-time python runtime bootstrap (option A) or pre-baking (option B).
# Direct platform binary download, same URLs as internal/runtimeops ensureUV.
# The tarball nests the binary under <pkg>/uv (older releases shipped a flat `uv`).
RUN set -eux \
 && case "$(uname -m)" in \
    x86_64)  UV_PKG=uv-x86_64-unknown-linux-gnu.tar.gz ;; \
    aarch64) UV_PKG=uv-aarch64-unknown-linux-gnu.tar.gz ;; \
    *) echo "unsupported architecture $(uname -m)" && exit 1 ;; \
 esac \
 && curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/$UV_PKG" -o /tmp/uv.tar.gz \
 && tar xzf /tmp/uv.tar.gz -C /tmp \
 && install -m 0755 "$(find /tmp -maxdepth 2 -name uv -type f | head -n1)" /usr/local/bin/uv \
 && rm -rf /tmp/uv.tar.gz /tmp/uv-* \
 && uv --version

# The app layer lives OUTSIDE the /app named volume so image rebuilds are
# picked up: the entrypoint syncs /opt/trainflow -> /app on every start.
# (Podman/Docker seed a named volume with image contents only on FIRST mount,
# so a volume mounted at /app would keep stale binaries/scripts forever.)
COPY --from=build /out/TrainFlow /out/TrainFlow_Runtime_Tool /opt/trainflow/
COPY training/ /opt/trainflow/training/
COPY scripts/entrypoint.sh /opt/trainflow/entrypoint.sh
COPY scripts/ensure-runtime.sh /opt/trainflow/ensure-runtime.sh
RUN chmod +x /opt/trainflow/TrainFlow /opt/trainflow/TrainFlow_Runtime_Tool /opt/trainflow/entrypoint.sh /opt/trainflow/ensure-runtime.sh \
 && mkdir -p /app/python_embeded /app/datasets

WORKDIR /app
ENV TRAINFLOW_NO_BROWSER=1 \
    TRAINFLOW_ADDR=:7860 \
    TORCH_BACKEND=cuda13
EXPOSE 7860 7870
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1
ENTRYPOINT ["/opt/trainflow/entrypoint.sh"]
