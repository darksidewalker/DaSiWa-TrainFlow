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
# uv for the one-time python runtime bootstrap (option A) or pre-baking (option B)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --bin-dir /usr/local/bin

COPY --from=build /out/TrainFlow /out/TrainFlow_Runtime_Tool /app-staging/
COPY training/ /app-staging/training/
COPY scripts/entrypoint.sh /app-staging/scripts/entrypoint.sh
COPY scripts/ensure-runtime.sh /app-staging/scripts/ensure-runtime.sh
RUN mkdir -p /app/python_embeded /app/models /app/datasets \
 && cp -a /app-staging/. /app/ \
 && chmod +x /app/TrainFlow /app/TrainFlow_Runtime_Tool /app/scripts/*.sh \
 && rm -rf /app-staging

WORKDIR /app
ENV TRAINFLOW_NO_BROWSER=1 \
    TRAINFLOW_ADDR=:7860 \
    TORCH_BACKEND=cuda13
EXPOSE 7860 7870
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
