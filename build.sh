#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Check if Go is available; if not, skip binary builds but allow other tasks.
if ! command -v go &> /dev/null; then
    echo "WARNING: Go compiler not found on PATH. Skipping binary builds."
    echo "         Python/runtime dependencies can still be updated separately."
    exit 0
fi

mkdir -p dist

echo "[1/4] Building Windows app starter..."
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o TrainFlow.exe ./cmd/trainflow
cp TrainFlow.exe dist/trainflow-windows-amd64.exe

echo "[2/4] Building Linux app starter..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o TrainFlow ./cmd/trainflow
cp TrainFlow dist/trainflow-linux-amd64

echo "[3/4] Building Windows runtime tool..."
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool.exe ./cmd/runtime-tool
cp TrainFlow_Runtime_Tool.exe dist/trainflow-runtime-tool-windows-amd64.exe

echo "[4/4] Building Linux runtime tool..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool ./cmd/runtime-tool
cp TrainFlow_Runtime_Tool dist/trainflow-runtime-tool-linux-amd64

echo ""
echo "Done:"
echo "  TrainFlow"
echo "  TrainFlow.exe"
echo "  TrainFlow_Runtime_Tool"
echo "  TrainFlow_Runtime_Tool.exe"
echo "  dist/trainflow-windows-amd64.exe"
echo "  dist/trainflow-linux-amd64"
echo "  dist/trainflow-runtime-tool-windows-amd64.exe"
echo "  dist/trainflow-runtime-tool-linux-amd64"
