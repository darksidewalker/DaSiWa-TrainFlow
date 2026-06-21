#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p dist

echo "[1/4] Building Windows app starter..."
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o dist/trainflow-windows-amd64.exe ./cmd/trainflow

echo "[2/4] Building Linux app starter..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o dist/trainflow-linux-amd64 ./cmd/trainflow

echo "[3/4] Building Windows runtime tool..."
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o dist/trainflow-runtime-tool-windows-amd64.exe ./cmd/runtime-tool

echo "[4/4] Building Linux runtime tool..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o dist/trainflow-runtime-tool-linux-amd64 ./cmd/runtime-tool

echo ""
echo "Done:"
echo "  dist/trainflow-windows-amd64.exe"
echo "  dist/trainflow-linux-amd64"
echo "  dist/trainflow-runtime-tool-windows-amd64.exe"
echo "  dist/trainflow-runtime-tool-linux-amd64"
