$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

New-Item -ItemType Directory -Force -Path dist | Out-Null

Write-Host "[1/4] Building Windows app starter..."
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"
go build -trimpath -ldflags="-s -w" -o dist/trainflow-windows-amd64.exe ./cmd/trainflow

Write-Host "[2/4] Building Linux app starter..."
$env:GOOS = "linux"
go build -trimpath -ldflags="-s -w" -o dist/trainflow-linux-amd64 ./cmd/trainflow

Write-Host "[3/4] Building Windows runtime tool..."
$env:GOOS = "windows"
go build -trimpath -ldflags="-s -w" -o dist/trainflow-runtime-tool-windows-amd64.exe ./cmd/runtime-tool

Write-Host "[4/4] Building Linux runtime tool..."
$env:GOOS = "linux"
go build -trimpath -ldflags="-s -w" -o dist/trainflow-runtime-tool-linux-amd64 ./cmd/runtime-tool

Write-Host ""
Write-Host "Done:"
Write-Host "  dist/trainflow-windows-amd64.exe"
Write-Host "  dist/trainflow-linux-amd64"
Write-Host "  dist/trainflow-runtime-tool-windows-amd64.exe"
Write-Host "  dist/trainflow-runtime-tool-linux-amd64"
