$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Check if Go is available; if not, skip binary builds but allow other tasks.
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: Go compiler not found on PATH. Skipping binary builds."
    Write-Host "         Python/runtime dependencies can still be updated separately."
    exit 0
}

New-Item -ItemType Directory -Force -Path dist | Out-Null

Write-Host "[1/4] Building Windows app starter..."
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"
go build -trimpath -ldflags="-s -w" -o TrainFlow.exe ./cmd/trainflow
Copy-Item -Force TrainFlow.exe dist/trainflow-windows-amd64.exe

Write-Host "[2/4] Building Linux app starter..."
$env:GOOS = "linux"
go build -trimpath -ldflags="-s -w" -o TrainFlow ./cmd/trainflow
Copy-Item -Force TrainFlow dist/trainflow-linux-amd64

Write-Host "[3/4] Building Windows runtime tool..."
$env:GOOS = "windows"
go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool.exe ./cmd/runtime-tool
Copy-Item -Force TrainFlow_Runtime_Tool.exe dist/trainflow-runtime-tool-windows-amd64.exe

Write-Host "[4/4] Building Linux runtime tool..."
$env:GOOS = "linux"
go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool ./cmd/runtime-tool
Copy-Item -Force TrainFlow_Runtime_Tool dist/trainflow-runtime-tool-linux-amd64

Write-Host ""
Write-Host "Done:"
Write-Host "  TrainFlow"
Write-Host "  TrainFlow.exe"
Write-Host "  TrainFlow_Runtime_Tool"
Write-Host "  TrainFlow_Runtime_Tool.exe"
Write-Host "  dist/trainflow-windows-amd64.exe"
Write-Host "  dist/trainflow-linux-amd64"
Write-Host "  dist/trainflow-runtime-tool-windows-amd64.exe"
Write-Host "  dist/trainflow-runtime-tool-linux-amd64"
