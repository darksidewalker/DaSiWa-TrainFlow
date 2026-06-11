$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

New-Item -ItemType Directory -Force -Path dist | Out-Null

Write-Host "[1/4] Building Windows app starter..."
$env:CGO_ENABLED = "0"
$env:GOOS = "windows"
$env:GOARCH = "amd64"
go build -trimpath -ldflags="-s -w" -o TrainFlow.exe ./cmd/trainflow

Write-Host "[2/4] Building Linux app starter..."
$env:GOOS = "linux"
go build -trimpath -ldflags="-s -w" -o TrainFlow ./cmd/trainflow

Write-Host "[3/4] Building Windows runtime tool..."
$env:GOOS = "windows"
go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool.exe ./cmd/runtime-tool

Write-Host "[4/4] Building Linux runtime tool..."
$env:GOOS = "linux"
go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool ./cmd/runtime-tool

Copy-Item TrainFlow dist/trainflow-linux-amd64 -Force
Copy-Item TrainFlow.exe dist/trainflow-windows-amd64.exe -Force
Copy-Item TrainFlow_Runtime_Tool dist/trainflow-runtime-tool-linux-amd64 -Force
Copy-Item TrainFlow_Runtime_Tool.exe dist/trainflow-runtime-tool-windows-amd64.exe -Force

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
