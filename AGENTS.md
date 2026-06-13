# Agent Notes

This repository is a portable Go wrapper around a Python `sd-scripts` training stack. Keep changes scoped and prefer the existing Go/embedded-web patterns.

## Project Shape

- Main app: `cmd/trainflow`
- Runtime installer/updater: `cmd/runtime-tool`
- Trainer logic: `internal/trainer`
- Runtime operations: `internal/runtimeops`
- Embedded app UI: `cmd/trainflow/web`
- Python training stack: `training/sd-scripts`
- Local model/runtime folders are intentionally ignored and should not be committed.

## Build And Test

Use these checks before committing Go changes:

```bash
go test ./...
go build ./cmd/trainflow
go build ./cmd/runtime-tool
```

Use `gofmt` on touched Go files. The project has no package manager step for the embedded web UI.

## Training Profiles

TrainFlow supports two profile families:

- Anima: DiT, Qwen3, VAE, `networks.lora_anima`, 64px bucket steps.
- SDXL / Pony / Illustrious: checkpoint, `networks.lora`, 32px bucket steps, SDXL learning-rate fields.

Profile selection is part of user settings. Do not infer SDXL from filenames if the settings already specify an architecture.

## Auto Calc

`applyStableDefaults` should calculate a profile-aware starting point from dataset image count while preserving user intent.

- Preserve the selected optimizer.
- Bring the learning-rate math to that optimizer.
- Prodigy uses `learning_rate = 1.0` and exports `lr_scheduler = "constant"`.
- AdamW and AdamW8bit use the standard `1e-4` path and export the default cosine scheduler.
- Keep paths, project name, trigger word, prompts, and resume path intact.

Add or update focused tests in `internal/trainer/config_test.go` when changing defaults, config generation, profile behavior, or resume behavior.

## Git Hygiene

Do not commit generated runtimes, downloaded models, caches, or local output. Root binaries may be versioned in this repo, but avoid rebuilding them unless the user explicitly asks for release artifacts.
