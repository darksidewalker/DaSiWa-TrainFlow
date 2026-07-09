# Training Type Integration Notes

TrainFlow supports multiple training backends through small profile-specific adapters. Use this document when adding the next sd-scripts or Musubi training type.

## Current backend split

- sd-scripts profiles live in `internal/trainer/profiles.go` and use the classic config generation path.
- Musubi profiles use `trainingFamilyMusubi` and route through `internal/trainer/musubi.go`.
- The frontend type selector is data-attribute based: add one button in `cmd/trainflow/web/index.html`, accept it in `normalizeArchitecture()` in `cmd/trainflow/web/app.js`, and add color/style in `cmd/trainflow/web/styles.css`.

## Adding a future Musubi script

1. Sync or vendor the upstream/forked Musubi files under `training/musubi-tuner`.
2. Add a new architecture constant in `internal/trainer/profiles.go`.
3. Add a `profileFor()` case with:
   - `Family: trainingFamilyMusubi`
   - the train script name
   - required model path keys
   - bucket/resize divisor
   - `Video: true` only for video datasets
4. Extend `normalizeArchitecture()` and `normalizeSettings()` with safe defaults for the new profile.
5. Extend `validateModelPaths()` with profile-specific labels so users know which model is missing.
6. Add a command builder in `internal/trainer/musubi.go`, following the existing LTX/Wan/Krea2 builders:
   - cache text command
   - cache latents command
   - train command through `accelerate.commands.launch`
   - append shared options with `appendCommonMusubiTrainArgs()` when compatible
7. Add required source files to `requiredMusubiFiles` so runtime verification fails early with an actionable message.
8. Add or reuse TOML generation:
   - video profiles use `video_directory`, `target_frames`, `frame_extraction`
   - image profiles use `image_directory`, `cache_directory`, image resolution/bucketing
9. Add focused tests in `internal/trainer/musubi_test.go`:
   - profile shape
   - generated command contains the expected script and model arguments
   - generated dataset TOML has the correct image/video keys
10. Run:
   - `gofmt -w internal/trainer/*.go`
   - `go test ./...`
   - `go build -o /tmp/TrainFlow ./cmd/trainflow`
   - `go build -o /tmp/TrainFlow_Runtime_Tool ./cmd/runtime-tool`

## Frontend checklist for a new type

- Add a colored `.architecture-button` with `data-architecture="newtype"`.
- Add the same value to `normalizeArchitecture()`.
- Add a branch in `setArchitecture()` to set model labels and defaults.
- Tag the required path fields with `profile-newtype` classes.
- If it is video, update `isVideoArchitecture()`; if it is image, keep image prep visible.
- Keep defaults conservative and profile-specific; do not overwrite user-provided custom values unless they are stale defaults from another architecture.

## Musubi source and LTX 2.3 note

Upstream `kohya-ss/musubi-tuner` currently carries Krea2 and the newer Musubi image/video scripts, but the local TrainFlow integration still depends on LTX 2.3 entrypoints such as:

- `ltx2_train_network.py`
- `ltx2_cache_latents.py`
- `ltx2_cache_text_encoder_outputs.py`
- `src/musubi_tuner/networks/lora_ltx2.py`

If upstream main does not include those LTX 2.3 files, update/refresh the LTX 2.3 portion from the Musubi LTX fork instead of deleting the existing TrainFlow LTX files. In practice:

1. Sync Krea2/new upstream files from `https://github.com/kohya-ss/musubi-tuner`.
2. Preserve or refresh LTX 2.3 files from the LTX-capable Musubi fork/local source.
3. Verify both `ArchitectureLTX23` and `ArchitectureKrea2` tests before committing.

Do not run a blind `rsync --delete` from upstream main if it removes LTX 2.3 files that TrainFlow still calls. Either sync from the LTX-capable fork, or restore the deleted LTX files immediately before testing.

## Runtime script hints

`scripts/sync-musubi-tuner.sh` intentionally vendors Musubi source into `training/musubi-tuner`. When changing it, keep these safeguards:

- exclude `.git`, virtualenvs, caches, model weights, logs, and output folders
- do not copy datasets or generated training output
- after syncing, check `requiredMusubiFiles` against the vendored tree
- verify no LTX 2.3 entrypoints disappeared
- keep runtime dependency additions in `training/requirements-musubi-overlay.txt` when they are TrainFlow-specific overlays rather than upstream Musubi requirements
