# DaSiWa-MusubiTunerApp Integration Plan

**Target:** Embed MusubiTunerApp video normalization, parallel worker pool, and training sequencer into DaSiWa-TrainFlow for LTX23/WAN22 training workflows.

**Status:** Phase 1 (Scaffolding), Phase 2 (parallel normalizer backend), Phase 5 (LTX/WAN video prep UI), and Phase 6 (Sequencer) complete. See protocol below.

---

## PROTOCOL: What Is Already Done

### Phase 1: Backend Scaffolding & Struct Extensions
| Item | Status | File |
|------|--------|------|
| `ArchitectureLTX23` / `ArchitectureWAN22` constants | DONE | `profiles.go` |
| `normalizeSettings` arch-specific defaults | DONE | `profiles.go` |
| `Settings` struct expanded with video fields | DONE | `types.go` |
| `startMusubiSequenced` in `manager.go` | DONE | `manager.go` |
| `writeMusubiDatasetTOMLIfReady` hook | DONE, uses canonical output/scope path | `manager.go`, `util.go` |
| `output_path` / scope output path drives TOML/configs/checkpoints | DONE | `types.go`, `util.go`, `manager.go`, `index.html`, `app.js` |
| `musubi-dataset-toml` action in `StartDatasetPrep` | DONE | `manager.go` |
| `StartResponse.Step` field | DONE | `types.go` |
| `createMusubiDatasetTOML` / `buildMusubiCommand` | DONE | `musubi.go` |
| `validateMusubiSource` / `requiredMusubiFiles` | DONE | `musubi.go` |
| `buildNormalizeVideoCommand` / `runVideoNormalization` | DONE | `normalizer.go` |
| `sync-musubi-tuner.sh` script | DONE | `scripts/sync-musubi-tuner.sh` |
| All tests passing (27/27) | DONE | `*_test.go` |

### Phase 5: Web UI Enhancements
| Item | Status | File |
|------|--------|------|
| LTX/WAN selection swaps Dataset Prep from image tag/resize to video normalizer controls | DONE | `index.html`, `app.js` |
| Video normalizer flag controls exposed: `-w`, `-h`, `-fps`, `-len`, `-codec`, `-quality`, `-encoder_preset`, `-workers`, `-speed`, `-skip`, `-noaudio`, extra args | DONE | `index.html`, `app.js` |
| `Normalize Video` / `Build TOML` / `Cache Text` / `Cache Latents` buttons | DONE | `index.html`, `app.js` |
| Architecture switch sets video defaults | DONE | `app.js` |
| Video parameter/source edits mark cache dirty and tell user TOML/cache rebuild automatically on Start | DONE | `app.js` |
| Scope window routing for Musubi output | DONE | `app.js` |

### Phase 6: Training Step Sequencer (NEW - Just Implemented)
| Item | Status | File |
|------|--------|------|
| `runPipelineStep` helper | DONE | `manager.go` |
| `setPipelineStep` helper | DONE | `manager.go` |
| `isVideoFile` helper | DONE | `manager.go` |
| Full sequential pipeline: normalize → TOML → text cache → latent cache → train | DONE | `manager.go` |
| Cancellable context for pipeline | DONE | `manager.go` |
| `context` import added | DONE | `manager.go` |

### Missing Functions Added During Sequencer Implementation
| Item | Status | File |
|------|--------|------|
| `datasetPrepLabel` | DONE | `manager.go` |
| `validatePrepModels` | DONE | `manager.go` |
| `scanLogChunk` | DONE | `manager.go` |
| `isProgressLog` | DONE | `manager.go` |
| `isBlacklistedLog` | DONE | `manager.go` |
| `decodeJSON` | DONE | `manager.go` |

---

## Remaining Phases

### Phase 2: Parallel Video Normalizer Backend (DONE)
- `VideoParallelWorkers` added to settings and UI.
- `runVideoNormalization` now uses a goroutine worker pool and copies captions next to normalized videos.
- `/api/dataset/prep` `normalize-video` action starts FFmpeg normalization into `training/prepared/<project>-video`.

### Phase 3: Architecture-Specific Video Presets (PARTIAL)
- `normalizeSettings` sets arch-specific defaults
- Need: preset config files for LTX23/WAN22 video settings
- Need: UI presets dropdown in normalizer panel

### Phase 4: Enhanced Dataset TOML Generation (DONE)
- `createMusubiDatasetTOML` generates proper TOML
- `writeMusubiDatasetTOMLIfReady` auto-generates on save
- `/api/dataset/prep` handles `musubi-dataset-toml`, `musubi-cache-text`, and `musubi-cache-latents` actions
- `Start` for LTX/WAN automatically runs normalize → TOML → text cache → latent cache → train, so parameter/caption/video changes do not require manual cache buttons before training

### Phase 7: Build Integration (NOT STARTED)
- Standalone `normalize-video` build target in Makefile
- MusubiTunerApp CLI flags mapped to Go struct fields
- `normalize-video` binary buildable from TrainFlow repo

---

## Key Architecture Decisions

1. **Sequencer Pattern**: `startMusubiSequenced` runs steps sequentially using `context.WithCancel` for cancellation support. Each step uses `runPipelineStep` which blocks until completion.

2. **Video Normalization**: Only runs if dataset contains video files. Uses `isVideoFile` to detect `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`.

3. **Dataset Path Update**: After normalization, `s.DatasetPath` is updated to `preparedVideoDatasetPath` for subsequent cache/training steps.

4. **MusubiTunerApp Fork**: Vendored at `training/musubi-tuner/` (submodule of `GitHub/musubi-tuner/`). Uses dedicated Python runtime.

5. **TOML/Checkpoint Output Path**: canonical output root comes from `outputProject(root, settings)`. If `settings.OutputPath` / `output_path` is set by the scope/output window, LTX/WAN TOML is written to `<output_path>/configs` and Musubi training receives `--output_dir <output_path>` for LoRA checkpoints. Fallback remains `<root>/training/output/<projectName>`.

6. **Video Normalizer CLI Flags** (mapped from MusubiTunerApp):
   - `-input`, `-output` (dataset paths)
   - `-w`, `-h` (resolution)
   - `-fps` (frame rate)
   - `-len` (duration)
   - `-speed` (speed multiplier)
   - `-skip` (skip frames)
   - `-codec` (hevc_nvenc, h264_nvenc, av1_nvenc, libx264, libx265)
   - `-quality` (CQ value)
   - `-encoder_preset` (p6/medium/fast)
   - `-noaudio` (disable audio)
   - `-workers` (parallel count)

---

## Test Commands

```bash
# Run all trainer tests
go test ./internal/trainer -v

# Run specific test
go test ./internal/trainer -run TestStartDatasetPrepWritesMusubiDatasetTOML -v

# Run build
go build ./...
```

## Current Test Status: 27/27 PASSING

---

## Next Steps

1. **Phase 2**: Implement parallel video normalizer worker pool
2. **Phase 7**: Add Makefile build targets for standalone normalizer
3. **UI**: Add "Start Training Pipeline" button that triggers `startMusubiSequenced`
4. **Testing**: Integration tests for full pipeline (normalize → cache → train)
