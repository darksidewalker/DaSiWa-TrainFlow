# DaSiWa-TrainFlow Code Review

## Summary

- **Build**: Both `cmd/trainflow` and `cmd/runtime-tool` compile successfully
- **Tests**: 38/38 passing across 7 packages
- **Vet**: `go vet ./...` clean — no issues
- **Format**: `gofmt` clean across the entire project

---

## 1. Dead Code — CORRECTED: None Found

**CORRECTION**: The initial review incorrectly flagged `jsonContains`, `analyzeDatasetResolution`, and `countDatasetVideos` as dead code. All three are actively called:

- `jsonContains` — `manager.go:61`, `manager.go:64` (settings load migration)
- `analyzeDatasetResolution` — `manager.go:146` (training start resolution detection)
- `countDatasetVideos` — `defaults.go:111` (video auto-calc)

No safe dead code removal identified at this time.

---

## 2. Code Duplication (Should Deduplicate)

### `pythonExecutable` — duplicated 2×
- `internal/trainer/util.go:256` (34 lines)
- `internal/runtimeops/runtimeops.go:667` (34 lines, identical logic)

Both scan the same candidate paths for Windows/Linux. Should be a shared utility in a common package or exported from `runtimeops`.

### `fileExists` — duplicated 2× with subtle difference
- `internal/trainer/util.go:246` — `err == nil && !info.IsDir()` (allows empty files)
- `internal/modelops/modelops.go:230` — `err == nil && !info.IsDir() && info.Size() > 0` (rejects empty files)

The modelops version rejects empty files, which could silently skip valid placeholder files. Should be unified with a clear contract.

### `modelOverrides` — duplicated 2×
- `internal/trainer/routes.go:233` — reads from `Settings` struct
- `cmd/runtime-tool/main.go:161` — reads from `training/settings.json` directly

Both produce the same `map[string]string` for model path overrides. The runtime-tool version should use the trainer package's function instead of duplicating the JSON parsing.

---

## 3. Security & Robustness

### `/samples/` path traversal — incomplete protection (routes.go:205-218)
Current check only rejects components containing `..`. Does NOT validate that the resolved path stays within the sample directory. An attacker could use encoded path tricks or symlinks to escape.

**Fix**: Use `filepath.Clean()` and verify the resolved path starts with the sample directory prefix.

### WebSocket hub — no ping/pong keep-alive (hub.go)
The hub reads a single byte and discards it in a loop. It never sends or handles ping/pong frames. Long-lived connections will eventually time out on proxies/load balancers.

**Fix**: Add periodic ping frames and handle pong responses.

### `download()` — no Content-Length validation (modelops.go:174-228)
Large model downloads (18+ GB) have no integrity check. A corrupted download would be silently accepted.

**Fix**: Consider adding SHA-256 verification for model downloads.

---

## 4. Clean Code Issues

### `StartDatasetPrep` — too long, too many responsibilities (manager.go:497-642)
145 lines handling 5+ distinct actions (normalize-video, musubi-cache-text, musubi-cache-latents, musubi-dataset-toml, tag, resize, all). Each action should be its own method.

### `applyStableDefaultsWithVRAM` — too long (defaults.go:104-202)
98 lines with complex video vs image branching. The video branch (lines 139-176) should be extracted to its own function.

### `installOptionalFlashAttention` — too long (runtimeops.go:155-246)
91 lines of nested conditionals. Each fallback strategy (env var, release wheel, community wheel, PyPI, source build) should be its own function.

### `detectLargestGPUMemoryMB` — only used once (defaults.go:227-240)
Only called from `routes.go:58`. Consider inlining or moving to the call site.

### `sampleDir` parameter — dead propagation (manager.go)
`pipeLogs` and `waitForExit` accept `sampleDir` but most callers pass `""`. The Musubi pipeline passes it for training but not for cache steps. This makes the parameter confusing.

---

## 5. Test Coverage Gaps

| Package | Tests | Coverage |
|---------|-------|----------|
| `internal/trainer` | 38 tests | Good — covers config, defaults, musubi, routes, manager |
| `internal/hwmon` | 0 | Missing — GPU parsing logic untested |
| `internal/modelops` | 0 | Missing — file checking, download logic untested |
| `internal/process` | 0 | Missing — platform-specific process control untested |
| `internal/trainer/hub.go` | 1 test | Missing WebSocket frame tests |
| `internal/trainer/pathbrowser.go` | 0 | Missing — path resolution untested |

---

## 6. Minor Issues

- ~~`musubi_test.go` needs `gofmt`~~ — DONE: fixed.
- `vramTier()` thresholds (app.js:304-308): 16GB → "high", 12GB → "medium", <12GB → "low". The preset table comment says "Low VRAM (~12GB-16GB)" but 12-16GB actually maps to "medium" tier. Thresholds should be adjusted or the comment clarified.
- `flashAttentionReleaseVersion()` makes a network call to PyPI that could block for 20 seconds. Consider making this async or cacheable.
- `recommendedGradAccum` hardcodes VRAM thresholds (30000, 20000) that should match the preset table tiers.

---

## Prioritized Fix Plan

### Phase 1: Quick Wins (no behavior change)
1. ~~Remove dead code~~ — CORRECTED: no dead code found.
2. Run `gofmt -w` on `musubi_test.go`

### Phase 2: Deduplication (low risk)
3. ~~Consolidate `pythonExecutable` into a shared location~~ — DONE: `process.PythonExecutable` in `internal/process/python.go`.
4. ~~Unify `fileExists` with clear contract (keep size > 0 check for models)~~ — DONE: `process.FileExists` (allows empty) and `process.FileExistsNonEmpty` (rejects empty). All 24 callers updated. Local duplicates removed from `util.go` and `modelops.go`.
5. ~~Have runtime-tool use trainer's `modelOverrides` logic~~ — DONE: runtime-tool now loads `trainer.Settings` and calls `trainer.ModelOverrides(settings)`.

### Phase 3: Security & Robustness
6. ~~Fix `/samples/` path traversal with `filepath.Clean` + prefix validation~~ — DONE: `routes.go` now cleans the path and verifies it stays within the samples root directory.
7. ~~Add WebSocket ping/pong keep-alive~~ — DONE: `hub.go` now sends ping frames every 25s, responds to client pings with pong, and properly parses WebSocket frames (replacing the broken 1-byte read loop).
8. ~~Add download integrity verification (SHA-256)~~ — DONE: `download()` now computes SHA-256 during transfer via `io.TeeReader`-style hash.Write, logs the hash, and verifies against an optional `Hash` field in `ModelFile`.

### Phase 4: Clean Code Refactoring
9. Extract `StartDatasetPrep` actions into separate methods
10. Extract video branch from `applyStableDefaultsWithVRAM`
11. Break `installOptionalFlashAttention` into strategy functions
12. Clarify `sampleDir` parameter usage

### Phase 5: Test Coverage
13. Add tests for `hwmon` GPU parsing
14. Add tests for `modelops` file checking
15. Add tests for `pathbrowser` path resolution
16. Add WebSocket frame tests for hub
