# Unified TrainFlow + Musubi Runtime Fusion Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fuse the MusubiTunerApp workflow and the needed `musubi-tuner` Python sources into `DaSiWa-TrainFlow` as one enhanced TrainFlow app, preserving TrainFlow's advanced web GUI and using one consolidated Python runtime/venv instead of a separate Musubi venv.

**Architecture:** TrainFlow remains the single Go backend + embedded advanced web UI. Python trainer sources live side-by-side under `training/`: existing `training/sd-scripts` and new `training/musubi-tuner`. The runtime tool installs one app-owned runtime at `python_embeded/<os>` and installs both sd-scripts and Musubi dependencies into that same interpreter. Musubi execution never shells into an external clone's `.venv`; it runs with TrainFlow's single runtime and the vendored/copied `training/musubi-tuner` source tree.

**Tech Stack:** Go, embedded HTML/CSS/JS, one app-owned Python runtime (`python_embeded/linux` or `python_embeded/windows`), pip/uv installer path, `accelerate`, `sd-scripts`, vendored `musubi-tuner`, FFmpeg.

---

## Supersedes / Refines Previous Plan

This plan supersedes the earlier runtime strategy in:

`/home/darksidewalker/GitHub/DaSiWa-TrainFlow/.hermes/plans/2026-06-17_195018-integrate-musubi-ltx23-wan22.md`

Important changes requested by the user:

1. Do not rely on `/home/darksidewalker/GitHub/musubi-tuner/.venv`.
2. Do not create a second Musubi venv.
3. Port/copy the needed Musubi Python source files into TrainFlow under a `training/musubi-tuner` subfolder, like `training/sd-scripts`.
4. Consolidate dependencies into TrainFlow's existing `python_embeded/<os>` runtime.
5. Fuse the apps into one enhanced TrainFlow app while preserving TrainFlow's advanced web GUI.
6. Treat `DaSiWa-MusubiTunerApp` as a source of behavior/commands/presets, not as a second GUI to embed.

---

## Repository Facts Verified

- TrainFlow has one runtime folder ignored by git:
  - `python_embeded/`
- TrainFlow already vendors Python trainer code under:
  - `training/sd-scripts`
- TrainFlow sd-scripts dependencies are in:
  - `training/sd-scripts/requirements.txt`
- Existing requirements already overlap heavily with Musubi:
  - `accelerate==1.6.0`
  - `diffusers==0.32.1`
  - `ftfy==6.3.1`
  - `opencv-python==4.10.0.84`
  - `einops==0.7.0`
  - `bitsandbytes`
  - `toml==0.10.2`
  - `voluptuous==0.15.2`
  - `huggingface-hub==0.34.3`
  - `sentencepiece==0.2.1`
- Musubi dependencies are declared in:
  - `/home/darksidewalker/GitHub/musubi-tuner/pyproject.toml`
- Musubi adds/needs at least:
  - `av==14.0.1`
  - `pillow>=11.3.0`
  - `safetensors>=0.4.5`
  - `transformers==4.56.1`
  - `easydict==1.13`
  - optional: `gradio`, `fastapi`, `uvicorn`, `pyarrow` are not needed for fused TrainFlow first pass
- Runtime installer currently installs torch CUDA 13.0 first, then `training/sd-scripts/requirements.txt`, then editable sd-scripts, then UI/prep deps in:
  - `internal/runtimeops/runtimeops.go:InstallRequirements`
  - `internal/runtimeops/runtimeops.go:installTrainerDeps`
- Runtime tool actions are dispatched in:
  - `cmd/runtime-tool/main.go`
- The advanced GUI to preserve is:
  - `cmd/trainflow/web/index.html`
  - `cmd/trainflow/web/app.js`
  - `cmd/trainflow/web/styles.css`

---

## Target Final Layout

```text
DaSiWa-TrainFlow/
  cmd/
    trainflow/                 # single enhanced app
    runtime-tool/              # installs/repairs single shared runtime
  internal/
    trainer/                   # orchestrates sd-scripts + musubi-tuner
    runtimeops/                # installs one consolidated Python runtime
  python_embeded/              # ignored, one runtime only
    linux/                     # Linux venv created by runtime tool
    windows/                   # Windows embedded Python/runtime
  training/
    sd-scripts/                # existing trainer source
      requirements.txt
    musubi-tuner/              # new vendored/copied Musubi source tree
      pyproject.toml
      README.md
      ltx2_train_network.py
      ltx2_cache_latents.py
      ltx2_cache_text_encoder_outputs.py
      wan_train_network.py
      wan_cache_latents.py
      wan_cache_text_encoder_outputs.py
      src/musubi_tuner/...
    requirements-unified.txt   # optional generated/maintained consolidated dependency list
    output/                    # ignored
    prepared/                  # ignored, add if missing
```

Important policy:

- `training/musubi-tuner/.venv` must not exist.
- `training/musubi-tuner/uv.lock` can be copied only if useful for auditing, but it must not drive a separate venv.
- Musubi commands run with `pythonExecutable(root)` from TrainFlow.
- Musubi `cmd.Dir` should be `training/musubi-tuner`.
- `PYTHONPATH` should include:
  - `training/musubi-tuner/src`
  - `training/musubi-tuner`
  - existing sd-scripts paths if needed

---

## High-Level Strategy

### What to preserve from TrainFlow

Preserve the advanced TrainFlow web GUI and backend model:

- Embedded browser UI.
- Architecture selector.
- Live log streaming.
- Hardware monitor.
- Path picker.
- Runtime Tool launcher.
- Model status pill.
- Dataset prep panel.
- Output folder handling.
- Existing Anima and SDXL behavior.

Do not port the Fyne GUI itself from `DaSiWa-MusubiTunerApp`. Instead, port the useful behavior:

- Musubi command builders.
- Dataset TOML generation.
- LTX/Wan defaults and presets.
- FFmpeg video normalizer settings.
- Cache text/cache latents/train workflow.

### What to preserve from Musubi

Port/copy the Python source tree from `/home/darksidewalker/GitHub/musubi-tuner` into TrainFlow:

- root wrapper scripts for LTX/Wan:
  - `ltx2_train_network.py`
  - `ltx2_cache_latents.py`
  - `ltx2_cache_text_encoder_outputs.py`
  - `wan_train_network.py`
  - `wan_cache_latents.py`
  - `wan_cache_text_encoder_outputs.py`
- entire `src/musubi_tuner` package because the wrappers import it
- `pyproject.toml` for dependency source of truth/auditing
- `README.md`/license files if present and required by license

Do not port:

- Musubi's `.venv`
- cache folders
- downloaded models
- output datasets
- generated LoRAs
- `.git` metadata unless intentionally vendoring as a git subtree/submodule

### Dependency consolidation strategy

Use one TrainFlow-owned Python runtime:

- Linux: `python_embeded/linux/bin/python`
- Windows: `python_embeded/windows/python.exe`

Install all Python packages into that runtime.

Preferred first implementation:

1. Keep `training/sd-scripts/requirements.txt` for sd-scripts.
2. Add `training/musubi-tuner/pyproject.toml` by copying from the Musubi fork.
3. Add a small consolidated overlay requirements file:

```text
# training/requirements-musubi-overlay.txt
av==14.0.1
pillow>=11.3.0
transformers==4.56.1
easydict==1.13
```

4. Update runtime installer to install:

```text
Torch CUDA wheels
training/sd-scripts/requirements.txt
-e training/sd-scripts
training/requirements-musubi-overlay.txt
-e training/musubi-tuner
TrainFlow UI/prep deps
```

If editable install of Musubi fails because its `pyproject.toml` packaging expects a particular layout, fallback to setting `PYTHONPATH` and installing overlay deps only. But the runtime verify step should still import `musubi_tuner`.

Do not install Musubi's `cu124/cu128/cu130` extras directly from `pyproject.toml` because TrainFlow already centrally chooses CUDA/Torch wheels. Installing extras can downgrade/upgrade torch unexpectedly.

---

## Phase 1: Vendor/Port Musubi Source Into TrainFlow

### Task 1: Create source sync plan/script for `training/musubi-tuner`

**Objective:** Make the source copy reproducible instead of manually dragging files.

**Files:**
- Create: `scripts/sync-musubi-tuner.sh`
- Modify: `.gitignore`
- Create: `training/musubi-tuner/` contents by running the script during implementation

**Script behavior:**

Use `rsync` from the local fork into TrainFlow:

```bash
#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/home/darksidewalker/GitHub/musubi-tuner}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/training/musubi-tuner"

if [[ ! -d "$SRC/src/musubi_tuner" ]]; then
  echo "Source does not look like musubi-tuner: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'datasets/' \
  --exclude 'output/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude '*.safetensors' \
  --exclude '*.ckpt' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  "$SRC/" "$DEST/"

echo "Synced musubi-tuner source to $DEST"
```

**`.gitignore` updates:**

Keep vendored source tracked but generated Musubi local folders ignored:

```gitignore
training/musubi-tuner/.venv/
training/musubi-tuner/venv/
training/musubi-tuner/datasets/
training/musubi-tuner/output/
training/musubi-tuner/outputs/
training/musubi-tuner/logs/
training/prepared/
```

**Verification:**

```bash
bash scripts/sync-musubi-tuner.sh /home/darksidewalker/GitHub/musubi-tuner
test -f training/musubi-tuner/ltx2_train_network.py
test -f training/musubi-tuner/wan_train_network.py
test -d training/musubi-tuner/src/musubi_tuner
test ! -d training/musubi-tuner/.venv
```

### Task 2: Decide vendoring method before first commit

**Objective:** Avoid future update pain.

Recommended choices, in order:

1. **Simple vendored copy with sync script** for fastest implementation.
2. Git subtree if long-term updates from the fork should be traceable.
3. Git submodule only if the user accepts submodule complexity.

For this project, choose option 1 initially: simple vendored copy + sync script. It matches existing `training/sd-scripts` style.

---

## Phase 2: Consolidate Dependencies Into One Runtime

### Task 3: Add a Musubi overlay requirements file

**Objective:** Install only dependencies missing or stricter than existing sd-scripts requirements.

**Files:**
- Create: `training/requirements-musubi-overlay.txt`
- Modify later if import tests reveal missing packages

**Initial content:**

```text
# Additional dependencies required by training/musubi-tuner.
# Torch/torchvision/torchaudio are intentionally NOT listed here; runtimeops installs the selected CUDA wheels centrally.
# Musubi optional GUI/dashboard deps are intentionally excluded because TrainFlow provides the GUI.
av==14.0.1
pillow>=11.3.0
transformers==4.56.1
easydict==1.13
```

**Why this is safer than installing `musubi-tuner[cu130]`:**

- RuntimeOps already installs torch from `https://download.pytorch.org/whl/cu130`.
- Musubi extras can alter torch/torchvision versions.
- One app runtime should have one CUDA/Torch policy.

### Task 4: Update runtime installer to install Musubi into the same Python runtime

**Objective:** Make Runtime Tool install a single working environment for Anima, SDXL, LTX23, and WAN22.

**Files:**
- Modify: `internal/runtimeops/runtimeops.go`
- Test: `internal/runtimeops/runtimeops_test.go`

**Change `installTrainerDeps`:**

After sd-scripts install:

```go
musubiDir := filepath.Join(root, "training", "musubi-tuner")
musubiReq := filepath.Join(root, "training", "requirements-musubi-overlay.txt")
if dirExists(musubiDir) {
    log("Installing musubi-tuner overlay requirements into shared TrainFlow runtime...")
    if err := installer.installIn(root, "-r", musubiReq); err != nil {
        return err
    }
    log("Installing musubi-tuner editable package into shared TrainFlow runtime...")
    if err := installer.installIn(musubiDir, "-e", musubiDir); err != nil {
        log("Editable musubi-tuner install failed; continuing with PYTHONPATH runtime mode: " + err.Error())
    }
}
```

If helper functions like `dirExists` are not in `runtimeops`, add local equivalents.

**Important:** Do not create or activate `training/musubi-tuner/.venv` anywhere.

### Task 5: Add runtime verification for Musubi imports

**Objective:** Runtime Tool should prove the unified runtime can import both stacks.

**Files:**
- Modify: `internal/runtimeops/runtimeops.go`

Extend `Verify` command from only torch check to include:

```python
import accelerate, torch
import musubi_tuner
import transformers, diffusers, av, cv2, safetensors, sentencepiece, easydict
print('musubi_tuner import ok')
```

Because editable install might be skipped/fail, run with `PYTHONPATH` including `training/musubi-tuner/src` and `training/musubi-tuner` during verify.

**Expected Runtime Tool log:**

```text
torch <version> cuda <version> available True/False
musubi_tuner import ok
sd-scripts import ok
```

---

## Phase 3: Backend Uses Vendored Musubi, Not External Clone

### Task 6: Replace external Musubi path strategy with in-repo Musubi root

**Objective:** Stop treating `/home/darksidewalker/GitHub/musubi-tuner` as the runtime path after source sync.

**Files:**
- Create/modify: `internal/trainer/musubi.go`
- Modify: `internal/trainer/types.go`
- Modify: `internal/trainer/profiles.go`
- Test: `internal/trainer/musubi_test.go`

**Rule:**

```go
func musubiRoot(root string) string {
    return filepath.Join(root, "training", "musubi-tuner")
}
```

Do not keep a user setting that points to an external clone unless it is explicitly named as a developer override, for example:

```go
MusubiDevPath string `json:"musubi_dev_path"`
```

If added, it must be hidden behind an advanced/dev toggle and must still use TrainFlow's Python runtime, not the dev clone's venv.

**Musubi command runtime:**

```go
python := pythonExecutable(m.root)
cmd := exec.Command(python, args...)
cmd.Dir = musubiRoot(m.root)
cmd.Env = musubiTrainingEnv(m.root)
```

**Env:**

```go
func musubiTrainingEnv(root string) []string {
    musubiDir := musubiRoot(root)
    env := trainingEnv(musubiDir)
    env = append(env, "PYTHONUNBUFFERED=1")
    env = prependPythonPath(env,
        filepath.Join(musubiDir, "src"),
        musubiDir,
        filepath.Join(root, "training", "sd-scripts"),
    )
    return env
}
```

### Task 7: Validate vendored Musubi files

**Objective:** Fail early when the app is missing needed Python source files.

**Files:**
- Modify: `internal/trainer/musubi.go`
- Test: `internal/trainer/musubi_test.go`

**Required files:**

```go
var requiredMusubiFiles = []string{
    "pyproject.toml",
    "src/musubi_tuner/__init__.py",
    "ltx2_train_network.py",
    "ltx2_cache_latents.py",
    "ltx2_cache_text_encoder_outputs.py",
    "wan_train_network.py",
    "wan_cache_latents.py",
    "wan_cache_text_encoder_outputs.py",
}
```

**Validation message:**

```text
Musubi trainer source is missing from training/musubi-tuner. Run scripts/sync-musubi-tuner.sh /home/darksidewalker/GitHub/musubi-tuner, then run Runtime Tool -> Install.
```

---

## Phase 4: Fuse MusubiTunerApp Features Into Enhanced TrainFlow UI

### Task 8: Keep TrainFlow web UI as the only GUI

**Objective:** Preserve the advanced TrainFlow GUI and add Musubi panels into it.

**Files:**
- Modify: `cmd/trainflow/web/index.html`
- Modify: `cmd/trainflow/web/app.js`
- Modify: `cmd/trainflow/web/styles.css` only for layout polish

**Do not:**

- Embed Fyne.
- Launch the MusubiTunerApp binary.
- Add a second local web app.

**Add to existing architecture toggle:**

```html
<button type="button" class="architecture-button" data-architecture="ltx23">LTX 2.3</button>
<button type="button" class="architecture-button" data-architecture="wan22">Wan 2.2</button>
```

**Preserve existing panels:**

- Top status bar.
- Runtime/model status pills.
- Save/start/stop/output buttons.
- Hardware overlay.
- Preview gallery.
- Existing Dataset Prep panel.

**Add video-specific panels only when `ltx23` or `wan22` is active:**

- `Video Dataset`
  - width
  - height
  - fps
  - duration
  - target frames
  - frame extraction
  - repeats
  - caption extension
  - enable bucket
- `Musubi Advanced`
  - target epochs
  - mixed precision
  - CPU threads
  - blocks to swap
  - network module
  - network alpha
  - timestep sampling
  - FP8 base/scaled
  - SDPA
  - gradient checkpointing
  - persistent workers
  - metadata author/tags
  - extra cache/train args
- `Musubi Pipeline Actions`
  - Normalize Video
  - Write Dataset TOML
  - Cache Text
  - Cache Latents
  - Start Training uses existing top Start button

### Task 9: Port MusubiTunerApp presets as frontend/backend defaults

**Objective:** Make LTX23/WAN22 selector set the right defaults without replacing user edits unnecessarily.

**Files:**
- Modify: `internal/trainer/types.go`
- Modify: `internal/trainer/profiles.go`
- Modify: `cmd/trainflow/web/app.js`
- Test: `internal/trainer/config_test.go`

**LTX23 defaults:**

```text
optimizer: Prodigy
learning_rate: 1.0
lr_scheduler: constant
network_module: networks.lora_ltx2
network_rank: 128
network_alpha: 1
blocks_to_swap: 14
mixed_precision: bf16
ltx_version: 2.3
ltx_mode: video
ltx_version_check_mode: error
timestep_sampling: shifted_logit_normal
fp8_base: true
fp8_scaled: true
use_pinned_memory_for_block_swap: true
gradient_checkpointing: true
sdpa: true
persistent_data_loader_workers: true
video_target_frames: 1,65,129
```

**WAN22 defaults:**

```text
optimizer: Prodigy
learning_rate: 1.0
lr_scheduler: constant
network_module: networks.lora_wan
network_rank: 128
network_alpha: 1
wan_task: i2v-A14B
mixed_precision: bf16
timestep_sampling: shift
discrete_flow_shift: 5.0
gradient_checkpointing: true
force_v2_1_time_embedding: true
sdpa: true
persistent_data_loader_workers: true
video_width: 720
video_height: 1280
video_target_frames: 1,65,129
```

---

## Phase 5: Musubi Dataset TOML and Commands

### Task 10: Generate Musubi video dataset TOML in TrainFlow output configs

**Objective:** Use Musubi's video TOML shape while keeping all generated files under TrainFlow output.

**Files:**
- Create/modify: `internal/trainer/musubi_config.go`
- Test: `internal/trainer/config_test.go`

**Output path:**

```text
training/output/<project>/configs/<project>_musubi_dataset.toml
```

**TOML shape:**

```toml
[general]
resolution = [768, 512]
batch_size = 1
caption_extension = ".txt"
enable_bucket = true

[[datasets]]
video_directory = "/absolute/path/to/videos"
target_frames = [1, 65, 129]
frame_extraction = "full"
num_repeats = 1
```

### Task 11: Build commands against `training/musubi-tuner`

**Objective:** Run vendored Musubi scripts with TrainFlow Python.

**Files:**
- Create/modify: `internal/trainer/musubi.go`
- Test: `internal/trainer/musubi_test.go`

**LTX commands:**

```text
python ltx2_cache_text_encoder_outputs.py --dataset_config ... --ltx2_checkpoint ... --gemma_safetensors ... --ltx2_mode video --ltx_version 2.3 --mixed_precision bf16
python ltx2_cache_latents.py --dataset_config ... --ltx2_checkpoint ... --device cuda --vae_dtype bf16 --ltx2_mode video
python -m accelerate.commands.launch ... ltx2_train_network.py ...
```

**Wan commands:**

```text
python wan_cache_text_encoder_outputs.py --dataset_config ... --t5 ... --batch_size 16
python wan_cache_latents.py --dataset_config ... --vae ... --i2v
python -m accelerate.commands.launch ... wan_train_network.py --task i2v-A14B ...
```

**No external venv rule:** every command must start with `pythonExecutable(root)`, never `training/musubi-tuner/.venv/bin/python` and never `/home/darksidewalker/GitHub/musubi-tuner/.venv/bin/python`.

---

## Phase 6: Runtime Tool UX for Unified Install

### Task 12: Update Runtime Tool labels and actions

**Objective:** Make the Runtime Tool clearly install/verify all trainer stacks in one runtime.

**Files:**
- Modify: `cmd/runtime-tool/web/index.html`
- Modify: `cmd/runtime-tool/web/app.js`
- Modify: `cmd/runtime-tool/main.go` if actions change
- Modify: `internal/runtimeops/runtimeops.go`

**UI language changes:**

- Change `Install Requirements` to `Install Unified Training Runtime`.
- Add status text explaining it installs:
  - PyTorch CUDA wheels
  - sd-scripts deps
  - Musubi deps
  - TrainFlow prep/UI deps
- Verification should show:
  - Torch/CUDA status
  - sd-scripts import status
  - Musubi import status

### Task 13: Add a source check action optionally

**Objective:** Help user know whether `training/musubi-tuner` has been synced.

Potential Runtime Tool action:

```text
Check Trainer Sources
```

It should validate:

- `training/sd-scripts` exists
- `training/musubi-tuner/src/musubi_tuner` exists
- required LTX/Wan wrapper scripts exist

This can be folded into existing verify if keeping UI simpler.

---

## Phase 7: Video Normalizer Fusion

### Task 14: Port normalizer behavior into TrainFlow backend

**Objective:** Use the advanced TrainFlow UI and live logs to run FFmpeg normalization.

**Files:**
- Create: `internal/trainer/normalizer.go`
- Test: `internal/trainer/normalizer_test.go`
- Modify: `internal/trainer/manager.go`
- Modify: `cmd/trainflow/web/index.html`
- Modify: `cmd/trainflow/web/app.js`

**Rules:**

- Use TrainFlow's dataset path as input.
- Write output to `training/prepared/<project>-video`.
- Copy `.txt` captions beside converted videos.
- Start with CPU-safe FFmpeg defaults (`libx264`) instead of NVENC-only defaults.
- Allow advanced codec/extra args later.
- Stream FFmpeg output through existing TrainFlow log panel.

**First-pass command shape:**

```bash
ffmpeg -y -i input.mp4 \
  -vf "fps=24,scale=w=768:h=512:force_original_aspect_ratio=decrease,pad=768:512:(ow-iw)/2:(oh-ih)/2" \
  -t 5 \
  -an \
  -c:v libx264 -crf 19 -preset medium \
  output.mp4
```

---

## Phase 8: Tests and Verification

### Task 15: Unit tests

**Files:**
- `internal/trainer/musubi_test.go`
- `internal/trainer/config_test.go`
- `internal/trainer/normalizer_test.go`
- `internal/runtimeops/runtimeops_test.go`

**Required tests:**

1. `TestMusubiRootUsesTrainingSubfolder`
   - asserts root is `training/musubi-tuner`.
2. `TestMusubiPythonUsesTrainFlowRuntime`
   - command builder must use `pythonExecutable(root)`, not Musubi `.venv`.
3. `TestValidateMusubiSourceFiles`
   - missing wrapper produces actionable error.
4. `TestCreateMusubiDatasetTOML`
   - TOML includes `[general]`, `video_directory`, `target_frames`.
5. `TestBuildLTX23Commands`
   - command args include `ltx2_train_network.py`, `networks.lora_ltx2`, `--fp8_base`.
6. `TestBuildWAN22Commands`
   - command args include `wan_train_network.py`, `--task i2v-A14B`, `networks.lora_wan`.
7. `TestUnifiedRequirementsOverlayPresent`
   - runtime install tests know `training/requirements-musubi-overlay.txt` exists.

### Task 16: Runtime verification commands

Run after implementing source sync and installer changes:

```bash
go test ./...
go build ./cmd/trainflow
go build ./cmd/runtime-tool
```

Then run Runtime Tool:

```bash
./runtime-tool
```

Use UI actions:

1. Install Unified Training Runtime.
2. Verify.

Expected verify should import:

```text
torch
accelerate
transformers
diffusers
musubi_tuner
av
cv2
easydict
```

Manual Python check using app runtime:

```bash
python_embeded/linux/bin/python -c "import torch, accelerate, transformers, diffusers, av, cv2, easydict; import musubi_tuner; print('unified runtime ok')"
```

### Task 17: UI smoke test

```bash
./trainflow
```

In browser:

1. Select `Anima`; existing fields remain intact.
2. Select `SDXL`; existing SDXL behavior remains intact.
3. Select `LTX 2.3`; Musubi/video fields appear.
4. Select `Wan 2.2`; labels change to Wan/T5/VAE wording.
5. Save settings and refresh; selection persists.
6. Click `Write Musubi TOML`; config appears under `training/output/<project>/configs`.
7. Click `Cache Text`; command runs from `training/musubi-tuner` using `python_embeded`.
8. Click `Cache Latents`; same.
9. Start training and stop after command/log validation if no full run is desired.

---

## Files Likely to Change

New/modified source management:

- `scripts/sync-musubi-tuner.sh` new
- `.gitignore`
- `training/musubi-tuner/**` new vendored source copy
- `training/requirements-musubi-overlay.txt` new

Runtime:

- `internal/runtimeops/runtimeops.go`
- `internal/runtimeops/runtimeops_test.go`
- `cmd/runtime-tool/main.go`
- `cmd/runtime-tool/web/index.html`
- `cmd/runtime-tool/web/app.js`

Trainer backend:

- `internal/trainer/types.go`
- `internal/trainer/profiles.go`
- `internal/trainer/config.go`
- `internal/trainer/defaults.go`
- `internal/trainer/manager.go`
- `internal/trainer/routes.go`
- `internal/trainer/musubi.go` new
- `internal/trainer/musubi_config.go` new optional split
- `internal/trainer/normalizer.go` new
- `internal/trainer/config_test.go`
- `internal/trainer/manager_test.go`
- `internal/trainer/musubi_test.go` new
- `internal/trainer/normalizer_test.go` new

TrainFlow web UI:

- `cmd/trainflow/web/index.html`
- `cmd/trainflow/web/app.js`
- `cmd/trainflow/web/styles.css`

Docs:

- `README.md`
- `RUNTIME_UPGRADE.md`
- `AGENTS.md`

---

## Dependency Conflict Notes

Known potential conflicts to watch:

- Existing TrainFlow has `transformers>=4.54.1`; Musubi pins `transformers==4.56.1`.
  - Use `transformers==4.56.1` if sd-scripts tests/imports pass.
- Existing TrainFlow has `safetensors==0.4.5`; Musubi says `safetensors>=0.4.5`.
  - Keep `0.4.5` unless Musubi import/runtime needs newer.
- Existing TrainFlow installs `gradio==6.14.0`; Musubi optional GUI wants `<6.0.0`.
  - Do not install Musubi `gui` extra. TrainFlow supplies the UI.
- Musubi has torch extras for CUDA 12.4/12.8/13.0.
  - Do not install them. RuntimeOps owns torch installation.
- `pillow>=11.3.0` may upgrade existing Pillow.
  - Accept if tests pass; image prep should be verified.

---

## Acceptance Criteria

- `training/musubi-tuner` exists in TrainFlow and contains the needed Musubi wrappers and `src/musubi_tuner` package.
- No `training/musubi-tuner/.venv` is created or required.
- Runtime Tool installs sd-scripts and Musubi dependencies into the same `python_embeded/<os>` runtime.
- Runtime Tool verify imports `musubi_tuner` from the shared runtime.
- TrainFlow backend runs Musubi commands with `pythonExecutable(root)` and `cmd.Dir = training/musubi-tuner`.
- LTX23 and WAN22 are selectable in the existing advanced TrainFlow GUI.
- Existing Anima/SDXL workflows still pass tests and manual smoke checks.
- Musubi dataset TOML is generated under `training/output/<project>/configs`.
- Cache text/cache latents/train actions stream logs into the existing TrainFlow log panel.
- Video normalization is available from the enhanced TrainFlow UI.
- `go test ./...` passes.
- `go build ./cmd/trainflow` passes.
- `go build ./cmd/runtime-tool` passes.

---

## Open Questions

1. Should `training/musubi-tuner` be committed as a full vendored source copy, or should it be regenerated by `scripts/sync-musubi-tuner.sh` during release packaging?
2. Should Runtime Tool expose a button to sync/update Musubi source from a local path, or should syncing stay a developer-only script?
3. Should the initial unified runtime keep CUDA 13.0 as TrainFlow currently does, or should the Runtime Tool offer CUDA 12.8/13.0 selection because Musubi upstream explicitly lists cu128 and cu130 extras?
4. Should advanced Musubi fields be visible directly or tucked into a collapsible `Advanced Musubi` section by default?
5. Should `qwen_path` be renamed in the UI only, or should backend add a generic `text_encoder_path` for LTX/Wan clarity?
