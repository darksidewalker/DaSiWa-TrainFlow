# Musubi LTX23/WAN22 Integration Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Integrate the local `DaSiWa-MusubiTunerApp` workflow into `DaSiWa-TrainFlow` so TrainFlow can select and run LTX 2.3 and Wan 2.2 video LoRA training through the local `musubi-tuner` fork.

**Architecture:** Keep TrainFlow as the Go HTTP backend plus embedded static web UI. Add Musubi as a separate profile family beside the existing `sd-scripts` Anima/SDXL profiles, with a small Musubi command builder layer for cache-text, cache-latents, and train commands. Reuse source behavior from `/home/darksidewalker/GitHub/DaSiWa-MusubiTunerApp/main.go` and scripts/dependencies from `/home/darksidewalker/GitHub/musubi-tuner`.

**Tech Stack:** Go, `net/http`, embedded HTML/CSS/JS, local Python runtime, `accelerate`, Musubi scripts (`ltx2_*`, `wan_*`), FFmpeg for video normalization.

---

## Planning Inputs Used

- Gemini CLI was run in plan/read-only mode with Google Code Assist OAuth enabled:
  - `GOOGLE_GENAI_USE_GCA=true npx --yes @google/gemini-cli --approval-mode plan --skip-trust --include-directories /home/darksidewalker/GitHub/DaSiWa-MusubiTunerApp,/home/darksidewalker/GitHub/musubi-tuner ...`
- Gemini produced a high-level integration outline. I then checked the actual repositories and refined this into an implementation-ready plan.
- Important source files inspected:
  - TrainFlow backend: `internal/trainer/types.go`, `profiles.go`, `config.go`, `manager.go`, `routes.go`, `defaults.go`
  - TrainFlow UI: `cmd/trainflow/web/index.html`, `cmd/trainflow/web/app.js`
  - MusubiTunerApp source: `DaSiWa-MusubiTunerApp/main.go`, especially command builders and dataset TOML generation
  - Musubi fork: `musubi_wizard.py`, root wrapper scripts, `src/musubi_tuner/wan_*`, `ltx2_*`

---

## Current Context

### TrainFlow today

- Existing profile selector supports:
  - `anima`
  - `sdxl`
- Backend profile selection is in:
  - `internal/trainer/profiles.go`
- Settings are persisted as JSON in:
  - `training/settings.json`
- Training starts through:
  - `internal/trainer/manager.go:Start`
- TrainFlow currently assumes training scripts live under:
  - `training/sd-scripts`
- Existing training command shape:
  - Python executable from `pythonExecutable(root)`
  - `-m accelerate.commands.launch`
  - generated bootstrap script
  - `--config_file <training.toml>`
  - `--dataset_config <dataset.toml>`
- Existing dataset prep only supports:
  - `tag`
  - `resize`
  - `all`
- Existing UI selector is button based in:
  - `cmd/trainflow/web/index.html:15-18`
  - `cmd/trainflow/web/app.js:142-169`

### MusubiTunerApp behavior to port

From `/home/darksidewalker/GitHub/DaSiWa-MusubiTunerApp/main.go`:

- Pipeline settings include:
  - `MusubiDir`
  - `ModelPath`
  - `TextEncoderPath`
  - `DatasetConfig`
  - `OutputDir`
  - `Name`
  - `TargetEpochs`
  - `CacheSettings`
  - `TrainingSettings`
  - `NormalizerSettings`
  - `DatasetTOMLSettings`
- LTX 2.3 defaults include:
  - checkpoint: `ltx-2.3-22b-dev.safetensors`
  - text encoder: `gemma_3_12B_it_fp8_e4m3fn.safetensors`
  - `optimizer_type = prodigyopt.Prodigy`
  - `learning_rate = 1.0`
  - `lr_scheduler = constant` in `musubi_wizard.py` but source defaults currently say cosine; prefer constant for Prodigy consistency with TrainFlow AGENTS notes
  - `timestep_sampling = shifted_logit_normal`
  - `ltx_version = 2.3`
  - `ltx2_mode = video`
  - `fp8_base = true`
  - `fp8_scaled = true`
  - `blocks_to_swap = 14`
  - `use_pinned_memory_for_block_swap = true`
  - `gradient_checkpointing = true`
  - `sdpa = true`
  - `network_module = networks.lora_ltx2`
  - `network_dim = 128`
  - `network_alpha = 1`
- LTX scripts:
  - `ltx2_cache_latents.py`
  - `ltx2_cache_text_encoder_outputs.py`
  - `ltx2_train_network.py`
- Wan scripts from `musubi_wizard.py`:
  - `src/musubi_tuner/wan_cache_text_encoder_outputs.py` or root `wan_cache_text_encoder_outputs.py`
  - `src/musubi_tuner/wan_cache_latents.py` or root `wan_cache_latents.py`
  - `src/musubi_tuner/wan_train_network.py` or root `wan_train_network.py`
- Wan 2.2 command defaults from `musubi_wizard.py` include:
  - `--task i2v-A14B`
  - `--optimizer_type prodigyopt.Prodigy`
  - `--learning_rate 1.0`
  - `--lr_scheduler constant`
  - `--timestep_sampling shift`
  - `--discrete_flow_shift 5.0`
  - `--dit <model>`
  - `--t5 <text encoder>`
  - `--vae <vae>` for cache latents
  - `--gradient_checkpointing`
  - `--force_v2_1_time_embedding`
  - `--sdpa`
  - `--network_module networks.lora_wan`
  - `--network_dim 128`
  - `--network_alpha 1`
  - `--persistent_data_loader_workers`
- Musubi dataset TOML shape from `datasetConfigFromSettings`:
  - `[general]`
  - `resolution = [width, height]`
  - `batch_size = <n>`
  - `caption_extension = ".txt"`
  - `enable_bucket = true/false`
  - `[[datasets]]`
  - `video_directory = "..."`
  - `target_frames = [1, 65, 129]`
  - `frame_extraction = "full"`
  - `num_repeats = <n>`

---

## Proposed Approach

Do not try to force Musubi into the existing `sd-scripts` config-file-only flow. Add explicit Musubi profile handling:

1. Extend TrainFlow settings with Musubi/video-specific fields.
2. Add `ltx23` and `wan22` architectures to `profileFor`.
3. Add a Musubi runtime root separate from `training/sd-scripts`:
   - default local source: `/home/darksidewalker/GitHub/musubi-tuner`
   - portable in-repo target: `training/musubi-tuner`
4. Generate Musubi dataset TOML separately from existing image dataset TOML.
5. Build Musubi commands as argv slices, not only TOML config files, because MusubiTunerApp and `musubi_wizard.py` use CLI flags heavily.
6. Add UI controls for the video architecture selector and only show Musubi fields when `ltx23` or `wan22` is active.
7. Add dataset prep actions for:
   - `normalize-video`
   - `musubi-dataset-toml`
   - `musubi-cache-text`
   - `musubi-cache-latents`
8. Keep training start as the final action, using cached outputs where Musubi expects them.

---

## Phase 1: Model the New Architectures

### Task 1: Add architecture constants and profile metadata

**Objective:** Teach TrainFlow that `ltx23` and `wan22` are valid architectures.

**Files:**
- Modify: `internal/trainer/profiles.go`
- Test: `internal/trainer/config_test.go` or new `internal/trainer/profiles_test.go`

**Implementation notes:**

Add constants:

```go
const (
    ArchitectureAnima = "anima"
    ArchitectureSDXL  = "sdxl"
    ArchitectureLTX23 = "ltx23"
    ArchitectureWAN22 = "wan22"
)
```

Extend `trainingProfile` with Musubi-specific metadata instead of overloading `Script` only:

```go
type trainingFamily string

const (
    trainingFamilySDScripts trainingFamily = "sd-scripts"
    trainingFamilyMusubi    trainingFamily = "musubi"
)

type trainingProfile struct {
    Architecture      string
    Label             string
    Family            trainingFamily
    Script            string
    InferenceScript   string
    RequiredPathNames []string
    BucketStep        int
    ResizeDivisor     int
    Video             bool
}
```

Add `profileFor` cases:

```go
case ArchitectureLTX23:
    return trainingProfile{
        Architecture:      ArchitectureLTX23,
        Label:             "LTX 2.3 Video LoRA",
        Family:            trainingFamilyMusubi,
        Script:            "ltx2_train_network.py",
        RequiredPathNames: []string{"checkpoint_path", "qwen_path"}, // checkpoint + Gemma text encoder
        BucketStep:        16,
        ResizeDivisor:     16,
        Video:             true,
    }
case ArchitectureWAN22:
    return trainingProfile{
        Architecture:      ArchitectureWAN22,
        Label:             "Wan 2.2 Video LoRA",
        Family:            trainingFamilyMusubi,
        Script:            "wan_train_network.py",
        RequiredPathNames: []string{"dit_path", "qwen_path", "vae_path"}, // DiT + T5 + VAE
        BucketStep:        16,
        ResizeDivisor:     16,
        Video:             true,
    }
```

Update `normalizeArchitecture`:

```go
switch strings.ToLower(strings.TrimSpace(value)) {
case ArchitectureSDXL:
    return ArchitectureSDXL
case ArchitectureLTX23:
    return ArchitectureLTX23
case ArchitectureWAN22:
    return ArchitectureWAN22
case ArchitectureAnima:
    fallthrough
default:
    return ArchitectureAnima
}
```

**Tests:**

```go
func TestNormalizeArchitectureAcceptsVideoProfiles(t *testing.T) {
    if got := normalizeArchitecture("ltx23"); got != ArchitectureLTX23 { t.Fatalf("got %q", got) }
    if got := normalizeArchitecture("wan22"); got != ArchitectureWAN22 { t.Fatalf("got %q", got) }
}

func TestProfileForVideoProfiles(t *testing.T) {
    ltx := profileFor(Settings{Architecture: ArchitectureLTX23})
    if ltx.Family != trainingFamilyMusubi || !ltx.Video || ltx.Script != "ltx2_train_network.py" { t.Fatalf("bad ltx profile: %#v", ltx) }
    wan := profileFor(Settings{Architecture: ArchitectureWAN22})
    if wan.Family != trainingFamilyMusubi || !wan.Video || wan.Script != "wan_train_network.py" { t.Fatalf("bad wan profile: %#v", wan) }
}
```

**Verify:**

```bash
go test ./internal/trainer -run 'TestNormalizeArchitecture|TestProfileForVideoProfiles' -v
```

---

## Phase 2: Extend Settings Safely

### Task 2: Add Musubi/video fields to `Settings`

**Objective:** Persist enough state for LTX/Wan training without breaking existing settings JSON.

**Files:**
- Modify: `internal/trainer/types.go`
- Modify: `internal/trainer/profiles.go` (`normalizeSettings`)
- Test: `internal/trainer/config_test.go`

**Fields to add:**

```go
// Musubi runtime / command fields
MusubiPath                    string `json:"musubi_path"`
TextEncoderPath               string `json:"text_encoder_path"` // optional alias if qwen_path should stay Qwen-specific later
TargetEpochs                  int    `json:"target_epochs"`
MixedPrecision                string `json:"mixed_precision"`
NumCPUThreads                 int    `json:"num_cpu_threads"`

// Video dataset fields
VideoWidth                    int    `json:"video_width"`
VideoHeight                   int    `json:"video_height"`
VideoFPS                      int    `json:"video_fps"`
VideoDuration                 string `json:"video_duration"`
VideoTargetFrames             string `json:"video_target_frames"` // e.g. "1,65,129"
VideoFrameExtraction          string `json:"video_frame_extraction"`
VideoNumRepeats               int    `json:"video_num_repeats"`
VideoCaptionExtension         string `json:"video_caption_extension"`
VideoEnableBucket             bool   `json:"video_enable_bucket"`

// Musubi optimization/training fields
LTXVersion                    string `json:"ltx_version"`
LTXMode                       string `json:"ltx_mode"`
LTXVersionCheckMode           string `json:"ltx_version_check_mode"`
WanTask                       string `json:"wan_task"`
BlocksToSwap                  int    `json:"blocks_to_swap"`
NetworkAlpha                  int    `json:"network_alpha"`
NetworkModule                 string `json:"network_module"`
TimestepSampling              string `json:"timestep_sampling"`
DiscreteFlowShift             string `json:"discrete_flow_shift"`
FP8Base                       bool   `json:"fp8_base"`
FP8Scaled                     bool   `json:"fp8_scaled"`
SDPA                          bool   `json:"sdpa"`
GradientCheckpointing         bool   `json:"gradient_checkpointing"`
UsePinnedMemoryForBlockSwap   bool   `json:"use_pinned_memory_for_block_swap"`
PersistentDataLoaderWorkers   bool   `json:"persistent_data_loader_workers"`
SaveStateOnTrainEnd           bool   `json:"save_state_on_train_end"`
MetadataAuthor                string `json:"metadata_author"`
MetadataTags                  string `json:"metadata_tags"`
ExtraTrainArgs                string `json:"extra_train_args"`
ExtraCacheTextArgs            string `json:"extra_cache_text_args"`
ExtraCacheLatentsArgs         string `json:"extra_cache_latents_args"`
```

**Default values in `DefaultSettings`:**

Use non-invasive defaults shared by both video profiles:

```go
MusubiPath:                  filepath.Join(home, "GitHub", "musubi-tuner"),
TargetEpochs:                6,
MixedPrecision:              "bf16",
NumCPUThreads:               8,
VideoWidth:                  768,
VideoHeight:                 512,
VideoFPS:                    24,
VideoDuration:               "5",
VideoTargetFrames:           "1,65,129",
VideoFrameExtraction:        "full",
VideoNumRepeats:             1,
VideoCaptionExtension:       ".txt",
VideoEnableBucket:           true,
LTXVersion:                  "2.3",
LTXMode:                     "video",
LTXVersionCheckMode:         "error",
WanTask:                     "i2v-A14B",
BlocksToSwap:                14,
NetworkAlpha:                1,
NetworkModule:               "",
TimestepSampling:            "",
DiscreteFlowShift:           "",
FP8Base:                     true,
FP8Scaled:                   true,
SDPA:                        true,
GradientCheckpointing:       true,
UsePinnedMemoryForBlockSwap: true,
PersistentDataLoaderWorkers: true,
SaveStateOnTrainEnd:         true,
MetadataAuthor:              "darksidewalker",
```

**Normalize per architecture:**

In `normalizeSettings`, after `s.Architecture` is normalized:

- For `ltx23`:
  - `Optimizer = "Prodigy"` if empty
  - `LearningRate = "1.0"` for Prodigy
  - `NetworkRank = 128` if missing/low default not user-settable yet
  - `NetworkAlpha = 1`
  - `NetworkModule = "networks.lora_ltx2"`
  - `TimestepSampling = "shifted_logit_normal"`
  - `DiscreteFlowShift = ""`
  - `LTXVersion = "2.3"`
  - `LTXMode = "video"`
- For `wan22`:
  - `Optimizer = "Prodigy"` if empty
  - `LearningRate = "1.0"` for Prodigy
  - `NetworkRank = 128`
  - `NetworkAlpha = 1`
  - `NetworkModule = "networks.lora_wan"`
  - `TimestepSampling = "shift"`
  - `DiscreteFlowShift = "5.0"`
  - `WanTask = "i2v-A14B"`

**Risk:** Existing `QwenPath` is currently Qwen3-specific for Anima but can function as a generic text encoder path for Wan/LTX. The UI label must change per profile. If this feels too confusing, add `TextEncoderPath` now and migrate UI to that for video profiles.

**Tests:**

- JSON settings lacking new fields still load and get sane defaults.
- `normalizeSettings(Settings{Architecture: "ltx23", Optimizer: "Prodigy"})` yields `LearningRate == "1.0"`, `NetworkModule == "networks.lora_ltx2"`, `NetworkAlpha == 1`.
- `normalizeSettings(Settings{Architecture: "wan22"})` yields `WanTask == "i2v-A14B"`, `NetworkModule == "networks.lora_wan"`.

---

## Phase 3: Resolve Musubi Runtime Location

### Task 3: Add Musubi path resolver and validation

**Objective:** Let TrainFlow use the provided local fork now, while supporting a portable in-repo `training/musubi-tuner` later.

**Files:**
- Create: `internal/trainer/musubi.go`
- Modify: `internal/trainer/profiles.go`
- Test: `internal/trainer/musubi_test.go`

**Functions:**

```go
func musubiRoot(root string, s Settings) string {
    candidates := []string{
        strings.TrimSpace(s.MusubiPath),
        filepath.Join(root, "training", "musubi-tuner"),
        filepath.Join(userHomeOrEmpty(), "GitHub", "musubi-tuner"),
    }
    for _, c := range candidates {
        if c != "" && dirExists(c) {
            return c
        }
    }
    if strings.TrimSpace(s.MusubiPath) != "" {
        return strings.TrimSpace(s.MusubiPath)
    }
    return filepath.Join(root, "training", "musubi-tuner")
}

func validateMusubiRuntime(root string, s Settings) error {
    base := musubiRoot(root, s)
    required := []string{"pyproject.toml", "src/musubi_tuner"}
    for _, rel := range required {
        if !exists(filepath.Join(base, rel)) {
            return fmt.Errorf("musubi-tuner runtime missing %s under %s", rel, base)
        }
    }
    return nil
}
```

**Notes:**

- Do not vendor/copy `/home/darksidewalker/GitHub/musubi-tuner` in this phase.
- If later desired, add a runtime-tool action to clone/sync the fork into `training/musubi-tuner`.

**Verify:**

```bash
go test ./internal/trainer -run TestMusubi -v
```

---

## Phase 4: Generate Musubi Dataset TOML

### Task 4: Add video file counting and target frame parsing

**Objective:** Support video datasets and exposure calculations.

**Files:**
- Modify: `internal/trainer/config.go` or create `internal/trainer/video_dataset.go`
- Test: `internal/trainer/config_test.go`

**Implement:**

```go
var validVideoExtensions = map[string]bool{
    ".mp4": true, ".mkv": true, ".mov": true, ".webm": true, ".avi": true, ".m4v": true,
}

func validVideoExt(name string) bool {
    return validVideoExtensions[strings.ToLower(filepath.Ext(name))]
}

func countDatasetVideos(datasetPath string) int { ... filepath.WalkDir ... }

func parseIntCSV(value string) ([]int, error) { ... "1,65,129" -> []int{1,65,129} ... }
```

**Tests:**

- `parseIntCSV("1,65,129")` works.
- Bad values (`"1,bad"`, `"0"`, empty) return errors or default intentionally.

### Task 5: Add `createMusubiDatasetTOML`

**Objective:** Write Musubi-compatible dataset TOML for LTX and Wan video training.

**Files:**
- Modify: `internal/trainer/config.go` or create `internal/trainer/musubi_config.go`
- Test: `internal/trainer/config_test.go`

**Function shape:**

```go
func createMusubiDatasetTOML(projectName string, s Settings, profile trainingProfile, outDir string) (string, error) {
    width := defaultInt(s.VideoWidth, s.Width)
    height := defaultInt(s.VideoHeight, s.Height)
    if width <= 0 { width = 768 }
    if height <= 0 { height = 512 }

    targetFrames, err := parseIntCSV(nonEmpty(s.VideoTargetFrames, "1,65,129"))
    if err != nil { return "", err }

    repeats := s.VideoNumRepeats
    if repeats <= 0 {
        videos := countDatasetVideos(s.DatasetPath)
        if videos <= 0 { videos = 1 }
        effectiveBatch := maxInt(1, s.TrainBatchSize*s.GradientAccumulationSteps)
        repeats = maxInt(1, (s.TrainingSteps*effectiveBatch+videos-1)/videos)
    }

    content := strings.Builder{}
    content.WriteString("[general]\n")
    content.WriteString(fmt.Sprintf("resolution = [%d, %d]\n", width, height))
    content.WriteString(fmt.Sprintf("batch_size = %d\n", maxInt(1, s.TrainBatchSize)))
    content.WriteString(fmt.Sprintf("caption_extension = %s\n", tomlString(nonEmpty(s.VideoCaptionExtension, ".txt"))))
    content.WriteString(fmt.Sprintf("enable_bucket = %t\n\n", s.VideoEnableBucket))
    content.WriteString("[[datasets]]\n")
    content.WriteString(fmt.Sprintf("video_directory = %s\n", tomlString(filepath.ToSlash(absPath(s.DatasetPath)))))
    content.WriteString(fmt.Sprintf("target_frames = [%s]\n", joinInts(targetFrames)))
    content.WriteString(fmt.Sprintf("frame_extraction = %s\n", tomlString(nonEmpty(s.VideoFrameExtraction, "full"))))
    content.WriteString(fmt.Sprintf("num_repeats = %d\n", repeats))

    path := filepath.Join(outDir, projectName+"_musubi_dataset.toml")
    return path, os.WriteFile(path, []byte(content.String()), 0644)
}
```

**Expected TOML example:**

```toml
[general]
resolution = [768, 512]
batch_size = 1
caption_extension = ".txt"
enable_bucket = true

[[datasets]]
video_directory = "/path/to/videos"
target_frames = [1, 65, 129]
frame_extraction = "full"
num_repeats = 1
```

**Verify:**

```bash
go test ./internal/trainer -run 'TestCreateMusubiDatasetTOML|TestParseIntCSV|TestCountDatasetVideos' -v
```

---

## Phase 5: Add Musubi Command Builders

### Task 6: Build Musubi cache commands

**Objective:** Construct exact argv for cache text and cache latents for LTX23 and WAN22.

**Files:**
- Create/modify: `internal/trainer/musubi.go`
- Test: `internal/trainer/musubi_test.go`

**Command builder shape:**

```go
type musubiCommandKind string

const (
    musubiCommandCacheText    musubiCommandKind = "cache-text"
    musubiCommandCacheLatents musubiCommandKind = "cache-latents"
    musubiCommandTrain        musubiCommandKind = "train"
)

func buildMusubiCommand(root string, kind musubiCommandKind, s Settings, datasetTOML, outputDir string) (dir string, args []string, env []string, err error) {
    profile := profileFor(s)
    dir = musubiRoot(root, s)
    switch profile.Architecture {
    case ArchitectureLTX23:
        return buildLTX23MusubiCommand(dir, kind, s, datasetTOML, outputDir)
    case ArchitectureWAN22:
        return buildWAN22MusubiCommand(dir, kind, s, datasetTOML, outputDir)
    default:
        return "", nil, nil, fmt.Errorf("not a Musubi architecture: %s", profile.Architecture)
    }
}
```

**LTX cache text command:**

```bash
python ltx2_cache_text_encoder_outputs.py \
  --dataset_config <dataset.toml> \
  --ltx2_checkpoint <checkpoint_path> \
  --gemma_safetensors <qwen_path or text_encoder_path> \
  --ltx2_mode video \
  --ltx_version 2.3 \
  --mixed_precision bf16
```

**LTX cache latents command:**

```bash
python ltx2_cache_latents.py \
  --dataset_config <dataset.toml> \
  --ltx2_checkpoint <checkpoint_path> \
  --device cuda \
  --vae_dtype bf16 \
  --ltx2_mode video
```

**Wan cache text command:**

```bash
python wan_cache_text_encoder_outputs.py \
  --dataset_config <dataset.toml> \
  --t5 <qwen_path or text_encoder_path> \
  --batch_size 16
```

**Wan cache latents command:**

```bash
python wan_cache_latents.py \
  --dataset_config <dataset.toml> \
  --vae <vae_path> \
  --i2v
```

**Path choice:**

Prefer root wrapper scripts when present:

- `wan_cache_text_encoder_outputs.py`
- `wan_cache_latents.py`
- `wan_train_network.py`

Fall back to `src/musubi_tuner/...` only if root wrappers are absent.

### Task 7: Build Musubi training commands

**Objective:** Launch `accelerate` for LTX23/WAN22 with the exact flags used by the working Musubi source.

**Files:**
- Modify: `internal/trainer/musubi.go`
- Test: `internal/trainer/musubi_test.go`

**LTX training command shape:**

```bash
python -m accelerate.commands.launch \
  --num_cpu_threads_per_process 8 \
  --mixed_precision bf16 \
  ltx2_train_network.py \
  --mixed_precision bf16 \
  --optimizer_type prodigyopt.Prodigy \
  --learning_rate 1.0 \
  --optimizer_args decouple=True weight_decay=0.01 d_coef=2.0 use_bias_correction=True safeguard_warmup=True \
  --lr_scheduler constant \
  --timestep_sampling shifted_logit_normal \
  --dataset_config <dataset.toml> \
  --output_dir <projectOut> \
  --output_name <projectName> \
  --ltx2_checkpoint <checkpoint_path> \
  --gemma_safetensors <qwen_path/text_encoder_path> \
  --ltx_version 2.3 \
  --ltx_version_check_mode error \
  --ltx2_mode video \
  --fp8_base \
  --fp8_scaled \
  --blocks_to_swap 14 \
  --use_pinned_memory_for_block_swap \
  --gradient_checkpointing \
  --sdpa \
  --network_module networks.lora_ltx2 \
  --network_dim 128 \
  --network_alpha 1 \
  --max_data_loader_n_workers 4 \
  --persistent_data_loader_workers \
  --save_every_n_epochs 1 \
  --save_state \
  --save_state_on_train_end \
  --max_train_epochs <target_epochs> \
  --metadata_title <projectName> \
  --metadata_author <metadata_author> \
  --metadata_tags <metadata_tags>
```

**Wan training command shape:**

```bash
python -m accelerate.commands.launch \
  --num_cpu_threads_per_process 8 \
  --mixed_precision bf16 \
  wan_train_network.py \
  --task i2v-A14B \
  --mixed_precision bf16 \
  --optimizer_type prodigyopt.Prodigy \
  --learning_rate 1.0 \
  --optimizer_args decouple=True weight_decay=0.01 d_coef=2.0 use_bias_correction=True safeguard_warmup=True \
  --lr_scheduler constant \
  --timestep_sampling shift \
  --discrete_flow_shift 5.0 \
  --dataset_config <dataset.toml> \
  --output_dir <projectOut> \
  --output_name <projectName> \
  --dit <dit_path> \
  --t5 <qwen_path/text_encoder_path> \
  --vae <vae_path> \
  --gradient_checkpointing \
  --force_v2_1_time_embedding \
  --sdpa \
  --network_module networks.lora_wan \
  --network_dim 128 \
  --network_alpha 1 \
  --max_data_loader_n_workers 4 \
  --persistent_data_loader_workers \
  --save_every_n_epochs 1 \
  --save_state \
  --save_state_on_train_end \
  --max_train_epochs <target_epochs> \
  --metadata_title <projectName> \
  --metadata_author <metadata_author> \
  --metadata_tags <metadata_tags>
```

**Important:**

- Use argv slices, not shell strings.
- Use `appendFields`-style parsing for extra args only where explicitly intended.
- Set `cmd.Dir = musubiRoot`.
- Set `PYTHONUNBUFFERED=1` in env to keep live logs smooth.
- Include Musubi `src` on `PYTHONPATH` if needed:

```go
func musubiEnv(musubiDir string) []string {
    env := trainingEnv(musubiDir)
    env = append(env, "PYTHONUNBUFFERED=1")
    env = append(env, "PYTHONPATH="+filepath.Join(musubiDir, "src")+string(os.PathListSeparator)+musubiDir)
    return env
}
```

**Tests:**

- LTX cache command contains `ltx2_cache_text_encoder_outputs.py`, `--gemma_safetensors`, `--ltx_version`, `2.3`.
- LTX train command contains `ltx2_train_network.py`, `--network_module`, `networks.lora_ltx2`, `--blocks_to_swap`, `14`.
- Wan train command contains `wan_train_network.py`, `--task`, `i2v-A14B`, `--network_module`, `networks.lora_wan`, `--force_v2_1_time_embedding`.
- No shell quoting is required in generated args.

---

## Phase 6: Wire Musubi Into Manager Lifecycle

### Task 8: Branch `Manager.Start` by profile family

**Objective:** Keep current Anima/SDXL behavior unchanged and launch Musubi commands for video architectures.

**Files:**
- Modify: `internal/trainer/manager.go`
- Create/modify: `internal/trainer/musubi.go`
- Test: `internal/trainer/manager_test.go`

**Implementation:**

Refactor `Start` so existing flow remains in a helper:

```go
func (m *Manager) Start(s Settings) (StartResponse, error) {
    s = normalizeSettings(s)
    profile := profileFor(s)
    if profile.Family == trainingFamilyMusubi {
        return m.startMusubi(s, profile)
    }
    return m.startSDScripts(s, profile)
}
```

`startMusubi` should:

1. Save settings.
2. Check `m.running`.
3. Validate settings and Musubi runtime.
4. Create output dirs:
   - `training/output/<project>/configs`
   - `training/output/<project>/sample` only if later preview support exists
5. Generate Musubi dataset TOML:
   - `<project>_musubi_dataset.toml`
6. Build Musubi train command.
7. Start the process using the same log piping/wait logic.
8. Set active GPU label to `profile.Label + " training"`.
9. Log generated command and config path.

**Do not:**

- Run cache steps implicitly on the first version unless explicitly requested. Caching can be long; make it a user-visible action first.
- Reuse sd-scripts training TOML for Musubi unless Musubi docs confirm every flag maps cleanly.

### Task 9: Add cache actions to `StartDatasetPrep`

**Objective:** Let the UI run Musubi pre-flight actions with live logs.

**Files:**
- Modify: `internal/trainer/manager.go`
- Test: `internal/trainer/manager_test.go`

**Allowed new actions:**

```go
const (
    datasetPrepTag                 = "tag"
    datasetPrepResize              = "resize"
    datasetPrepAll                 = "all"
    datasetPrepNormalizeVideo      = "normalize-video"
    datasetPrepMusubiDatasetTOML   = "musubi-dataset-toml"
    datasetPrepMusubiCacheText     = "musubi-cache-text"
    datasetPrepMusubiCacheLatents  = "musubi-cache-latents"
)
```

Behavior:

- `musubi-dataset-toml`: generate TOML and return without starting a long process.
- `musubi-cache-text`: generate TOML then run cache-text command.
- `musubi-cache-latents`: generate TOML then run cache-latents command.
- `normalize-video`: run FFmpeg normalizer from Phase 7.

**Status labels:**

- cache text: `Musubi text encoder cache`
- cache latents: `Musubi latent cache`
- normalize: `video normalization`

---

## Phase 7: Port Video Normalizer

### Task 10: Add Go FFmpeg normalizer backend

**Objective:** Bring the useful MusubiTunerApp `video-normalizer` behavior into TrainFlow dataset prep.

**Files:**
- Create: `internal/trainer/normalizer.go`
- Test: `internal/trainer/normalizer_test.go`

**Minimum viable behavior:**

- Validate `ffmpeg` exists:

```go
func validateFFmpeg() error {
    if _, err := exec.LookPath("ffmpeg"); err != nil {
        return fmt.Errorf("ffmpeg not found in PATH; install ffmpeg before video normalization")
    }
    return nil
}
```

- Input: `s.DatasetPath`
- Output: `training/prepared/<project>-video`
- Copy captions with matching basename `.txt`.
- Supported source extensions from `validVideoExtensions`.
- Presets:
  - `ltx23`:
    - landscape default: `768x512`
    - portrait optional later: `512x768`
    - fps `24`
    - duration from `VideoDuration`
  - `wan22`:
    - default: `720x1280`
    - fps `24`
- Build FFmpeg command with scaling and padding, for example:

```bash
ffmpeg -y -i input.mp4 \
  -vf "fps=24,scale=w=768:h=512:force_original_aspect_ratio=decrease,pad=768:512:(ow-iw)/2:(oh-ih)/2" \
  -t 5 \
  -an \
  -c:v libx264 -crf 19 -preset medium \
  output.mp4
```

**Avoid overbuilding:**

- Do not port every Fyne UI feature at first.
- Do not add NVENC-only defaults because not every environment has `hevc_nvenc`.
- Provide `ExtraTrainArgs`/future fields if the user needs NVENC tuning later.

### Task 11: Wire normalize action into manager

**Objective:** Add `normalize-video` action to `/api/dataset/prep`.

**Files:**
- Modify: `internal/trainer/manager.go`
- Modify: `internal/trainer/routes.go` only if request/response shape changes; otherwise no route change needed.

Behavior:

- When normalize starts, set running status and stream FFmpeg logs.
- On success, return or log prepared path.
- Consider switching `DatasetPath` to prepared output only after successful normalization. If switching at start, preserve old path in log.

---

## Phase 8: Frontend Selector and Video Controls

### Task 12: Add LTX23/WAN22 selector buttons

**Objective:** Let users choose LTX23 and WAN22 from the existing architecture toggle.

**Files:**
- Modify: `cmd/trainflow/web/index.html`
- Modify: `cmd/trainflow/web/app.js`
- Modify: `cmd/trainflow/web/styles.css` only if layout needs adjustment

**HTML change:**

```html
<button type="button" class="architecture-button" data-architecture="ltx23" title="Use Musubi LTX 2.3 video LoRA training.">LTX 2.3</button>
<button type="button" class="architecture-button" data-architecture="wan22" title="Use Musubi Wan 2.2 video LoRA training.">Wan 2.2</button>
```

**JS change:**

```js
function normalizeArchitecture(value) {
  return ["anima", "sdxl", "ltx23", "wan22"].includes(value) ? value : "anima";
}

function isVideoArchitecture(architecture) {
  return architecture === "ltx23" || architecture === "wan22";
}
```

Update `setArchitecture`:

- Video profiles show video fields.
- LTX23 labels:
  - checkpoint path label: `LTX 2.3 Checkpoint`
  - qwen/text encoder label: `Gemma Text Encoder`
  - VAE label hidden or optional unless needed.
- WAN22 labels:
  - dit label: `Wan DiT Checkpoint`
  - qwen/text encoder label: `T5 Text Encoder`
  - VAE label: `Wan VAE`
- Defaults on selector click:
  - optimizer to `Prodigy`
  - learning rate to `1.0`
  - rank to `128` if not manually changed
  - network alpha to `1`

### Task 13: Add UI fields for video/Musubi settings

**Objective:** Expose required Musubi settings without cluttering Anima/SDXL.

**Files:**
- Modify: `cmd/trainflow/web/index.html`
- Modify: `cmd/trainflow/web/app.js`

Add these IDs to `fields` and `numericFields` as appropriate:

```js
"musubi_path",
"target_epochs",
"mixed_precision",
"num_cpu_threads",
"video_width",
"video_height",
"video_fps",
"video_duration",
"video_target_frames",
"video_frame_extraction",
"video_num_repeats",
"video_caption_extension",
"video_enable_bucket",
"ltx_version",
"ltx_mode",
"ltx_version_check_mode",
"wan_task",
"blocks_to_swap",
"network_alpha",
"network_module",
"timestep_sampling",
"discrete_flow_shift",
"fp8_base",
"fp8_scaled",
"sdpa",
"gradient_checkpointing",
"use_pinned_memory_for_block_swap",
"persistent_data_loader_workers",
"save_state_on_train_end",
"metadata_author",
"metadata_tags",
"extra_train_args",
"extra_cache_text_args",
"extra_cache_latents_args"
```

Suggested UI sections:

1. `Model Paths`
   - Musubi path browser (directory)
   - LTX/Wan checkpoint fields via existing path inputs
2. `Video Dataset`
   - width, height, fps, duration
   - target frames
   - frame extraction
   - repeats
   - caption extension
   - enable bucket checkbox
3. `Musubi Training`
   - epochs
   - mixed precision
   - CPU threads
   - blocks to swap
   - network alpha
   - network module
   - timestep sampling
   - FP8/base/scaled, SDPA, gradient checkpointing checkboxes

**Keep initial UI simple:**

If the page becomes too crowded, put advanced Musubi fields in a collapsed `<details>` block.

### Task 14: Add Musubi dataset action buttons

**Objective:** Add user-triggered actions for the Musubi pipeline.

**Files:**
- Modify: `cmd/trainflow/web/index.html`
- Modify: `cmd/trainflow/web/app.js`

Add buttons in Dataset Prep panel or a new `Video Prep` panel:

```html
<button id="normalizeVideoButton" class="secondary">Normalize Video</button>
<button id="writeMusubiDatasetButton" class="secondary">Write Musubi TOML</button>
<button id="cacheMusubiTextButton" class="secondary">Cache Text</button>
<button id="cacheMusubiLatentsButton" class="secondary">Cache Latents</button>
```

Add constants:

```js
const normalizeVideoButton = document.getElementById("normalizeVideoButton");
const writeMusubiDatasetButton = document.getElementById("writeMusubiDatasetButton");
const cacheMusubiTextButton = document.getElementById("cacheMusubiTextButton");
const cacheMusubiLatentsButton = document.getElementById("cacheMusubiLatentsButton");
```

Add listeners:

```js
normalizeVideoButton.addEventListener("click", () => runDatasetPrep("normalize-video"));
writeMusubiDatasetButton.addEventListener("click", () => runDatasetPrep("musubi-dataset-toml"));
cacheMusubiTextButton.addEventListener("click", () => runDatasetPrep("musubi-cache-text"));
cacheMusubiLatentsButton.addEventListener("click", () => runDatasetPrep("musubi-cache-latents"));
```

Disable these buttons when not a video architecture.

---

## Phase 9: Runtime Tool / Dependency Strategy

### Task 15: Decide how Musubi dependencies are installed

**Objective:** Avoid breaking the existing `sd-scripts` runtime while ensuring Musubi scripts can run.

**Files likely to inspect/modify:**
- `cmd/runtime-tool/main.go`
- `internal/runtimeops/runtimeops.go`
- `README.md`

**Recommended first implementation:**

- Do not merge Musubi dependencies into existing `training/sd-scripts` blindly.
- Add a Musubi runtime status check that verifies:
  - `musubiRoot` exists
  - `pyproject.toml` exists
  - scripts exist
  - active Python can import key modules only if runtime is installed
- Document that the local Musubi fork is expected at:
  - `/home/darksidewalker/GitHub/musubi-tuner`
- Later add runtime-tool support for:

```bash
cd /home/darksidewalker/GitHub/musubi-tuner
uv sync
```

or a project-specific venv, depending on the fork's actual dependency manager.

**Open question:** Does TrainFlow's existing Python runtime already contain Musubi dependencies? If not, decide whether to run Musubi with:

- existing TrainFlow Python (`pythonExecutable(root)`), or
- Musubi venv Python (`<musubiRoot>/.venv/bin/python`) when present.

**Preferred resolver:**

```go
func musubiPython(root string, s Settings) string {
    venvPython := filepath.Join(musubiRoot(root, s), ".venv", "bin", "python")
    if fileExists(venvPython) { return venvPython }
    return pythonExecutable(root)
}
```

---

## Phase 10: Validation and Tests

### Task 16: Unit tests for config and commands

**Objective:** Catch command regressions without launching real training.

**Files:**
- Modify: `internal/trainer/config_test.go`
- Add: `internal/trainer/musubi_test.go`

Test cases:

1. `TestCreateMusubiDatasetTOML_LTX23`
   - creates temp video dataset with `.mp4` and `.txt`
   - asserts TOML has `video_directory`, `target_frames`, `[general]`, `resolution`
2. `TestBuildMusubiLTX23CacheTextCommand`
3. `TestBuildMusubiLTX23TrainCommand`
4. `TestBuildMusubiWAN22CacheCommands`
5. `TestBuildMusubiWAN22TrainCommand`
6. `TestNormalizeSettingsVideoProfiles`
7. `TestValidateSettingsVideoProfilesRequireCorrectPaths`

### Task 17: Frontend smoke test manually

**Objective:** Verify UI can save/load new fields and selector state.

Manual steps:

```bash
go test ./...
go build ./cmd/trainflow
go build ./cmd/runtime-tool
./trainflow
```

Then in browser:

1. Select `LTX 2.3`.
2. Confirm video-only controls appear.
3. Save settings.
4. Refresh page.
5. Confirm selected architecture and fields persist.
6. Select `Wan 2.2`.
7. Confirm labels change to Wan/T5/VAE wording.
8. Switch back to `Anima` and `SDXL`; confirm old fields still work.

### Task 18: Dry-run command verification

**Objective:** Verify constructed commands without a full expensive training run.

Add a debug log before process start:

```go
m.appendLog("Command: " + shellPreview(python, args))
```

Manual checks:

1. Use a tiny temp video dataset.
2. Click `Write Musubi TOML`; inspect TOML under `training/output/<project>/configs`.
3. Click `Cache Text`; verify command starts and reaches Musubi argument parsing.
4. Click `Cache Latents`; verify command starts and reaches Musubi argument parsing.
5. Click `Start`; stop after command construction/initial logs if full training is not desired.

---

## Files Likely to Change

Backend:

- `internal/trainer/types.go`
- `internal/trainer/profiles.go`
- `internal/trainer/config.go`
- `internal/trainer/defaults.go`
- `internal/trainer/manager.go`
- `internal/trainer/routes.go` only if response/API shape changes
- `internal/trainer/util.go` if helper functions belong there
- `internal/trainer/musubi.go` (new)
- `internal/trainer/musubi_test.go` (new)
- `internal/trainer/normalizer.go` (new)
- `internal/trainer/normalizer_test.go` (new)
- `internal/trainer/config_test.go`
- `internal/trainer/manager_test.go`

Frontend:

- `cmd/trainflow/web/index.html`
- `cmd/trainflow/web/app.js`
- `cmd/trainflow/web/styles.css` if needed

Docs/runtime:

- `README.md`
- `AGENTS.md` if new training profile notes should be added
- `cmd/runtime-tool/main.go` only if Musubi install/checks are added
- `internal/runtimeops/runtimeops.go` only if runtime status needs Musubi awareness

Do not commit/generated-copy unless explicitly intended:

- `training/output/**`
- `training/prepared/**`
- `training/settings.json` local state
- `training/musubi-tuner/**` if it is a local clone/cache and not intended as source

---

## Risks and Mitigations

### Risk: Python dependency conflicts between `sd-scripts` and `musubi-tuner`

Mitigation:

- Prefer `musubiRoot/.venv/bin/python` if present.
- Fall back to TrainFlow Python only when Musubi venv does not exist.
- Log which Python and Musubi root are being used.

### Risk: Wrong text encoder field semantics

Mitigation:

- In UI labels, make `qwen_path` dynamic:
  - Anima: `Qwen3`
  - LTX23: `Gemma Text Encoder`
  - WAN22: `T5 Text Encoder`
- Optionally add a dedicated `text_encoder_path` and migrate video profiles to it.

### Risk: Wan script path mismatch

Mitigation:

- Resolve script path by checking root wrappers first, then `src/musubi_tuner/<script>`.
- Unit test against the actual local fork file layout.

### Risk: Caching requirements are easy to skip

Mitigation:

- Add visible `Cache Text` and `Cache Latents` buttons.
- When starting training for Musubi, log a warning if expected cache artifacts are not found, but do not block unless Musubi requires it.

### Risk: FFmpeg/NVENC availability

Mitigation:

- Start with CPU `libx264` defaults.
- Add codec choice later.
- Validate `ffmpeg` before running normalization.

### Risk: Video previews are not images

Mitigation:

- Do not promise preview images for Musubi in the first pass.
- Keep `sample` gallery working for existing Anima/SDXL.
- Add video sample support later if Musubi produces preview videos.

---

## Suggested Implementation Order

1. Add architecture constants/profile support and tests.
2. Extend `Settings` and defaults with video/Musubi fields and tests.
3. Add Musubi root/Python/script resolution and tests.
4. Add Musubi dataset TOML generation and tests.
5. Add Musubi command builders and tests.
6. Refactor `Manager.Start` into sd-scripts vs Musubi paths.
7. Add Musubi dataset prep/cache actions.
8. Add UI selector buttons and normalize frontend architecture handling.
9. Add video/Musubi UI fields.
10. Add normalizer backend and UI button.
11. Update README/AGENTS notes.
12. Run full verification:

```bash
go test ./...
go build ./cmd/trainflow
go build ./cmd/runtime-tool
```

---

## Acceptance Criteria

- `LTX 2.3` and `Wan 2.2` are selectable in the TrainFlow UI.
- Selection persists through `/api/settings` and page reload.
- Existing `Anima` and `SDXL` behavior remains unchanged.
- LTX23 settings generate a Musubi dataset TOML with `video_directory` and `target_frames`.
- WAN22 settings generate a Musubi dataset TOML with the same video TOML shape.
- LTX23 command builder emits `ltx2_cache_*` and `ltx2_train_network.py` commands matching the local Musubi fork expectations.
- WAN22 command builder emits `wan_cache_*` and `wan_train_network.py` commands with `--task i2v-A14B`, `--dit`, `--t5`, and `--vae`.
- Cache text and cache latents can be started from the UI and stream logs.
- Training can be started from the UI and streams logs.
- `go test ./...` passes.
- `go build ./cmd/trainflow` passes.
- `go build ./cmd/runtime-tool` passes.

---

## Open Questions Before Implementation

1. Should TrainFlow use the local Musubi checkout in place (`/home/darksidewalker/GitHub/musubi-tuner`) or copy/clone it into `DaSiWa-TrainFlow/training/musubi-tuner`?
2. Does `/home/darksidewalker/GitHub/musubi-tuner/.venv/bin/python` exist and contain all dependencies, or should TrainFlow's existing Python runtime install Musubi dependencies?
3. Should `qwen_path` be reused as the generic text encoder field for video profiles, or should a new `text_encoder_path` be added to avoid confusing names?
4. Should video normalization switch `dataset_path` to `training/prepared/<project>-video` automatically after success, or should the user choose it manually?
5. Should first implementation support both LTX landscape/portrait presets, or keep only one resolution default and let users edit width/height?
