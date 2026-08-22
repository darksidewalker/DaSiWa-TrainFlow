# DaSiWa TrainFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/darksidewalker/DaSiWa-TrainFlow/blob/main/LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.22-00ADD8?style=flat&logo=go&logoColor=white)](https://go.dev/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20Windows-E34F26?style=flat&logo=gnu&logoColor=white)](https://github.com/darksidewalker/DaSiWa-TrainFlow)
[![GitHub Stars](https://img.shields.io/github/stars/darksidewalker/DaSiWa-TrainFlow?style=flat)](https://github.com/darksidewalker/DaSiWa-TrainFlow/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/darksidewalker/DaSiWa-TrainFlow?style=flat)](https://github.com/darksidewalker/DaSiWa-TrainFlow/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/darksidewalker/DaSiWa-TrainFlow?style=flat)](https://github.com/darksidewalker/DaSiWa-TrainFlow/issues)

> Portable Go wrapper around the `sd-scripts` and Musubi Python training stacks — train LoRAs and Textual Inversions for Anima, SDXL/Pony/Illustrious, LTX 2.3/Wan 2.2 video, and Krea 2 image models from a single embedded UI on Linux and Windows.

![TrainFlow preview](assets/DaSiWa-TrainFlow.webp)

---

## Quick Start

**Linux (with Git)**
```bash
git clone --depth 1 https://github.com/darksidewalker/DaSiWa-TrainFlow.git && cd DaSiWa-TrainFlow
chmod +x TrainFlow TrainFlow_Runtime_Tool && ./TrainFlow_Runtime_Tool
```

**Linux (without Git)**
```bash
curl -L -o TrainFlow.zip https://github.com/darksidewalker/DaSiWa-TrainFlow/archive/refs/heads/main.zip
unzip TrainFlow.zip && cd DaSiWa-TrainFlow-main
chmod +x TrainFlow TrainFlow_Runtime_Tool && ./TrainFlow_Runtime_Tool
```

**Windows PowerShell (with Git)**
```powershell
git clone --depth 1 https://github.com/darksidewalker/DaSiWa-TrainFlow.git; cd DaSiWa-TrainFlow
.\TrainFlow_Runtime_Tool.exe
```

**Windows PowerShell (without Git)**
```powershell
Invoke-WebRequest -Uri https://github.com/darksidewalker/DaSiWa-TrainFlow/archive/refs/heads/main.zip -OutFile TrainFlow.zip
Expand-Archive TrainFlow.zip -Force; cd DaSiWa-TrainFlow-main
.\TrainFlow_Runtime_Tool.exe
```

The runtime tool opens at `http://127.0.0.1:7870`. Click **Verify Runtime**, then **Install Requirements**. Once ready, launch the trainer (`./TrainFlow` or `.\TrainFlow.exe`) — the UI opens at `http://127.0.0.1:7860`.

---

## How To Use

1. **Set up the runtime** — run the Runtime Tool, verify it, install requirements
2. **Pick a profile** — choose your model family in the main UI
3. **Point to your data** — select model files and dataset folders using the Browse buttons
4. **Configure training** — set trigger word, rank, optimizer, steps — or click **Auto Calc** for smart defaults based on your dataset size and GPU VRAM
5. **Start training** — watch the live log, monitor hardware, browse preview samples
6. **Finish** — click Quit when done; your outputs are in the project folder

---

## Support Matrix

|                        | Linux | Windows |
|------------------------|:-----:|:-------:|
| CUDA (NVIDIA GPU)      |  &#x2705;  |  &#x2705;  |
| ROCm (AMD GPU, Linux)  |  &#x2705;  |  &#x1F7E5;  |
| AMD GPU monitoring     |  &#x2705;  |  &#x2705;  |
| Portable Python runtime|  &#x2705;  |  &#x2705;  |
| uv / pip installs      |  &#x2705;  |  &#x2705;  |
| Hardware overlay       |  &#x2705;  |  &#x2705;  |

| Hardware Monitor        | NVIDIA (CUDA) | AMD (Linux sysfs) | AMD (Windows WMI) |
|-------------------------|:-------------:|:-----------------:|:-----------------:|
| GPU utilization         |    &#x2705;    |      &#x2705;      |    &#x1F7E5;      |
| VRAM used / total       |    &#x2705;    |      &#x2705;      |    &#x2705;      |
| Temperature             |    &#x2705;    |      &#x2705;      |    &#x1F7E5;      |
| Power draw / limit      |    &#x2705;    |      &#x2705;      |    &#x1F7E5;      |
| CPU usage               |    &#x2705;    |      &#x2705;      |    &#x2705;      |
| CPU temperature         |    &#x2705;    |      &#x2705;      |    &#x2705;      |
| RAM usage               |    &#x2705;    |      &#x2705;      |    &#x2705;      |

---

## Feature Matrix

| Feature                          | Anima | SDXL / Pony / Illustrious | LTX 2.3 Video | Wan 2.2 Video | Krea 2 Image |
|----------------------------------|:-----:|:-------------------------:|:-------------:|:-------------:|:------------:|
| LoRA training                    |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Textual Inversion                |  &#x2705;  |           &#x2705;           |    &#x1F7E5;    |    &#x1F7E5;    |   &#x1F7E5;   |
| Auto Calc (profile-aware)        |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Training previews (enabled by default) | &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Dynamic multi-prompt sampling    |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Resume from state                |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Prodigy optimizer                |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| AdamW / AdamW8bit                |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Flash Attention                  |  &#x2705;  |           &#x1F7E5;           |    &#x1F7E5;    |    &#x1F7E5;    |   &#x1F7E5;   |
| torch.compile                    |  &#x2705;  |           &#x1F7E5;           |    &#x1F7E5;    |    &#x1F7E5;    |   &#x1F7E5;   |
| FP8 base / scaled                |  &#x1F7E5;  |           &#x1F7E5;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Native FP8 checkpoint detection  |  &#x1F7E5;  |           &#x1F7E5;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Block swap + pinned memory       |  &#x1F7E5;  |           &#x1F7E5;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| H2D-only LoRA block swap         |  &#x1F7E5;  |           &#x1F7E5;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| VRAM-based batch sizing          |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Metadata (author + tags)         |  &#x2705;  |           &#x2705;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Text/latent caching              |  &#x1F7E5;  |           &#x1F7E5;           |    &#x2705;    |    &#x2705;    |   &#x2705;   |
| Video normalization              |  &#x1F7E5;  |           &#x1F7E5;           |    &#x2705;    |    &#x2705;    |   &#x1F7E5;   |
| Managed model download           |  &#x2705;  |           &#x1F7E5;           |    &#x1F7E5;    |    &#x1F7E5;    |   &#x2705;   |

&#x2705; Supported — &#x1F7E5; Not applicable or not supported

---

## Training Profiles

| Profile | Model Files | Network Module | Bucket Step | Pipeline |
|---------|-------------|----------------|-------------|----------|
| **Anima** | DiT + Qwen3 + VAE | `networks.lora_anima` | 64px | sd-scripts |
| **SDXL / Pony / Illustrious** | checkpoint (+ optional VAE) | `networks.lora` | 32px | sd-scripts |
| **LTX 2.3** | checkpoint + Gemma encoder | `networks.lora_ltx2` | 16px | Musubi video |
| **Wan 2.2** | DiT + T5 + VAE | `networks.lora_wan` | 16px | Musubi video |
| **Krea 2** | RAW DiT + Qwen3-VL + Qwen-Image VAE | `networks.lora_krea2` | 32px | Musubi image |

**Auto Calc** reads your profile and dataset count, then picks rank, learning rates, batch size, gradient accumulation, steps, and save/sample intervals — all tuned to your available VRAM. It preserves your chosen optimizer (Prodigy stays at `lr=1.0` constant; AdamW/AdamW8bit stay at `1e-4` cosine).

---

## Features At A Glance

### Embedded Training GUI
Single portable binary, no separate web build step. Everything runs from one download:
- Colored profile switcher for all five model families
- Local file browser for datasets, models, and resume paths
- Settings autosaved to `training/settings.json`
- Optimizer-aware defaults (Prodigy, AdamW8bit, AdamW)
- Full training controls: rank, alpha, learning rates, batch, grad accum, steps, intervals
- SDXL-specific UNet/text-encoder LR fields and UNet-only toggle
- Optional Flash Attention and torch.compile (Anima)
- Multi-prompt sample generation with color-coded prompts
- Training preview toggle (on by default, configurable per session)
- Resume panel with automatic latest-state discovery
- Live training log streamed in real time
- Preview gallery with image overlay
- Output button to open the project output folder

### Dataset Preparation
- WD EVA02 ONNX tagging/captioning
- Combined Tag + Resize workflow with configurable thresholds
- Resize-copy helper (`training/prepared/<project>`)
- Video normalization pipeline: resolution, FPS, duration, codec, quality, parallel workers, speed control, skip frames
- Automatic Musubi dataset TOML generation with text/latent cache rebuild triggers

### Runtime & Model Management
- Companion Runtime Tool at `http://127.0.0.1:7870`
- Verify, update, and install Python runtime dependencies (uv-first with pip fallback)
- Download Anima base models and Krea 2 runtime models directly from the UI
- Download prep assets (WD tagger, U2Net)
- PyTorch backend selector: CUDA 12.4 default, experimental ROCm 6.4, or existing user-managed install
- GPU auto-detection with vendor badge in the header
- Vendor-colored backend panel (green NVIDIA/CUDA, orange AMD/ROCm) with mismatch warnings
- Platform-specific portable Python (no system Python required)

### Hardware Monitoring Overlay
Compact real-time display inside the sampler panel:
- CPU usage, RAM usage, CPU temperature
- GPU utilization, VRAM, temperature, power draw/limit, active task labels
- NVIDIA via nvidia-smi; AMD via kernel sysfs (Linux) or WMI (Windows)

---

## Advanced

<details>
<summary><strong>Resume Training</strong></summary>

Three options in the Resume panel:
- **Resume training** enables state resume
- **Use latest saved state** auto-finds the newest state directory
- **Resume State Path** lets you pick a specific folder manually

All runs write state by default (`save_last_n_steps_state = 1`, `save_last_n_epochs_state = 1`).

</details>

<details>
<summary><strong>Textual Inversion</strong></summary>

Switch Training Mode to **Textual Inversion** to train text embeddings instead of LoRAs. Configure:
- Placeholder token (default `*test*`)
- Number of vectors (default 16)
- Initializer word
- Learning rate (auto-scaled by vector count)
- Batch size (VRAM-aware)
- Random cropping toggle

Supported for Anima and SDXL/Pony/Illustrious profiles.

</details>

<details>
<summary><strong>Anima LoRA Metadata</strong></summary>

New LoRA files include Anima safetensors metadata. Inspect or repair existing LoRAs:

```bash
python training/sd-scripts/tools/anima_lora_metadata.py path/to/lora.safetensors
python training/sd-scripts/tools/anima_lora_metadata.py path/to/lora.safetensors --fix
```

</details>

<details>
<summary><strong>Required Models</strong></summary>

Use the Runtime Tool (**Download Models**) to fetch base files:

```
models/anima/dit/anima-base-v1.0.safetensors
models/anima/text_encoder/qwen_3_06b_base.safetensors
models/anima/vae/qwen_image_vae.safetensors
```

Use **Download Prep** for dataset-prep models:

```
models/wd-eva02-large-tagger-v3/    (WD EVA02 tagger)
models/u2net/u2net.onnx             (U2Net background removal)
```

Or install manually:
```bash
git clone https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3 models/wd-eva02-large-tagger-v3
curl -L -o models/u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

</details>

<details>
<summary><strong>torch.compile (Anima)</strong></summary>

Per-block DiT compilation for faster Anima training. Configurable mode, backend (inductor), dynamic shape handling, and cache size. Requires Triton — automatically disabled when ROCm or custom PyTorch backends are selected. On Windows with dynamic shapes, requires MSVC Build Tools environment.

</details>

<details>
<summary><strong>ROCm and Custom PyTorch Note</strong></summary>

CUDA remains the fully supported default. ROCm 6.4 and existing-PyTorch modes are experimental — they disable CUDA-only features (Flash Attention, torch.compile/Triton) automatically. AMD GPU monitoring on Linux works via kernel sysfs and does not require ROCm installed.

</details>

<details>
<summary><strong>H2D-only Musubi Block Swap</strong></summary>

LTX 2.3, Wan 2.2, and Krea 2 LoRA profiles use Musubi's H2D-only block swap by default when block swap is enabled. The frozen base weights keep a master copy in CPU RAM, so only host-to-device transfers are needed; the redundant device-to-host copy used by classic block swap is skipped. This is especially useful with FP8 base/scaled weights.

The Advanced Musubi dialog exposes **H2D-only block swap (LoRA)** and **H2D ring buffers**. The default ring size is `2` for transfer/compute overlap; select `1` to minimize VRAM at the cost of that overlap.

Requirements and limits:
- CUDA and frozen-base LoRA / LoHa / LoKr training only; it is not compatible with full base-model fine-tuning.
- Gradient checkpointing is required and is enabled by the Musubi profile defaults.
- Do not combine it with LTX 2 aggressive block-swap modes.

</details>

<details>
<summary><strong>Adding New Training Types</strong></summary>

Future types should use small profile adapters rather than UI/backend branches. See `docs/training-type-integration.md` for the full checklist.

Short version: add an architecture constant and profile case, Musubi command builder, TOML generation, colored UI button, and focused tests.

Note: upstream Musubi source is `https://github.com/kohya-ss/musubi-tuner`. The LTX 2.3 integration depends on LTX2 entrypoints that may live in an LTX-capable fork — preserve those files when updating vendored Musubi.

</details>

<details>
<summary><strong>System Requirements</strong></summary>

| Requirement | Notes |
|-------------|-------|
| **GPU** | NVIDIA (CUDA) recommended; AMD (ROCm 6.4) on Linux |
| **Python 3.12** | Recommended for the embedded runtime |
| **nvidia-smi** | Needed for NVIDIA GPU monitoring |
| **amdgpu kernel driver** | Needed for AMD GPU monitoring on Linux |
| **Go 1.22+** | Only when building from source |

</details>

---

## Container (Docker / Podman)

You can run the whole app in a container. It behaves the same on Docker and
on Podman — `scripts/run.sh` detects which engine you have and sets the right
GPU flags automatically.

**First start** installs the Python training stack (Python 3.12 + PyTorch +
trainer dependencies) into the `trainflow_tf` volume. That takes about
5–10 minutes and uses ~8 GB. **Every start after that is fast.**

### 1. Build and start

The first time, build the image **and** start the app in one command:

```bash
TRAINFLOW_BUILD=1 ./scripts/run.sh
```

- **First start** has no runtime volume yet, so it runs in the foreground
  and shows the one-time bootstrap as it happens. Let it finish.
- **Every later start** just runs the app in the background — the image is
  already built and the volume already exists:

```bash
./scripts/run.sh
```

(If you ever need to rebuild the image on its own without starting the app,
run `podman build --format docker -t dasiwa/trainflow:latest .` — or
`docker build -t dasiwa/trainflow:latest .` on a Docker host.)

When it is up, open the UI at `http://127.0.0.1:7860`.

### 2. Where your files go

The container looks for your data next to the project folder:

```
datasets/     # your image / video datasets
models/       # base model files (DiT, VAE, text encoders, ...)
```

Both folders are created automatically if they do not exist yet.
In the UI, choose paths under `/app/datasets/...` and `/app/models/...`.

**Datasets live on your machine, not in the container.** Both folders are
bind-mounted in, so everything you put in `datasets/` or `models/` stays on
the host, is visible from both sides, and is safe across container restarts,
image rebuilds, and even `volume rm trainflow_tf` (the volume only holds the
Python runtime and app state — never your data).

To keep your data in a different place, point `DATASETS_DIR` (and
`MODELS_DIR`) at it — works for both `scripts/run.sh` and compose:

```bash
DATASETS_DIR=/path/to/datasets ./scripts/run.sh
DATASETS_DIR=/path/to/datasets podman compose up -d
```

Model files (~20 GB) can be downloaded through the Runtime Tool (step 3) or
copied over from a normal, non-container install.

One exception: **training results** (the project output folder) are written
inside the container to `/app/training/output/<project>` — that lives in the
`trainflow_tf` volume, not on your machine. To grab a finished LoRA, copy it
out of the volume (e.g. `podman cp trainflow:/app/training/output/<project> .`).

### 3. Runtime Tool (optional)

The Runtime Tool is where you install or update the Python runtime and where
you download model files. It shares the same volume as the main app, so you
start it with compose:

```bash
# use whichever engine you have — podman or docker:
podman compose up -d trainflow-tool
```

It opens at `http://127.0.0.1:7871`. When you are done:

```bash
podman compose stop trainflow-tool
```

### Good to know

- **GPU.** `scripts/run.sh` passes the GPU to the container for you —
  Docker uses `--gpus all`, Podman uses the host's NVIDIA CDI
  (`--device nvidia.com/gpu=all`). Training only uses GPU 0.
  For AMD ROCm instead: add `--device /dev/kfd --device /dev/dri` and set
  `TORCH_BACKEND=rocm`.
- **GPU tiles in the UI.** The launcher bind-mounts the host's `nvidia-smi`
  so the live GPU tiles work. If it is missing, training still runs — the
  GPU tiles just stay empty.
- **Update the app.** Rebuild the image: `TRAINFLOW_BUILD=1 ./scripts/run.sh`.
- **Start over from scratch.** Delete the runtime volume to force a
  re-bootstrap: `podman volume rm trainflow_tf` (or `docker volume rm trainflow_tf`).

---

## Development

<details>
<summary><strong>Build From Source</strong></summary>

Only needed when modifying Go code or rebuilding release artifacts.

**Linux:**
```bash
go build -trimpath -ldflags="-s -w" -o TrainFlow ./cmd/trainflow
go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool ./cmd/runtime-tool
```

**Cross-compile (Windows):**
```powershell
.\build.ps1
```

Outputs:
```
TrainFlow                              # Linux binary
TrainFlow.exe                          # Windows binary
TrainFlow_Runtime_Tool                 # Linux runtime tool
TrainFlow_Runtime_Tool.exe             # Windows runtime tool
dist/trainflow-linux-amd64             # Linux release
dist/trainflow-windows-amd64.exe       # Windows release
dist/trainflow-runtime-tool-linux-amd64
dist/trainflow-runtime-tool-windows-amd64.exe
```

</details>

<details>
<summary><strong>Shipping & Distribution</strong></summary>

Do not commit `python_embeded/` to Git.

**Normal distribution** — ship only the root binaries; the runtime tool creates the platform runtime on the user's machine:
- Windows: `TrainFlow.exe` + `TrainFlow_Runtime_Tool.exe`
- Linux: `TrainFlow` + `TrainFlow_Runtime_Tool`

**Fully offline packages** — create a ZIP/7z containing binaries plus `python_embeded/<platform>` and upload as a GitHub Release asset.

</details>

---

## Credits

Built on the [`sd-scripts`](https://github.com/kohya-ss/sd-scripts) training stack. Video training powered by [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). TrainFlow's Musubi integration, including H2D-only block swap support, credits [AkaneTendo25/musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner). The original [Anima TrainFlow](https://github.com/ThetaCursed/Anima-TrainFlow) by ThetaCursed inspired the portable trainer concept.
