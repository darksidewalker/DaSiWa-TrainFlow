# DaSiWa TrainFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/darksidewalker/DaSiWa-TrainFlow/blob/main/LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.22-00ADD8?style=flat&logo=go&logoColor=white)](https://go.dev/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20Windows-E34F26?style=flat&logo=gnu&logoColor=white)](https://github.com/darksidewalker/DaSiWa-TrainFlow)
[![GitHub Stars](https://img.shields.io/github/stars/darksidewalker/DaSiWa-TrainFlow?style=flat)](https://github.com/darksidewalker/DaSiWa-TrainFlow/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/darksidewalker/DaSiWa-TrainFlow?style=flat)](https://github.com/darksidewalker/DaSiWa-TrainFlow/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/darksidewalker/DaSiWa-TrainFlow?style=flat)](https://github.com/darksidewalker/DaSiWa-TrainFlow/issues)

> Portable Go wrapper around the `sd-scripts` Python training stack — train LoRAs for Anima, SDXL/Pony/Illustrious, and LTX 2.3/Wan 2.2 video from a single embedded UI on Linux and Windows.

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

The runtime tool opens a local installer UI at `http://127.0.0.1:7870`. Click **Verify Runtime** first if you downloaded a bundled build, then **Update Runtime** or **Install Requirements** as needed.

Once the runtime is ready, launch the trainer:

| Platform | Command |
|----------|---------|
| Linux    | `./TrainFlow` |
| Windows  | `.\TrainFlow.exe` |

The UI opens at `http://127.0.0.1:7860` (or open it manually if your browser doesn't launch).

---

## Training Workflow

1. Run the **Runtime Tool** and install/update dependencies.
2. Launch **TrainFlow** and choose a training profile: **Anima**, **SDXL / Pony / Illustrious**, or **LTX 2.3 / Wan 2.2**.
3. Use the **Browse** buttons to select model files and dataset folders.
4. For video training (LTX/WAN), configure the normalizer (resolution, FPS, duration, etc.) — TrainFlow auto-generates the Musubi dataset TOML and caches text/latents when parameters change.
5. Set trigger word, rank, optimizer, steps, and preview settings — or click **Auto Calc** for a profile-aware starting point based on your dataset size.
6. Click **Start**.
7. Click **Quit** in the top bar when finished.

---

## Training Profiles

| Profile | Model Path | Network | Bucket Step | Notes |
|---------|-----------|---------|-------------|-------|
| **Anima** | DiT + Qwen3 + VAE | `networks.lora_anima` | 64px | DiT/Qwen3 text encoder, Anima metadata |
| **SDXL / Pony / Illustrious** | checkpoint (+ optional VAE) | `networks.lora` | 32px | UNet/text-encoder LR fields, SDPA by default |
| **LTX 2.3 / Wan 2.2** | video checkpoint | Musubi pipeline | — | Auto TOML, video normalization, sequenced pipeline |

**Auto Calc** reads the selected profile and dataset image count, then updates rank, learning rates, batch, gradient accumulation, training steps, save interval, and sample interval. It preserves the selected optimizer (Prodigy stays on `lr=1.0` constant; AdamW/AdamW8bit stay on `1e-4` cosine).

---

## Features

### Embedded Training GUI
- Single portable Go binary with embedded HTML/CSS/JS — no separate web build step
- Profile switcher for Anima, SDXL/Pony/Illustrious, and LTX 2.3/Wan 2.2
- Clickable local path browser for datasets, model files, and resume-state folders
- Autosaved settings in `training/settings.json` with a manual **Save** button
- Optimizer-aware defaults for Prodigy, AdamW8bit, and AdamW
- Training controls: rank, learning rates, batch size, steps, gradient accumulation, save/sample interval
- SDXL-specific UNet/text-encoder learning-rate fields and **UNet only** toggle
- Optional Anima Flash Attention and `torch.compile` (per-block DiT compilation)
- Resume panel with automatic latest-state discovery
- Sample prompt editor (positive/negative, width, height, CFG, seed)
- Live training log streamed from the running process
- Preview gallery with image overlay for browsing outputs
- **Output** button to open the current project output folder

### Dataset Prep GUI
- WD EVA02 ONNX caption/tagging
- LTX/WAN video normalization controls (resolution, FPS, duration, codec, quality, workers, etc.)
- Automatic Musubi dataset TOML generation and cache rebuild triggers
- Resize-copy helper (`training/prepared/<project>`)
- Combined **Tag + Resize** workflow with configurable thresholds

### Runtime & Model Tooling
- Companion `TrainFlow_Runtime_Tool` at `http://127.0.0.1:7870`
- **Verify Runtime**, **Update Runtime**, **Install Requirements**
- **Download Models** for Anima base files; **Download Prep** for tagger/U2Net assets
- Status pills in the main UI with quick-launch buttons
- Platform-specific portable Python (`python_embeded/windows` or `python_embeded/linux`)
- uv-first dependency installation with pip fallback
- PyTorch backend selector: CUDA 13.0 by default, experimental ROCm 6.4, or existing user-managed PyTorch
- GPU auto-detection (NVIDIA via `nvidia-smi`, AMD via `lspci`/sysfs/PowerShell) with vendor badge in the header
- Vendor-colored backend selector panel (green for NVIDIA/CUDA, red for AMD/ROCm) with mismatch warnings

> **ROCm / custom PyTorch warning:** CUDA 13.0 remains the fully supported default. The ROCm and existing-PyTorch runtime modes are advanced/experimental and are intended for users who already know their PyTorch build works with their hardware. When ROCm or existing PyTorch is selected, TrainFlow disables CUDA-only optional installers such as Flash Attention and Anima `torch.compile`/Triton deps. Note that AMD GPU monitoring on Linux works via kernel sysfs and does not require ROCm.

### Hardware Monitoring
- Compact overlay inside the sampler panel
- CPU usage, RAM usage, CPU temperature (when available)
- NVIDIA GPU utilization, memory, temperature, and active task labels (via `nvidia-smi`)
- AMD GPU monitoring on Linux via `amdgpu` sysfs (utilization, VRAM, temperature, power, frequency — kernel driver only, no ROCm needed)
- AMD GPU detection on Windows via WMI (name and VRAM total)

---

## Advanced

<details>
<summary><strong>Resume Training</strong> — state resume, latest-state discovery</summary>

The UI includes a **Resume** panel:

- **Resume training** enables `sd-scripts` state resume
- **Use latest saved state** auto-discovers the newest `*-state` directory in the project output folder
- **Resume State Path** lets you pick a specific state folder manually

All new runs write `save_state = true`, `save_last_n_steps_state = 1`, and `save_last_n_epochs_state = 1` so state is always available.

</details>

<details>
<summary><strong>Anima LoRA Metadata</strong> — inspect and repair safetensors headers</summary>

New LoRA files include Anima-specific safetensors metadata (`ss_base_model_version = "anima-base-v1.0"`). To inspect or repair existing LoRAs:

```bash
python training/sd-scripts/tools/anima_lora_metadata.py path/to/lora.safetensors
python training/sd-scripts/tools/anima_lora_metadata.py path/to/lora.safetensors --fix
```

</details>

<details>
<summary><strong>Required Models</strong> — Anima base files, prep models, manual install</summary>

Use **Download Models** in `TrainFlow_Runtime_Tool` to fetch Anima base files:

```
models/anima/dit/anima-base-v1.0.safetensors
models/anima/text_encoder/qwen_3_06b_base.safetensors
models/anima/vae/qwen_image_vae.safetensors
```

Use **Download Prep** for optional dataset-prep models:

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
<summary><strong>System Requirements</strong> — GPU, Python, host dependencies</summary>

| Requirement | Notes |
|-------------|-------|
| **GPU** | NVIDIA (CUDA) recommended; AMD (ROCm 6.4) supported on Linux |
| **Python 3.12** | Recommended on Linux for the embedded runtime |
| **nvidia-smi** | Needed for NVIDIA GPU monitoring overlay |
| **amdgpu kernel driver** | Needed for AMD GPU monitoring on Linux (sysfs) |
| **Go 1.22+** | Only when building from source |

</details>

---

## Development

<details>
<summary><strong>Build From Source</strong> — compile Go binaries, cross-compile</summary>

Only needed when modifying Go code or rebuilding release artifacts.

**Linux:**
```bash
go build -trimpath -ldflags="-s -w" -o TrainFlow ./cmd/trainflow
go build -trimpath -ldflags="-s -w" -o TrainFlow_Runtime_Tool ./cmd/runtime-tool
```

**Cross-compile (Windows PowerShell):**
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
<summary><strong>Shipping & Distribution</strong> — release packaging, python_embeded</summary>

Do not commit `python_embeded/` to Git — the runtime contains thousands of files plus large ML wheels.

**For normal installs**, ship only the root binaries and let the runtime tool create the platform runtime on the user's machine:
- Windows: `TrainFlow.exe` + `TrainFlow_Runtime_Tool.exe`
- Linux: `TrainFlow` + `TrainFlow_Runtime_Tool`

**For fully offline packages**, create a release ZIP/7z outside Git containing the binaries plus `python_embeded/<platform>`, and upload as a GitHub Release asset.

</details>

---

## Credits

Built on the [`sd-scripts`](https://github.com/kohya-ss/sd-scripts) training stack. Video training powered by [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). The original [Anima TrainFlow](https://github.com/ThetaCursed/Anima-TrainFlow) by ThetaCursed inspired the portable trainer concept.
