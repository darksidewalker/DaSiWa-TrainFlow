# Runtime Upgrade Notes

`python_embeded/` is a generated local runtime. Keep it out of Git; the folder can contain very large dependency wheels and thousands of installed package files. Use `TrainFlow_Runtime_Tool` to recreate it, or distribute a compressed release archive outside the repository if you need an offline package.

## Python 3.12 + CUDA 13.0

Run this on Windows from the project root:

```bat
TrainFlow_Runtime_Tool.exe
```

Then click **Update Runtime**.

The tool:

- backs up `python_embeded/windows` to `python_embeded_windows_backup_YYYYMMDD_HHMMSS` during replacement
- deletes that backup after a successful update unless **Keep backup before update** is checked
- keeps the backup automatically if the update fails
- downloads the official Python 3.12.10 embeddable package
- enables `site-packages` in `python312._pth`
- installs pip
- installs `uv` into the embedded runtime
- installs Python dependencies with `uv pip install --python <embedded-python>` when available
- falls back to `python -m pip install` if uv bootstrap or uv install fails
- installs PyTorch/torchvision/torchaudio from `https://download.pytorch.org/whl/cu130`
- reinstalls `training/sd-scripts` requirements and the editable sd-scripts package
- installs Flash Attention from a prebuilt wheel only when selected; CUDA 13.0 uses the default torch/SDPA attention path when no wheel exists
- prints Python, Torch, CUDA, and CUDA availability at the end

Python 3.12.10 is used intentionally: Python.org says Python 3.12.10 was the last 3.12 release with Windows binary installers/embeddable packages. Newer 3.12 security releases are source-only.

If the CUDA 13.0 wheel set is not compatible with a dependency in `sd-scripts`, restore the backup folder or reinstall with CUDA 12.8:

```bat
python_embeded\windows\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Linux

There is no official Python "embeddable package" equivalent for Linux. The clickable runtime tool creates a local venv at `python_embeded/linux`, which gives the app a portable folder layout:

```bash
./TrainFlow_Runtime_Tool
```

Then click **Update Runtime** or **Install Requirements**.

Dependency installation also uses uv first on Linux. Everything still targets the local `python_embeded/linux` interpreter, so packages are installed into the app runtime rather than your system Python.

Flash Attention is optional. The runtime tool first tries the upstream release wheel name that matches the current Python, Torch, CUDA, C++ ABI, and platform, then asks pip/uv for a binary-only PyPI wheel. If none exists, CUDA 13.0 skips local compilation and uses the default `attn_mode = "torch"` path, which relies on PyTorch SDPA. Non-CUDA-13 source builds remain opt-in with `DASIWA_FLASH_ATTN_SOURCE_BUILD=1`, run single-job, and are skipped unless available RAM plus swap is at least 96 GiB.

The Go launcher checks these paths first:

```text
python_embeded/linux/bin/python3
python_embeded/linux/bin/python
```

If `python3.12` is available on `PATH`, the updater uses it. Otherwise it falls back to `python3`.
