const logs = document.getElementById("logs");
const statusText = document.getElementById("statusText");
const modelStatus = document.getElementById("modelStatus");
const osBadge = document.getElementById("osBadge");
const gpuBadge = document.getElementById("gpuBadge");
const keepBackup = document.getElementById("keepBackup");
const installFlashAttention = document.getElementById("installFlashAttention");
const installTorchCompile = document.getElementById("installTorchCompile");
const torchBackend = document.getElementById("torchBackend");
const torchWarning = document.getElementById("torchWarning");
const torchPanel = torchBackend.closest(".torch-panel");
const buttons = {
  install: document.getElementById("installButton"),
  update: document.getElementById("updateButton"),
  models: document.getElementById("modelsButton"),
  prepModels: document.getElementById("prepModelsButton"),
  verify: document.getElementById("verifyButton"),
  quit: document.getElementById("quitButton")
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function run(action) {
  const resp = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({
      action,
      keepBackup: keepBackup.checked,
      installFlashAttention: installFlashAttention.checked,
      installTorchCompile: installTorchCompile.checked,
      torchBackend: torchBackend.value
    })
  });
  statusText.textContent = resp.message;
}

async function quitApp() {
  const resp = await api("/api/app/quit", { method: "POST" });
  statusText.textContent = resp.message;
  for (const button of Object.values(buttons)) {
    button.disabled = true;
  }
  keepBackup.disabled = true;
  installFlashAttention.disabled = true;
  installTorchCompile.disabled = true;
  torchBackend.disabled = true;
}

// Detect GPU vendor from backend string (e.g. "NVIDIA GeForce RTX 5090", "AMD Radeon RX 7900 XTX")
function detectedVendor() {
  const gpu = gpuBadge.textContent || "";
  const lower = gpu.toLowerCase();
  if (lower.includes("nvidia") || lower.includes("geforce") || lower.includes("rtx") || lower.includes("quadro")) return "nvidia";
  if (lower.includes("amd") || lower.includes("radeon") || lower.includes("radon")) return "amd";
  return "";
}

// Check if the selected backend matches the detected GPU vendor
function backendMatchesGPU() {
  const vendor = detectedVendor();
  if (!vendor) return true; // No GPU detected, no mismatch
  const backend = torchBackend.value;
  if (vendor === "nvidia" && backend === "rocm") return false;
  if (vendor === "amd" && backend === "cuda13") return false;
  return true;
}

function renderTorchBackend() {
  const backend = torchBackend.value;
  const cudaOnlyDisabled = backend !== "cuda13";
  if (cudaOnlyDisabled) {
    installFlashAttention.checked = false;
    installTorchCompile.checked = false;
  }
  installFlashAttention.disabled = cudaOnlyDisabled;
  installTorchCompile.disabled = cudaOnlyDisabled;

  // Vendor-colored panel
  torchPanel.classList.remove("backend-nvidia", "backend-amd", "backend-existing");
  if (backend === "cuda13") torchPanel.classList.add("backend-nvidia");
  else if (backend === "rocm") torchPanel.classList.add("backend-amd");
  else torchPanel.classList.add("backend-existing");

  if (backend === "rocm") {
    torchWarning.hidden = false;
    torchWarning.textContent = "WARNING: ROCm runtime install is experimental. TrainFlow will install ROCm PyTorch, but CUDA-only optional installers are disabled: Flash Attention and torch.compile/Triton deps will NOT be installed. NVIDIA GPU monitoring via nvidia-smi will not work. Use only if your AMD GPU and ROCm stack are supported by PyTorch.";
  } else if (backend === "skip") {
    torchWarning.hidden = false;
    torchWarning.textContent = "WARNING: Existing PyTorch mode will not install or upgrade torch/torchvision/torchaudio. CUDA-only optional installers are disabled automatically; install any compatible acceleration packages yourself.";
  } else {
    torchWarning.hidden = true;
    torchWarning.textContent = "";
  }

  // Show mismatch warning
  if (!backendMatchesGPU()) {
    const vendor = detectedVendor();
    const vendorLabel = vendor === "nvidia" ? "NVIDIA" : "AMD";
    const backendLabel = backend === "rocm" ? "ROCm (AMD)" : "CUDA (NVIDIA)";
    torchWarning.hidden = false;
    if (backend === "rocm" || backend === "cuda13") {
      // Append to existing warning
      torchWarning.textContent += ` MISMATCH: Your detected GPU is ${vendorLabel} but you selected ${backendLabel}. This may not work.`;
    } else {
      torchWarning.textContent = `WARNING: Your detected GPU is ${vendorLabel} but you selected ${backendLabel}. This may not work.`;
    }
  }
}

function renderStatus(data) {
  osBadge.textContent = data.os || "";
  if (data.gpu && data.gpu !== "") {
    gpuBadge.textContent = data.gpu;
    gpuBadge.hidden = false;
  } else {
    gpuBadge.hidden = true;
  }
  const lines = data.logs || [];
  logs.textContent = lines.join("\n");
  logs.scrollTop = logs.scrollHeight;
  for (const button of Object.values(buttons)) {
    button.disabled = Boolean(data.running);
  }
  buttons.quit.disabled = false;
  keepBackup.disabled = Boolean(data.running);
  torchBackend.disabled = Boolean(data.running);
  renderTorchBackend();
  if (data.running) {
    installFlashAttention.disabled = true;
    installTorchCompile.disabled = true;
  }
  statusText.textContent = data.running ? "Running" : "Ready";
  renderModelStatus(data.models);
}

function renderModelStatus(status) {
  if (!status) return;
  const required = status.ready ? "Models ready" : `Models missing ${status.missing}`;
  const optional = status.optional_ready ? "Prep ready" : `Prep missing ${status.optional_missing}`;
  modelStatus.textContent = `${required} / ${optional}`;
  modelStatus.title = (status.files || [])
    .map((file) => `${file.ok ? "OK" : "Missing"}: ${file.found || file.path}`)
    .join("\n");
  modelStatus.className = `status-pill ${status.ready && status.optional_ready ? "ready" : "missing"}`;
}

function connectWS() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws`);
  ws.addEventListener("close", () => setTimeout(connectWS, 1000));
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "status") renderStatus(msg.data);
  });
}

async function runWithConfirm(action) {
  if (!backendMatchesGPU()) {
    const vendor = detectedVendor();
    const vendorLabel = vendor === "nvidia" ? "NVIDIA" : "AMD";
    const backendLabel = torchBackend.value === "rocm" ? "ROCm (AMD)" : "CUDA (NVIDIA)";
    const ok = confirm(`GPU mismatch detected!\n\nYour GPU: ${gpuBadge.textContent}\nSelected backend: ${backendLabel}\n\nThis combination may not work. Continue anyway?`);
    if (!ok) return;
  }
  await run(action);
}

buttons.install.addEventListener("click", () => runWithConfirm("install"));
buttons.update.addEventListener("click", () => runWithConfirm("update"));
buttons.models.addEventListener("click", () => run("models"));
buttons.prepModels.addEventListener("click", () => run("prep-models"));
buttons.verify.addEventListener("click", () => run("verify"));
buttons.quit.addEventListener("click", () => quitApp().catch((err) => (statusText.textContent = err.message)));
torchBackend.addEventListener("change", renderTorchBackend);

api("/api/status").then(renderStatus).catch((err) => {
  statusText.textContent = err.message;
});
renderTorchBackend();
connectWS();
