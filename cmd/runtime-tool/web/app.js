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
const modelPicker = document.getElementById("modelPicker");
const modelGroups = document.getElementById("modelGroups");
const startModelsButton = document.getElementById("startModelsButton");
const selectMissingModels = document.getElementById("selectMissingModels");
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
  const modelKeys = action === "models" ? selectedModelKeys() : [];
  const resp = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({
      action,
      keepBackup: keepBackup.checked,
      installFlashAttention: installFlashAttention.checked,
      installTorchCompile: installTorchCompile.checked,
      torchBackend: torchBackend.value,
      modelKeys
    })
  });
  statusText.textContent = resp.message;
}

function modelSelectionKey(file) {
  return `${file.arch}:${file.key}:${(file.path || "").split(/[\\/]/).pop()}`;
}

function selectedModelKeys() {
  return Array.from(modelGroups.querySelectorAll("input[data-model-key]:checked")).map((input) => input.dataset.modelKey);
}

function renderModelCatalog(catalog) {
  if (!catalog || !modelGroups) return;
  modelGroups.innerHTML = "";
  for (const group of catalog) {
    const section = document.createElement("section");
    section.className = "model-group";
    const title = document.createElement("h3");
    title.textContent = group.label || group.architecture;
    section.appendChild(title);

    if (!group.files || group.files.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted-small";
      empty.textContent = "Noch keine Standard-Downloads hinterlegt.";
      section.appendChild(empty);
      modelGroups.appendChild(section);
      continue;
    }

    for (const file of group.files) {
      const label = document.createElement("label");
      label.className = "model-row";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.dataset.modelKey = modelSelectionKey(file);
      input.checked = !file.ok;
      const meta = document.createElement("span");
      meta.className = "model-meta";
      const name = document.createElement("strong");
      name.textContent = `${file.category || "Model"}: ${file.name}`;
      const path = document.createElement("span");
      path.textContent = file.ok ? `OK: ${file.found || file.path}` : `${file.size || ""} → ${file.path}`;
      meta.appendChild(name);
      meta.appendChild(path);
      label.appendChild(input);
      label.appendChild(meta);
      section.appendChild(label);
    }
    modelGroups.appendChild(section);
  }
}

function selectMissingModelRows() {
  for (const input of modelGroups.querySelectorAll("input[data-model-key]")) {
    const rowText = input.closest(".model-row")?.textContent || "";
    input.checked = !rowText.includes("OK:");
  }
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
  renderModelCatalog(data.catalog);
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
buttons.models.addEventListener("click", () => {
  modelPicker.hidden = !modelPicker.hidden;
});
startModelsButton.addEventListener("click", () => run("models"));
selectMissingModels.addEventListener("click", selectMissingModelRows);
buttons.prepModels.addEventListener("click", () => run("prep-models"));
buttons.verify.addEventListener("click", () => run("verify"));
buttons.quit.addEventListener("click", () => quitApp().catch((err) => (statusText.textContent = err.message)));
torchBackend.addEventListener("change", renderTorchBackend);

api("/api/status").then(renderStatus).catch((err) => {
  statusText.textContent = err.message;
});
renderTorchBackend();
connectWS();
