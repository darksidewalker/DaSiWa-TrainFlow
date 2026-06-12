const fields = [
  "trigger_word",
  "dataset_path",
  "dit_path",
  "qwen_path",
  "vae_path",
  "network_rank",
  "learning_rate",
  "optimizer",
  "training_steps",
  "save_steps",
  "sample_steps",
  "pos_prompt",
  "neg_prompt",
  "width",
  "height",
  "sample_cfg",
  "sample_seed",
  "train_batch_size",
  "gradient_accumulation_steps",
  "train_unet_only",
  "resume_enabled",
  "auto_resume",
  "resume_path",
  "side_min",
  "side_max",
  "tagger_gen_thresh",
  "tagger_char_thresh",
  "tagger_overwrite"
];

const numericFields = new Set([
  "network_rank",
  "training_steps",
  "save_steps",
  "sample_steps",
  "width",
  "height",
  "sample_cfg",
  "sample_seed",
  "train_batch_size",
  "gradient_accumulation_steps",
  "side_min",
  "side_max",
  "tagger_gen_thresh",
  "tagger_char_thresh"
]);

const els = Object.fromEntries(fields.map((id) => [id, document.getElementById(id)]));
const logs = document.getElementById("logs");
const statusText = document.getElementById("statusText");
const runtimeStatus = document.getElementById("runtimeStatus");
const runtimeLaunch = document.getElementById("runtimeLaunch");
const modelStatus = document.getElementById("modelStatus");
const modelLaunch = document.getElementById("modelLaunch");
const socketState = document.getElementById("socketState");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const quitButton = document.getElementById("quitButton");
const saveButton = document.getElementById("saveButton");
const monitorToggle = document.getElementById("monitorToggle");
const hardwareOverlay = document.getElementById("hardwareOverlay");
const gallery = document.getElementById("gallery");
const imageOverlay = document.getElementById("imageOverlay");
const overlayImage = document.getElementById("overlayImage");
const overlayCaption = document.getElementById("overlayCaption");
const overlayClose = document.getElementById("overlayClose");
const overlayPrev = document.getElementById("overlayPrev");
const overlayNext = document.getElementById("overlayNext");
const pathDialog = document.getElementById("pathDialog");
const pathDialogTitle = document.getElementById("pathDialogTitle");
const pathClose = document.getElementById("pathClose");
const pathCancel = document.getElementById("pathCancel");
const pathChoose = document.getElementById("pathChoose");
const pathUp = document.getElementById("pathUp");
const pathGo = document.getElementById("pathGo");
const pathCurrent = document.getElementById("pathCurrent");
const pathRoots = document.getElementById("pathRoots");
const pathEntries = document.getElementById("pathEntries");

let galleryImages = [];
let overlayIndex = 0;

let saveTimer = 0;
let runtimePollTimer = 0;
let modelPollTimer = 0;
let running = false;
let picker = {
  target: "",
  mode: "directory",
  selected: ""
};

function collectSettings() {
  const data = {};
  for (const id of fields) {
    const el = els[id];
    if (el.type === "checkbox") {
      data[id] = el.checked;
    } else if (numericFields.has(id)) {
      data[id] = Number(el.value);
    } else {
      data[id] = el.value;
    }
  }
  data.train_seed = 42;
  data.sample_steps_gen = 30;
  return data;
}

function applySettings(data) {
  for (const id of fields) {
    if (!els[id] || data[id] === undefined) continue;
    if (els[id].type === "checkbox") {
      els[id].checked = Boolean(data[id]);
    } else {
      els[id].value = data[id];
    }
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function saveSettings() {
  saveButton.disabled = true;
  try {
    await api("/api/settings", {
      method: "POST",
      body: JSON.stringify(collectSettings())
    });
    saveButton.textContent = "Saved";
    setTimeout(() => (saveButton.textContent = "Save"), 700);
  } catch (err) {
    setStatus(err.message, false);
  } finally {
    saveButton.disabled = false;
  }
}

function queueSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSettings, 450);
}

async function startTraining() {
  setRunning(true);
  try {
    const resp = await api("/api/train/start", {
      method: "POST",
      body: JSON.stringify(collectSettings())
    });
    setStatus(resp.message, resp.ok);
    if (!resp.ok) setRunning(false);
  } catch (err) {
    setStatus(err.message, false);
    setRunning(false);
  }
}

async function stopTraining() {
  try {
    const resp = await api("/api/train/stop", { method: "POST" });
    setStatus(resp.message, resp.ok);
  } catch (err) {
    setStatus(err.message, false);
  }
}

async function quitApp() {
  try {
    const resp = await api("/api/app/quit", { method: "POST" });
    setStatus(resp.message, resp.ok);
    startButton.disabled = true;
    stopButton.disabled = true;
    saveButton.disabled = true;
    quitButton.disabled = true;
  } catch (err) {
    setStatus(err.message, false);
  }
}

async function refreshRuntimeStatus() {
  try {
    const status = await api("/api/runtime");
    renderRuntimeStatus(status);
    if (!status.ready && runtimePollTimer === 0) {
      runtimePollTimer = window.setInterval(refreshRuntimeStatus, 5000);
    }
    if (status.ready && runtimePollTimer !== 0) {
      window.clearInterval(runtimePollTimer);
      runtimePollTimer = 0;
    }
  } catch (err) {
    runtimeStatus.textContent = "Runtime check failed";
    runtimeStatus.title = err.message;
    runtimeStatus.className = "runtime-pill error";
    runtimeLaunch.classList.remove("hidden");
  }
}

async function refreshModelStatus() {
  try {
    const status = await api("/api/models");
    renderModelStatus(status);
    if (!status.ready && modelPollTimer === 0) {
      modelPollTimer = window.setInterval(refreshModelStatus, 5000);
    }
    if (status.ready && modelPollTimer !== 0) {
      window.clearInterval(modelPollTimer);
      modelPollTimer = 0;
    }
  } catch (err) {
    modelStatus.textContent = "Models check failed";
    modelStatus.title = err.message;
    modelStatus.className = "runtime-pill error";
    modelLaunch.classList.remove("hidden");
  }
}

function renderRuntimeStatus(status) {
  runtimeStatus.textContent = status.ready ? "Runtime ready" : "Runtime missing";
  runtimeStatus.title = status.ready ? status.path : `Expected at ${status.expected}`;
  runtimeStatus.className = `runtime-pill ${status.ready ? "ready" : "missing"}`;
  runtimeLaunch.classList.toggle("hidden", Boolean(status.ready));
  runtimeLaunch.disabled = false;
}

function renderModelStatus(status) {
  if (status.ready && status.optional_ready) {
    modelStatus.textContent = "Models ready";
  } else if (status.ready) {
    modelStatus.textContent = `Prep missing ${status.optional_missing}`;
  } else {
    modelStatus.textContent = `Models missing ${status.missing}`;
  }
  modelStatus.title = (status.files || [])
    .map((file) => `${file.ok ? "OK" : "Missing"}: ${file.found || file.path}`)
    .join("\n");
  modelStatus.className = `runtime-pill ${status.ready && status.optional_ready ? "ready" : "missing"}`;
  modelLaunch.classList.toggle("hidden", Boolean(status.ready && status.optional_ready));
  modelLaunch.disabled = false;
}

async function launchRuntimeTool() {
  runtimeLaunch.disabled = true;
  modelLaunch.disabled = true;
  try {
    const resp = await api("/api/runtime/launch", { method: "POST" });
    setStatus(resp.message, resp.ok);
    if (!resp.ok) {
      runtimeLaunch.disabled = false;
      modelLaunch.disabled = false;
      return;
    }
    await refreshRuntimeStatus();
    await refreshModelStatus();
  } catch (err) {
    setStatus(err.message, false);
    runtimeLaunch.disabled = false;
    modelLaunch.disabled = false;
  }
}

function setStatus(text, ok = true) {
  statusText.textContent = text;
  statusText.style.color = ok ? "var(--muted)" : "var(--rose)";
}

function setRunning(value) {
  running = value;
  startButton.disabled = value;
  stopButton.disabled = !value;
  statusText.textContent = value ? "Training" : "Idle";
}

function setLogs(value) {
  logs.textContent = value || "";
  logs.scrollTop = logs.scrollHeight;
}

async function openPathPicker(target, mode) {
  picker = {
    target,
    mode,
    selected: els[target]?.value || ""
  };
  pathDialogTitle.textContent = `Choose ${target.replaceAll("_", " ")}`;
  pathDialog.classList.remove("hidden");
  await loadPath(picker.selected, mode);
}

async function loadPath(path, mode = picker.mode) {
  const params = new URLSearchParams({ path: path || "", mode });
  const data = await api(`/api/path/list?${params.toString()}`, { headers: {} });
  pathCurrent.value = data.path;
  picker.selected = data.path;
  renderRoots(data.roots || []);
  renderPathEntries(data.entries || [], data.parent);
}

function renderRoots(roots) {
  pathRoots.innerHTML = "";
  for (const root of roots) {
    const button = document.createElement("button");
    button.className = "secondary";
    button.textContent = root.name;
    button.addEventListener("click", () => loadPath(root.path));
    pathRoots.append(button);
  }
}

function renderPathEntries(entries, parent) {
  pathEntries.innerHTML = "";
  pathUp.onclick = () => loadPath(parent || pathCurrent.value);
  for (const entry of entries) {
    const row = document.createElement("button");
    row.className = `path-entry ${entry.isDir ? "directory" : "file"}`;
    row.innerHTML = `<span>${entry.isDir ? "/" : ""}${escapeHTML(entry.name)}</span><small>${escapeHTML(entry.path)}</small>`;
    row.addEventListener("click", () => {
      picker.selected = entry.path;
      pathCurrent.value = entry.path;
      if (entry.isDir) {
        loadPath(entry.path);
      }
    });
    row.addEventListener("dblclick", () => {
      if (!entry.isDir || picker.mode === "directory") chooseCurrentPath();
    });
    pathEntries.append(row);
  }
}

function closePathPicker() {
  pathDialog.classList.add("hidden");
}

function chooseCurrentPath() {
  if (!picker.target) return;
  els[picker.target].value = pathCurrent.value || picker.selected;
  els[picker.target].dispatchEvent(new Event("change", { bubbles: true }));
  closePathPicker();
}

function renderGallery(images) {
  galleryImages = images ? images.slice(0, 8) : [];
  gallery.innerHTML = "";
  if (!galleryImages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-gallery";
    empty.textContent = "No previews yet";
    gallery.append(empty);
    return;
  }

  galleryImages.forEach((image, index) => {
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `${image.src}?t=${Date.now()}`;
    img.alt = image.name;
    const cap = document.createElement("figcaption");
    cap.textContent = image.name;
    figure.append(img, cap);
    figure.addEventListener("click", () => openImageOverlay(index));
    gallery.append(figure);
  });
}

function openImageOverlay(index) {
  if (!galleryImages.length) return;
  overlayIndex = ((index % galleryImages.length) + galleryImages.length) % galleryImages.length;
  const image = galleryImages[overlayIndex];
  overlayImage.src = `${image.src}?t=${Date.now()}`;
  overlayImage.alt = image.name;
  overlayCaption.textContent = image.name;
  imageOverlay.classList.remove("hidden");
}

function closeImageOverlay() {
  imageOverlay.classList.add("hidden");
}

function showOverlayIndex(delta) {
  openImageOverlay(overlayIndex + delta);
}

function connectWS() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws`);
  ws.addEventListener("open", () => {
    socketState.textContent = "online";
  });
  ws.addEventListener("close", () => {
    socketState.textContent = "offline";
    setTimeout(connectWS, 1000);
  });
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "hw_stats") renderHardware(msg.data);
    if (msg.type === "log") {
      setLogs(msg.data.logs);
      setRunning(Boolean(msg.data.running));
    }
    if (msg.type === "images") renderGallery(msg.data);
    if (msg.type === "training_state") setRunning(Boolean(msg.data.running));
  });
}

function renderHardware(data) {
  const cpu = Math.max(0, Math.min(100, data.cpu || 0));
  document.getElementById("cpuValue").textContent = `${cpu}%${data.cpuTemp ? ` / ${data.cpuTemp}C` : ""}`;
  document.getElementById("cpuBar").style.width = `${cpu}%`;

  const total = data.ram?.total || 0;
  const used = data.ram?.used || 0;
  const ramPct = total ? Math.round((used / total) * 100) : 0;
  document.getElementById("ramValue").textContent = `${formatBytes(used)} / ${formatBytes(total)}`;
  document.getElementById("ramBar").style.width = `${ramPct}%`;

  const gpuList = document.getElementById("gpuList");
  gpuList.innerHTML = "";
  if (!data.gpus || data.gpus.length === 0) {
    const empty = document.createElement("div");
    empty.className = "gpu-card";
    empty.innerHTML = "<span>No NVIDIA GPU stats</span>";
    gpuList.append(empty);
    return;
  }
  for (const gpu of data.gpus) {
    const memPct = gpu.memTotal ? Math.round((gpu.memUsed / gpu.memTotal) * 100) : 0;
    const card = document.createElement("div");
    card.className = "gpu-card";
    card.innerHTML = `
      <div class="gpu-title"><span>${escapeHTML(gpu.name)}</span><em>${gpu.activity || "idle"}</em></div>
      <div class="metric">
        <span>GPU ${gpu.index} / ${gpu.temp}C / ${gpu.powerDraw}W</span>
        <strong>${gpu.util}%</strong>
        <div class="bar"><i style="width:${Math.max(0, Math.min(100, gpu.util))}%"></i></div>
      </div>
      <div class="metric">
        <span>VRAM</span>
        <strong>${gpu.memUsed} / ${gpu.memTotal} MB</strong>
        <div class="bar"><i style="width:${memPct}%"></i></div>
      </div>`;
    gpuList.append(card);
  }
}

function formatBytes(value) {
  if (!value) return "0 GB";
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function boot() {
  const settings = await api("/api/settings");
  applySettings(settings);
  const status = await api("/api/status");
  setRunning(Boolean(status.running));
  setLogs(status.logs);
  renderGallery(status.images);
  await refreshRuntimeStatus();
  await refreshModelStatus();
  connectWS();
}

for (const field of Object.values(els)) {
  field.addEventListener("input", queueSave);
  field.addEventListener("change", queueSave);
}

for (const button of document.querySelectorAll(".browse-button")) {
  button.addEventListener("click", () => openPathPicker(button.dataset.target, button.dataset.mode));
}

saveButton.addEventListener("click", saveSettings);
startButton.addEventListener("click", startTraining);
stopButton.addEventListener("click", stopTraining);
quitButton.addEventListener("click", quitApp);
runtimeLaunch.addEventListener("click", launchRuntimeTool);
modelLaunch.addEventListener("click", launchRuntimeTool);
monitorToggle.addEventListener("click", () => hardwareOverlay.classList.toggle("hidden"));
pathClose.addEventListener("click", closePathPicker);
pathCancel.addEventListener("click", closePathPicker);
pathChoose.addEventListener("click", chooseCurrentPath);
pathGo.addEventListener("click", () => loadPath(pathCurrent.value));
pathCurrent.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadPath(pathCurrent.value);
});

overlayClose.addEventListener("click", closeImageOverlay);
overlayPrev.addEventListener("click", (event) => {
  event.stopPropagation();
  showOverlayIndex(-1);
});
overlayNext.addEventListener("click", (event) => {
  event.stopPropagation();
  showOverlayIndex(1);
});
imageOverlay.addEventListener("click", (event) => {
  if (event.target === imageOverlay) closeImageOverlay();
});
window.addEventListener("focus", refreshRuntimeStatus);
window.addEventListener("focus", refreshModelStatus);
window.addEventListener("keydown", (event) => {
  if (!imageOverlay || imageOverlay.classList.contains("hidden")) return;
  if (event.key === "Escape") {
    closeImageOverlay();
    return;
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    showOverlayIndex(-1);
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    showOverlayIndex(1);
  }
});

boot().catch((err) => setStatus(err.message, false));
