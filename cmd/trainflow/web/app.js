const fields = [
  "architecture",
  "project_name",
  "trigger_word",
  "dataset_path",
  "output_path",
  "dit_path",
  "checkpoint_path",
  "qwen_path",
  "vae_path",
  "network_rank",
  "learning_rate",
  "unet_lr",
  "text_encoder_lr1",
  "text_encoder_lr2",
  "auto_trigger",
  "optimizer",
  "training_steps",
  "save_steps",
  "sample_steps",
  "prompt_1",
  "prompt_2",
  "prompt_3",
  "neg_prompt",
  "width",
  "height",
  "sample_cfg",
  "sample_seed",
  "train_batch_size",
  "gradient_accumulation_steps",
  "target_vram_percent",
  "train_unet_only",
  "flash_attention",
  "torch_compile",
  "torch_compile_mode",
  "torch_compile_backend",
  "torch_compile_dynamic",
  "torch_compile_fullgraph",
  "torch_compile_cache_size_limit",
  "cuda_allow_tf32",
  "cuda_cudnn_benchmark",
  "resume_enabled",
  "auto_resume",
  "resume_path",
  "side_min",
  "side_max",
  "tagger_gen_thresh",
  "tagger_char_thresh",
  "tagger_overwrite",
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
  "video_codec",
  "video_quality",
  "video_encoder_preset",
  "video_speed",
  "video_skip_frames",
  "video_parallel_workers",
  "video_include_audio",
  "video_extra_args",
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
  "target_vram_percent",
  "torch_compile_cache_size_limit",
  "side_min",
  "side_max",
  "tagger_gen_thresh",
  "tagger_char_thresh",
  "target_epochs",
  "num_cpu_threads",
  "video_width",
  "video_height",
  "video_fps",
  "video_num_repeats",
  "video_skip_frames",
  "video_parallel_workers",
  "blocks_to_swap",
  "network_alpha"
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
const autoCalcButton = document.getElementById("autoCalcButton");
const openOutputButton = document.getElementById("openOutputButton");
const tagDatasetButton = document.getElementById("tagDatasetButton");
const resizeDatasetButton = document.getElementById("resizeDatasetButton");
const prepDatasetButton = document.getElementById("prepDatasetButton");
const copyLogButton = document.getElementById("copyLogButton");
const clearLogButton = document.getElementById("clearLogButton");
const normalizeVideoButton = document.getElementById("normalizeVideoButton");
const writeMusubiDatasetButton = document.getElementById("writeMusubiDatasetButton");
const cacheMusubiTextButton = document.getElementById("cacheMusubiTextButton");
const cacheMusubiLatentsButton = document.getElementById("cacheMusubiLatentsButton");
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
const architectureButtons = Array.from(document.querySelectorAll(".architecture-button"));
const profileFields = Array.from(document.querySelectorAll(".profile-field"));
const optimizerLrFields = Array.from(document.querySelectorAll(".optimizer-lr-field"));
const datasetPrepTitle = document.getElementById("datasetPrepTitle");
const imageDatasetPrep = document.getElementById("imageDatasetPrep");
const videoDatasetPrep = document.getElementById("videoDatasetPrep");
const videoNormalizerProxyMap = {
  video_norm_width_proxy: "video_width",
  video_norm_height_proxy: "video_height",
  video_norm_fps_proxy: "video_fps",
  video_norm_duration_proxy: "video_duration"
};
const videoNormalizerProxies = Object.fromEntries(Object.keys(videoNormalizerProxyMap).map((id) => [id, document.getElementById(id)]));

// Div32-aligned resolution presets for LTX-Video 2.3 / WAN 2.2
// VRAM tiers: low = ~12-16GB (proxy/low-res), medium = ~24GB (balanced), high = ~32GB+ (max fidelity)
const videoResPresetTable = [
  { ratio: "16:9", type: "Cinematic Landscape", low: [768, 448], medium: [1024, 576], high: [1280, 704] },
  { ratio: "9:16", type: "Social / Mobile Vertical", low: [448, 768], medium: [576, 1024], high: [704, 1280] },
  { ratio: "4:3",  type: "Standard Landscape",    low: [640, 480], medium: [896, 672], high: [1024, 768] },
  { ratio: "3:4",  type: "Portrait / Character",   low: [480, 640], medium: [672, 896], high: [768, 1024] },
  { ratio: "1:1",  type: "Square / Subject Focus",  low: [512, 512], medium: [768, 768], high: [960, 960] },
  { ratio: "21:9", type: "Ultrawide",               low: [768, 320], medium: [1024, 448], high: [1344, 576] },
];
const videoResPresets = [];
for (const row of videoResPresetTable) {
  for (const tier of ["low", "medium", "high"]) {
    const [w, h] = row[tier];
    videoResPresets.push({ label: `${w}×${h} (${row.ratio} ${row.type})`, w, h, vram: tier });
  }
}

const videoResPreset = document.getElementById("video_res_preset");
const videoResFlip = document.getElementById("video_res_flip");
const videoResHint = document.getElementById("video_res_hint");
const videoResCustom = document.getElementById("video_res_custom");

let gpuVramMB = 0; // set from hardware stats
const ditLabel = document.getElementById("ditLabel");
const checkpointLabel = document.getElementById("checkpointLabel");
const qwenLabel = document.getElementById("qwenLabel");
const vaeLabel = document.getElementById("vaeLabel");

let galleryImages = [];
let overlayIndex = 0;

let saveTimer = 0;
let runtimePollTimer = 0;
let modelPollTimer = 0;
let running = false;
let musubiCacheDirty = false;
let picker = {
  target: "",
  mode: "directory",
  selected: ""
};

function collectSettings() {
  const data = {};
  for (const id of fields) {
    const el = els[id];
    if (!el) continue;
    if (el.type === "checkbox") {
      data[id] = el.checked;
    } else if (numericFields.has(id)) {
      data[id] = Number(el.value);
    } else {
      data[id] = el.value;
    }
  }
  // Build sample_prompts from the three prompt fields
  const prompts = [data.prompt_1, data.prompt_2, data.prompt_3].map((s) => s.trim()).filter(Boolean);
  data.sample_prompts = prompts.length ? prompts : [data.prompt_1 || ""];
  data.pos_prompt = data.sample_prompts[0] || "";
  data.architecture = normalizeArchitecture(data.architecture);
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
  // Populate prompt fields from sample_prompts or pos_prompt (backward compat)
  if (Array.isArray(data.sample_prompts) && data.sample_prompts.length) {
    els.prompt_1.value = data.sample_prompts[0] || "";
    els.prompt_2.value = data.sample_prompts[1] || "";
    els.prompt_3.value = data.sample_prompts[2] || "";
  } else if (data.pos_prompt) {
    els.prompt_1.value = data.pos_prompt;
  }
  syncVideoNormalizerProxiesFromSettings();
  setArchitecture(data.architecture || "anima", false);
  updateOptimizerLrFields();
}

function normalizeArchitecture(value) {
  return ["anima", "sdxl", "ltx23", "wan22"].includes(value) ? value : "anima";
}

function isVideoArchitecture(architecture) {
  return architecture === "ltx23" || architecture === "wan22";
}

function syncVideoNormalizerProxiesFromSettings() {
  for (const [proxyID, sourceID] of Object.entries(videoNormalizerProxyMap)) {
    const proxy = videoNormalizerProxies[proxyID];
    const source = els[sourceID];
    if (proxy && source) proxy.value = source.value;
  }
  // Update resolution dropdown to match current settings
  const w = els.video_width?.value;
  const h = els.video_height?.value;
  if (w && h) {
    selectResPresetByValue(`${w}x${h}`);
  }
}

function syncVideoNormalizerProxyToSetting(proxyID) {
  const sourceID = videoNormalizerProxyMap[proxyID];
  const proxy = videoNormalizerProxies[proxyID];
  const source = els[sourceID];
  if (!proxy || !source) return;
  source.value = proxy.value;
  source.dispatchEvent(new Event("change", { bubbles: true }));
}

// --- Resolution selector helpers ---

function populateResPresetDropdown() {
  if (!videoResPreset) return;
  videoResPreset.innerHTML = "";
  const customOpt = document.createElement("option");
  customOpt.value = "custom";
  customOpt.textContent = "Custom…";
  videoResPreset.appendChild(customOpt);

  const groups = { "16:9 Widescreen": [], "3:4 Portrait": [], "4:3 Landscape": [], "9:16 Vertical": [], "1:1 Square": [], "21:9 Ultrawide": [] };
  for (const p of videoResPresets) {
    if (p.label.includes("16:9")) groups["16:9 Widescreen"].push(p);
    else if (p.label.includes("21:9")) groups["21:9 Ultrawide"].push(p);
    else if (p.label.includes("3:4")) groups["3:4 Portrait"].push(p);
    else if (p.label.includes("4:3")) groups["4:3 Landscape"].push(p);
    else if (p.label.includes("9:16")) groups["9:16 Vertical"].push(p);
    else groups["1:1 Square"].push(p);
  }
  for (const [groupName, presets] of Object.entries(groups)) {
    if (!presets.length) continue;
    const og = document.createElement("optgroup");
    og.label = groupName;
    for (const p of presets) {
      const opt = document.createElement("option");
      opt.value = `${p.w}x${p.h}`;
      opt.textContent = p.label;
      opt.dataset.vram = p.vram;
      og.appendChild(opt);
    }
    videoResPreset.appendChild(og);
  }
}

function vramTier() {
  if (gpuVramMB >= 16384) return "high";
  if (gpuVramMB >= 12288) return "medium";
  return "low";
}

function updateResHint() {
  if (!videoResHint) return;
  if (gpuVramMB === 0) {
    videoResHint.textContent = "";
    return;
  }
  const tier = vramTier();
  const gb = Math.round(gpuVramMB / 1024);
  const tierLabel = tier === "high" ? `${gb} GB — high-res OK` : tier === "medium" ? `${gb} GB — medium recommended` : `${gb} GB — low-res recommended`;
  const recommended = videoResPresets.filter((p) => p.vram === tier);
  const examples = recommended.map((p) => p.label.split(" ")[0]).join(", ");
  videoResHint.textContent = `${tierLabel}: ${examples}`;
}

function selectResPresetByValue(value) {
  if (!videoResPreset) return;
  // Try to find matching preset
  const match = videoResPresets.find((p) => `${p.w}x${p.h}` === value);
  if (match) {
    videoResPreset.value = value;
    videoResCustom.classList.add("hidden");
  } else {
    videoResPreset.value = "custom";
    videoResCustom.classList.remove("hidden");
  }
}

function onResPresetChange() {
  if (!videoResPreset) return;
  const val = videoResPreset.value;
  if (val === "custom") {
    videoResCustom.classList.remove("hidden");
    return;
  }
  videoResCustom.classList.add("hidden");
  const [w, h] = val.split("x").map(Number);
  els.video_width.value = w;
  els.video_height.value = h;
  syncVideoNormalizerProxiesFromSettings();
  markMusubiCacheDirty("resolution changed");
  queueSave();
}

function onResFlip() {
  const w = Number(els.video_width.value);
  const h = Number(els.video_height.value);
  if (!w || !h) return;
  els.video_width.value = h;
  els.video_height.value = w;
  syncVideoNormalizerProxiesFromSettings();
  // After flip, the preset likely won't match, so show custom
  selectResPresetByValue(`${h}x${w}`);
  markMusubiCacheDirty("aspect flipped");
  queueSave();
}

// When custom W/H inputs change, sync back and switch dropdown to "Custom"
function onCustomResChange() {
  syncVideoNormalizerProxyToSetting("video_norm_width_proxy");
  syncVideoNormalizerProxyToSetting("video_norm_height_proxy");
  videoResPreset.value = "custom";
  videoResCustom.classList.remove("hidden");
  markMusubiCacheDirty("resolution changed");
  queueSave();
}

function setArchitecture(value, save = true) {
  const architecture = normalizeArchitecture(value);
  els.architecture.value = architecture;
  document.body.dataset.architecture = architecture;
  for (const button of architectureButtons) {
    button.classList.toggle("active", button.dataset.architecture === architecture);
    button.setAttribute("aria-pressed", String(button.dataset.architecture === architecture));
  }
  for (const field of profileFields) {
    const visible = field.classList.contains(`profile-${architecture}`);
    field.classList.toggle("hidden", !visible);
  }
  const videoMode = isVideoArchitecture(architecture);
  if (imageDatasetPrep) imageDatasetPrep.classList.toggle("hidden", videoMode);
  if (videoDatasetPrep) videoDatasetPrep.classList.toggle("hidden", !videoMode);
  if (datasetPrepTitle) datasetPrepTitle.textContent = videoMode ? "Video Normalize / Musubi Cache" : "Dataset Prep";
  syncVideoNormalizerProxiesFromSettings();
  if (architecture === "sdxl") {
    checkpointLabel.textContent = "SDXL Checkpoint";
    qwenLabel.textContent = "Qwen3";
    vaeLabel.textContent = "VAE (optional)";
    applyImageDefaults("sdxl");
  } else if (architecture === "ltx23") {
    checkpointLabel.textContent = "LTX 2.3 Checkpoint";
    qwenLabel.textContent = "Gemma Text Encoder";
    applyVideoDefaults("ltx23", save);
  } else if (architecture === "wan22") {
    ditLabel.textContent = "Wan DiT Checkpoint";
    qwenLabel.textContent = "T5 Text Encoder";
    vaeLabel.textContent = "Wan VAE";
    applyVideoDefaults("wan22", save);
  } else {
    ditLabel.textContent = "DiT";
    qwenLabel.textContent = "Qwen3";
    vaeLabel.textContent = "VAE";
    applyImageDefaults("anima");
  }
  normalizeOptimizerLearningRate();
  if (save) {
    queueSave();
    refreshModelStatus();
  }
}

function applyImageDefaults(architecture) {
  // Always set profile-aware rank/alpha — don't preserve stale values from other arches.
  if (architecture === "sdxl") {
    els.network_rank.value = 64;
    els.network_alpha.value = 32;
  } else {
    // anima default
    els.network_rank.value = 48;
    els.network_alpha.value = 32;
  }
}

function applyVideoDefaults(architecture, save) {
  if (!save) return;
  if (!els.optimizer.value) els.optimizer.value = "Prodigy";
  // Video models (LTX 2.3, WAN 2.2) use rank 64 / alpha 64 for detail + motion.
  els.network_rank.value = 64;
  els.network_alpha.value = 64;
  els.mixed_precision.value = els.mixed_precision.value || "bf16";
  els.num_cpu_threads.value = els.num_cpu_threads.value || 8;
  els.video_target_frames.value = els.video_target_frames.value || "1,65,129";
  els.video_frame_extraction.value = els.video_frame_extraction.value || "full";
  els.video_caption_extension.value = els.video_caption_extension.value || ".txt";
  els.video_enable_bucket.checked = true;
  els.blocks_to_swap.value = els.blocks_to_swap.value || 14;
  els.fp8_base.checked = true;
  els.fp8_scaled.checked = true;
  els.sdpa.checked = true;
  els.gradient_checkpointing.checked = true;
  els.use_pinned_memory_for_block_swap.checked = true;
  els.persistent_data_loader_workers.checked = true;
  els.save_state_on_train_end.checked = true;
  if (architecture === "ltx23") {
    els.network_module.value = "networks.lora_ltx2";
    els.timestep_sampling.value = "shifted_logit_normal";
    els.ltx_version.value = els.ltx_version.value || "2.3";
    els.ltx_mode.value = els.ltx_mode.value || "video";
    els.ltx_version_check_mode.value = els.ltx_version_check_mode.value || "error";
    if (!els.video_width.value) els.video_width.value = 768;
    if (!els.video_height.value) els.video_height.value = 512;
  } else {
    els.network_module.value = "networks.lora_wan";
    els.wan_task.value = els.wan_task.value || "i2v-A14B";
    els.timestep_sampling.value = "shift";
    els.discrete_flow_shift.value = els.discrete_flow_shift.value || "5.0";
    if (!els.video_width.value) els.video_width.value = 720;
    if (!els.video_height.value) els.video_height.value = 1280;
  }
  if (!els.video_fps.value) els.video_fps.value = 24;
  if (!els.video_duration.value) els.video_duration.value = 5;
  if (!els.video_codec.value) els.video_codec.value = "libx264";
  if (!els.video_quality.value) els.video_quality.value = "19";
  if (!els.video_encoder_preset.value) els.video_encoder_preset.value = "medium";
  if (!els.video_speed.value) els.video_speed.value = "1.0";
  if (!els.video_parallel_workers.value) els.video_parallel_workers.value = 1;
  if (els.video_skip_frames.value === "") els.video_skip_frames.value = 0;
  if (!els.video_num_repeats.value) els.video_num_repeats.value = 1;
  if (!els.target_epochs.value) els.target_epochs.value = 6;
  syncVideoNormalizerProxiesFromSettings();
}

function normalizeOptimizerLearningRate() {
  if (els.optimizer.value === "Prodigy") {
    els.learning_rate.value = "1.0";
  } else if (!els.learning_rate.value || els.learning_rate.value === "1.0") {
    els.learning_rate.value = "1e-4";
  }
  updateOptimizerLrFields();
}

function updateOptimizerLrFields() {
  const isProdigy = els.optimizer.value === "Prodigy";
  for (const field of optimizerLrFields) {
    field.classList.toggle("hidden", isProdigy);
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

async function loadSettings() {
  try {
    const settings = await api("/api/settings");
    applySettings(settings);
  } catch (err) {
    // silent — settings load is best-effort
  }
}

async function refreshImages() {
  try {
    const status = await api("/api/status");
    renderGallery(status.images);
  } catch (err) {
    // silent — image refresh is best-effort
  }
}

async function autoCalculateTraining() {
  autoCalcButton.disabled = true;
  try {
    const resp = await api("/api/settings/defaults", {
      method: "POST",
      body: JSON.stringify(collectSettings())
    });
    applySettings(resp.settings);
    setStatus(resp.message, resp.ok);
    await saveSettings();
    await refreshModelStatus();
  } catch (err) {
    setStatus(err.message, false);
  } finally {
    autoCalcButton.disabled = false;
  }
}

function queueSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSettings, 450);
}

function markMusubiCacheDirty(reason = "video dataset settings changed") {
  if (!isVideoArchitecture(els.architecture.value)) return;
  musubiCacheDirty = true;
  setStatus(`${reason}; TOML/cache will rebuild automatically on Start.`, true);
}

async function startTraining() {
  setRunning(true);
  try {
    // Explicit save before start — ensures output_path and other changes
    // are persisted even if the debounce hasn't fired yet.
    const settings = collectSettings();
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify(settings)
      });
    } catch (saveErr) {
      // Non-fatal — backend also saves on start
    }
    const resp = await api("/api/train/start", {
      method: "POST",
      body: JSON.stringify(settings)
    });
    setStatus(resp.message, resp.ok);
    if (resp.ok) musubiCacheDirty = false;
    if (resp.ok && resp.prepared_path) {
      els.dataset_path.value = resp.prepared_path;
      queueSave();
    }
    if (!resp.ok) setRunning(false);
  } catch (err) {
    setStatus(err.message, false);
    setRunning(false);
  }
}

async function runDatasetPrep(action) {
  setRunning(true);
  try {
    const resp = await api("/api/dataset/prep", {
      method: "POST",
      body: JSON.stringify({
        action,
        settings: collectSettings()
      })
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
  openOutputButton.disabled = true;
  autoCalcButton.disabled = true;
  quitButton.disabled = true;
  } catch (err) {
    setStatus(err.message, false);
  }
}

async function openOutputFolder() {
  openOutputButton.disabled = true;
  try {
    const resp = await api("/api/output/open", {
      method: "POST",
      body: JSON.stringify(collectSettings())
    });
    setStatus(resp.message, resp.ok);
  } catch (err) {
    setStatus(err.message, false);
  } finally {
    openOutputButton.disabled = false;
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
  runtimeStatus.textContent = status.ready ? "Runtime ready" : status.message || "Runtime missing";
  runtimeStatus.title = status.ready ? status.path : status.path || `Expected at ${status.expected}`;
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
  autoCalcButton.disabled = value;
  tagDatasetButton.disabled = value;
  resizeDatasetButton.disabled = value;
  prepDatasetButton.disabled = value;
  normalizeVideoButton.disabled = value;
  writeMusubiDatasetButton.disabled = value;
  cacheMusubiTextButton.disabled = value;
  cacheMusubiLatentsButton.disabled = value;
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
  galleryImages = images || [];
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
    if (image.label) {
      const badge = document.createElement("div");
      badge.className = "preview-badge";
      badge.textContent = image.label;
      figure.append(badge);
    }
    const cap = document.createElement("figcaption");
    cap.textContent = image.label ? `${image.label} - ${image.name}` : image.name;
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
    if (msg.type === "training_state") {
      setRunning(Boolean(msg.data.running));
      // Refresh settings and gallery when training stops so the UI
      // reflects the saved output_path and latest sample images.
      if (!msg.data.running) {
        loadSettings();
        refreshImages();
      }
    }
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
    // Track first GPU VRAM for resolution suggestion
    if (gpuVramMB === 0 && gpu.memTotal) {
      gpuVramMB = gpu.memTotal;
      updateResHint();
    }
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
  if (!field) continue;
  field.addEventListener("input", () => {
    queueSave();
    if (isVideoArchitecture(els.architecture.value) && (field.id.startsWith("video_") || field.id === "dataset_path" || field.id === "output_path")) {
      markMusubiCacheDirty(field.id === "dataset_path" ? "video source changed" : field.id === "output_path" ? "output path changed" : "video parameter changed");
    }
  });
  field.addEventListener("change", () => {
    queueSave();
    if (isVideoArchitecture(els.architecture.value) && (field.id.startsWith("video_") || field.id === "dataset_path" || field.id === "output_path")) {
      markMusubiCacheDirty(field.id === "dataset_path" ? "video source changed" : field.id === "output_path" ? "output path changed" : "video parameter changed");
    }
  });
}

for (const [proxyID, proxy] of Object.entries(videoNormalizerProxies)) {
  if (!proxy) continue;
  proxy.addEventListener("input", () => {
    syncVideoNormalizerProxyToSetting(proxyID);
    markMusubiCacheDirty("video normalizer parameter changed");
  });
  proxy.addEventListener("change", () => {
    syncVideoNormalizerProxyToSetting(proxyID);
    markMusubiCacheDirty("video normalizer parameter changed");
  });
}

// Resolution selector wiring
if (videoResPreset) {
  populateResPresetDropdown();
  videoResPreset.addEventListener("change", onResPresetChange);
}
if (videoResFlip) {
  videoResFlip.addEventListener("click", onResFlip);
}
// Custom W/H inputs sync back to settings
for (const id of ["video_norm_width_proxy", "video_norm_height_proxy"]) {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", onCustomResChange);
}

for (const button of architectureButtons) {
  button.addEventListener("click", () => setArchitecture(button.dataset.architecture));
}

for (const id of ["dit_path", "checkpoint_path", "qwen_path", "vae_path"]) {
  els[id].addEventListener("change", refreshModelStatus);
}

els.optimizer.addEventListener("change", () => {
  normalizeOptimizerLearningRate();
});

for (const button of document.querySelectorAll(".browse-button")) {
  button.addEventListener("click", () => openPathPicker(button.dataset.target, button.dataset.mode));
}

saveButton.addEventListener("click", saveSettings);
autoCalcButton.addEventListener("click", autoCalculateTraining);
openOutputButton.addEventListener("click", openOutputFolder);
startButton.addEventListener("click", startTraining);
tagDatasetButton.addEventListener("click", () => runDatasetPrep("tag"));
resizeDatasetButton.addEventListener("click", () => runDatasetPrep("resize"));
prepDatasetButton.addEventListener("click", () => runDatasetPrep("all"));
copyLogButton.addEventListener("click", () => {
  navigator.clipboard.writeText(logs.textContent).catch(() => {
    const ta = document.createElement("textarea");
    ta.value = logs.textContent;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  });
});
clearLogButton.addEventListener("click", () => { logs.textContent = ""; });
normalizeVideoButton.addEventListener("click", () => runDatasetPrep("normalize-video"));
writeMusubiDatasetButton.addEventListener("click", () => runDatasetPrep("musubi-dataset-toml"));
cacheMusubiTextButton.addEventListener("click", () => runDatasetPrep("musubi-cache-text"));
cacheMusubiLatentsButton.addEventListener("click", () => runDatasetPrep("musubi-cache-latents"));
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

// --- Advanced options dialog ---
const advancedDialog = document.getElementById("advancedDialog");
const advancedOptionsButton = document.getElementById("advancedOptionsButton");
const advancedClose = document.getElementById("advancedClose");
const advancedSave = document.getElementById("advancedSave");

function openAdvancedDialog() {
  if (advancedDialog) advancedDialog.classList.remove("hidden");
}

function closeAdvancedDialog() {
  if (advancedDialog) advancedDialog.classList.add("hidden");
}

if (advancedOptionsButton) advancedOptionsButton.addEventListener("click", openAdvancedDialog);
if (advancedClose) advancedClose.addEventListener("click", closeAdvancedDialog);
if (advancedSave) advancedSave.addEventListener("click", () => { queueSave(); closeAdvancedDialog(); });
if (advancedDialog) advancedDialog.addEventListener("click", (event) => {
  if (event.target === advancedDialog) closeAdvancedDialog();
});

// --- Musubi advanced options dialog ---
const musubiAdvancedDialog = document.getElementById("musubiAdvancedDialog");
const musubiAdvancedButton = document.getElementById("musubiAdvancedButton");
const musubiAdvancedClose = document.getElementById("musubiAdvancedClose");
const musubiAdvancedSave = document.getElementById("musubiAdvancedSave");

function openMusubiAdvancedDialog() {
  if (musubiAdvancedDialog) musubiAdvancedDialog.classList.remove("hidden");
}

function closeMusubiAdvancedDialog() {
  if (musubiAdvancedDialog) musubiAdvancedDialog.classList.add("hidden");
}

if (musubiAdvancedButton) musubiAdvancedButton.addEventListener("click", openMusubiAdvancedDialog);
if (musubiAdvancedClose) musubiAdvancedClose.addEventListener("click", closeMusubiAdvancedDialog);
if (musubiAdvancedSave) musubiAdvancedSave.addEventListener("click", () => { queueSave(); closeMusubiAdvancedDialog(); });
if (musubiAdvancedDialog) musubiAdvancedDialog.addEventListener("click", (event) => {
  if (event.target === musubiAdvancedDialog) closeMusubiAdvancedDialog();
});

boot().catch((err) => setStatus(err.message, false));
