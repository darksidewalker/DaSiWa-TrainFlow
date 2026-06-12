const logs = document.getElementById("logs");
const statusText = document.getElementById("statusText");
const modelStatus = document.getElementById("modelStatus");
const osBadge = document.getElementById("osBadge");
const keepBackup = document.getElementById("keepBackup");
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
    body: JSON.stringify({ action, keepBackup: keepBackup.checked })
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
}

function renderStatus(data) {
  osBadge.textContent = data.os || "";
  const lines = data.logs || [];
  logs.textContent = lines.join("\n");
  logs.scrollTop = logs.scrollHeight;
  for (const button of Object.values(buttons)) {
    button.disabled = Boolean(data.running);
  }
  buttons.quit.disabled = false;
  keepBackup.disabled = Boolean(data.running);
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

buttons.install.addEventListener("click", () => run("install"));
buttons.update.addEventListener("click", () => run("update"));
buttons.models.addEventListener("click", () => run("models"));
buttons.prepModels.addEventListener("click", () => run("prep-models"));
buttons.verify.addEventListener("click", () => run("verify"));
buttons.quit.addEventListener("click", () => quitApp().catch((err) => (statusText.textContent = err.message)));

api("/api/status").then(renderStatus).catch((err) => {
  statusText.textContent = err.message;
});
connectWS();
