const logs = document.getElementById("logs");
const statusText = document.getElementById("statusText");
const osBadge = document.getElementById("osBadge");
const keepBackup = document.getElementById("keepBackup");
const buttons = {
  install: document.getElementById("installButton"),
  update: document.getElementById("updateButton"),
  verify: document.getElementById("verifyButton")
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

function renderStatus(data) {
  osBadge.textContent = data.os || "";
  const lines = data.logs || [];
  logs.textContent = lines.join("\n");
  logs.scrollTop = logs.scrollHeight;
  for (const button of Object.values(buttons)) {
    button.disabled = Boolean(data.running);
  }
  keepBackup.disabled = Boolean(data.running);
  statusText.textContent = data.running ? "Running" : "Ready";
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
buttons.verify.addEventListener("click", () => run("verify"));

api("/api/status").then(renderStatus).catch((err) => {
  statusText.textContent = err.message;
});
connectWS();
