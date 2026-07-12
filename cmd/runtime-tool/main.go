package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"trainflow/internal/modelops"
	"trainflow/internal/process"
	"trainflow/internal/runtimeops"
	"trainflow/internal/trainer"
)

//go:embed web/*
var webFS embed.FS

type runner struct {
	root    string
	hub     *trainer.Hub
	mu      sync.Mutex
	running bool
	logs    []string
}

func main() {
	root, err := detectRoot()
	if err != nil {
		log.Fatal(err)
	}
	hub := trainer.NewHub()
	runState := &runner{root: root, hub: hub}

	mux := http.NewServeMux()
	var server *http.Server
	mux.Handle("/ws", hub)
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, runState.status())
	})
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, map[string]any{
			"status":  modelops.CheckWithOverrides(runState.root, modelOverrides(runState.root)),
			"catalog": modelops.Catalog(runState.root),
		})
	})
	mux.HandleFunc("/api/run", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Action                string   `json:"action"`
			KeepBackup            bool     `json:"keepBackup"`
			InstallFlashAttention bool     `json:"installFlashAttention"`
			InstallTorchCompile   bool     `json:"installTorchCompile"`
			TorchBackend          string   `json:"torchBackend"`
			ModelKeys             []string `json:"modelKeys"`
		}
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ok, msg := runState.start(body.Action, body.KeepBackup, body.ModelKeys, runtimeops.TorchInstallOptions{
			Backend:                 body.TorchBackend,
			InstallFlashAttention:   body.InstallFlashAttention,
			InstallTorchCompileDeps: body.InstallTorchCompile,
		})
		writeJSON(w, map[string]any{"ok": ok, "message": msg})
	})
	mux.HandleFunc("/api/app/quit", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, map[string]any{"ok": true, "message": "Shutting down runtime tool."})
		go func() {
			time.Sleep(150 * time.Millisecond)
			_ = server.Close()
		}()
	})
	mux.Handle("/", http.FileServer(mustSub(webFS)))

	server = &http.Server{
		Addr:              ":7870",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	url := "http://127.0.0.1" + server.Addr
	log.Printf("TrainFlow Runtime Tool is running at %s", url)
	go openBrowser(url)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func (r *runner) start(action string, keepBackup bool, modelKeys []string, torchOpts runtimeops.TorchInstallOptions) (bool, string) {
	r.mu.Lock()
	if r.running {
		r.mu.Unlock()
		return false, "Another runtime task is already running."
	}
	r.running = true
	r.logs = nil
	r.mu.Unlock()

	go func() {
		defer func() {
			r.mu.Lock()
			r.running = false
			r.mu.Unlock()
			r.broadcast()
		}()
		var err error
		switch action {
		case "install":
			err = runtimeops.InstallRequirementsWithOptions(r.root, torchOpts, r.append)
		case "update":
			err = runtimeops.UpdateRuntimeWithOptions(r.root, keepBackup, torchOpts, r.append)
		case "verify":
			err = runtimeops.Verify(r.root, r.append)
		case "models":
			if len(modelKeys) > 0 {
				err = modelops.DownloadSelected(r.root, modelKeys, r.append)
			} else {
				err = modelops.DownloadRequired(r.root, r.append)
			}
		case "prep-models":
			err = modelops.DownloadOptional(r.root, r.append)
		default:
			err = fmt.Errorf("unknown action: %s", action)
		}
		if err != nil {
			r.append("[ERROR] " + err.Error())
			return
		}
		r.append("Done.")
	}()
	return true, "Started."
}

func (r *runner) append(line string) {
	r.mu.Lock()
	r.logs = append(r.logs, line)
	if len(r.logs) > 1200 {
		r.logs = r.logs[len(r.logs)-1200:]
	}
	r.mu.Unlock()
	r.broadcast()
}

func (r *runner) status() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	return map[string]any{
		"running": r.running,
		"logs":    r.logs,
		"os":      runtime.GOOS,
		"models":  modelops.CheckWithOverrides(r.root, modelOverrides(r.root)),
		"catalog": modelops.Catalog(r.root),
		"gpu":     detectGPU(),
	}
}

// detectGPU returns a short description of the GPU(s) found on the host.
func detectGPU() string {
	// Try nvidia-smi first (NVIDIA)
	if name := nvidiaGPU(); name != "" {
		return name
	}
	// Try lspci (Linux AMD/intel)
	if runtime.GOOS == "linux" {
		if name := lspciGPU(); name != "" {
			return name
		}
	}
	// Try PowerShell Get-WmiObject (Windows AMD/NVIDIA)
	if runtime.GOOS == "windows" {
		if name := windowsGPU(); name != "" {
			return name
		}
	}
	return ""
}

func nvidiaGPU() string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return ""
	}
	first := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	if first == "" {
		return ""
	}
	// nvidia-smi already prefixes "NVIDIA" — don't double it
	if !strings.HasPrefix(first, "NVIDIA") {
		first = "NVIDIA " + first
	}
	// Also grab VRAM if available
	ctx2, cancel2 := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel2()
	vramOut, err2 := exec.CommandContext(ctx2, "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output()
	if err2 != nil {
		return first
	}
	vram := strings.TrimSpace(strings.Split(string(vramOut), "\n")[0])
	vramInt, _ := strconv.Atoi(vram)
	if vramInt > 0 {
		return fmt.Sprintf("%s (%d MB VRAM)", first, vramInt)
	}
	return first
}

func lspciGPU() string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "lspci", "-nn", "-d", "1002:").Output() // AMD vendor ID
	if err != nil || len(out) == 0 {
		// Try NVIDIA via lspci as fallback
		ctx2, cancel2 := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel2()
		out2, err2 := exec.CommandContext(ctx2, "lspci", "-nn", "-d", "10de:").Output()
		if err2 != nil || len(out2) == 0 {
			return ""
		}
		return parseLspciLine(out2)
	}
	return parseLspciLine(out)
}

func parseLspciLine(out []byte) string {
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := scanner.Text()
		// Extract device name between ] and (
		// e.g. "01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Device [1002:7441] (rev c1)"
		afterColon := strings.Index(line, ": ")
		if afterColon < 0 {
			continue
		}
		dev := line[afterColon+2:]
		// Strip vendor prefix "Advanced Micro Devices, Inc. [AMD/ATI] " or "NVIDIA Corporation "
		// Keep the last meaningful part
		return dev
	}
	return ""
}

func windowsGPU() string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "powershell.exe", "-Command",
		"Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name").Output()
	if err != nil || len(out) == 0 {
		return ""
	}
	// Grab first GPU line
	first := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	// Strip trailing empty lines
	for strings.TrimSpace(first) == "" {
		rest := strings.SplitN(string(out), "\n", 2)
		if len(rest) < 2 {
			return ""
		}
		first = strings.TrimSpace(rest[1])
		break
	}
	if first == "" {
		return ""
	}
	return first
}

func modelOverrides(root string) map[string]string {
	data, err := os.ReadFile(filepath.Join(root, "training", "settings.json"))
	if err != nil {
		return nil
	}
	var settings trainer.Settings
	if err := json.Unmarshal(data, &settings); err != nil {
		return nil
	}
	return trainer.ModelOverrides(settings)
}

func (r *runner) broadcast() {
	r.hub.BroadcastJSON("status", r.status())
}

func detectRoot() (string, error) {
	cwd, _ := os.Getwd()
	if looksLikeRoot(cwd) {
		return cwd, nil
	}
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		if looksLikeRoot(dir) {
			return dir, nil
		}
	}
	return "", fmt.Errorf("run from the TrainFlow folder or place the binary beside TrainFlow or training/sd-scripts")
}

func looksLikeRoot(dir string) bool {
	if dir == "" {
		return false
	}
	for _, marker := range []string{
		filepath.Join(dir, "training", "sd-scripts"),
		filepath.Join(dir, "TrainFlow"),
		filepath.Join(dir, "TrainFlow.exe"),
	} {
		if _, err := os.Stat(marker); err == nil {
			return true
		}
	}
	return false
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = process.StartDetached(cmd)
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func mustSub(embedded embed.FS) http.FileSystem {
	sub, err := fs.Sub(embedded, "web")
	if err != nil {
		panic(err)
	}
	return http.FS(sub)
}
