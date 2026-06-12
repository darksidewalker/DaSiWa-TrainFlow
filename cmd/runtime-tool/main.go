package main

import (
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
	"sync"
	"time"

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
	mux.Handle("/ws", hub)
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, runState.status())
	})
	mux.HandleFunc("/api/run", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Action     string `json:"action"`
			KeepBackup bool   `json:"keepBackup"`
		}
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ok, msg := runState.start(body.Action, body.KeepBackup)
		writeJSON(w, map[string]any{"ok": ok, "message": msg})
	})
	mux.Handle("/", http.FileServer(mustSub(webFS)))

	server := &http.Server{
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

func (r *runner) start(action string, keepBackup bool) (bool, string) {
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
			err = runtimeops.InstallRequirements(r.root, r.append)
		case "update":
			err = runtimeops.UpdateRuntime(r.root, keepBackup, r.append)
		case "verify":
			err = runtimeops.Verify(r.root, r.append)
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
	}
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
	_ = cmd.Start()
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
