package trainer

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"trainflow/internal/modelops"
)

func RegisterRoutes(mux *http.ServeMux, embedded fs.FS, manager *Manager, hub *Hub, onQuit func()) {
	web, err := fs.Sub(embedded, "web")
	if err != nil {
		panic(err)
	}

	mux.HandleFunc("/api/settings", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			writeJSON(w, manager.Settings())
		case http.MethodPost:
			var settings Settings
			if err := decodeJSON(r, &settings); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if err := manager.SaveSettings(settings); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			writeJSON(w, StartResponse{OK: true, Message: "Settings saved."})
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, manager.Status())
	})

	mux.HandleFunc("/api/runtime", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, runtimeStatus(manager.root))
	})

	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, modelops.Check(manager.root))
	})

	mux.HandleFunc("/api/runtime/launch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		status := runtimeStatus(manager.root)
		if status.Ready {
			writeJSON(w, StartResponse{OK: true, Message: "Runtime is already ready."})
			return
		}
		if err := launchRuntimeTool(manager.root); err != nil {
			writeJSON(w, StartResponse{OK: false, Message: err.Error()})
			return
		}
		writeJSON(w, StartResponse{OK: true, Message: "Runtime tool launched."})
	})

	mux.HandleFunc("/api/app/quit", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, StartResponse{OK: true, Message: "Shutting down TrainFlow."})
		if onQuit != nil {
			onQuit()
		}
	})

	mux.HandleFunc("/api/train/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var settings Settings
		if err := decodeJSON(r, &settings); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp, err := manager.Start(settings)
		if err != nil && resp.Message == "" {
			resp = StartResponse{OK: false, Message: err.Error()}
		}
		writeJSON(w, resp)
	})

	mux.HandleFunc("/api/train/stop", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		resp, err := manager.Stop()
		if err != nil && resp.Message == "" {
			resp = StartResponse{OK: false, Message: err.Error()}
		}
		writeJSON(w, resp)
	})

	mux.HandleFunc("/api/images", func(w http.ResponseWriter, r *http.Request) {
		settings := manager.Settings()
		writeJSON(w, listLatestImages(filepath.Join(outputProject(manager.root, settings.TriggerWord), "sample")))
	})

	mux.HandleFunc("/api/path/list", func(w http.ResponseWriter, r *http.Request) {
		mode := r.URL.Query().Get("mode")
		switch mode {
		case "directory", "file", "model":
		default:
			mode = "directory"
		}
		writeJSON(w, listPath(r.URL.Query().Get("path"), mode))
	})

	mux.HandleFunc("/samples/", func(w http.ResponseWriter, r *http.Request) {
		rel := strings.TrimPrefix(r.URL.Path, "/samples/")
		parts := strings.SplitN(rel, "/", 2)
		if len(parts) != 2 || strings.Contains(parts[0], "..") || strings.Contains(parts[1], "..") {
			http.NotFound(w, r)
			return
		}
		path := filepath.Join(manager.root, "training", "output", parts[0], "sample", parts[1])
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, path)
	})

	mux.Handle("/ws", hub)
	mux.Handle("/", http.FileServer(http.FS(web)))
}

type RuntimeStatus struct {
	Ready    bool   `json:"ready"`
	OS       string `json:"os"`
	Path     string `json:"path"`
	Expected string `json:"expected"`
	Message  string `json:"message"`
}

func runtimeStatus(root string) RuntimeStatus {
	expected := filepath.Join(root, "python_embeded", "linux")
	candidates := []string{
		filepath.Join(expected, "bin", "python3"),
		filepath.Join(expected, "bin", "python"),
	}
	if runtime.GOOS == "windows" {
		expected = filepath.Join(root, "python_embeded", "windows")
		candidates = []string{filepath.Join(expected, "python.exe")}
	}
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return RuntimeStatus{
				Ready:    true,
				OS:       runtime.GOOS,
				Path:     candidate,
				Expected: expected,
				Message:  "Runtime ready",
			}
		}
	}
	return RuntimeStatus{
		Ready:    false,
		OS:       runtime.GOOS,
		Expected: expected,
		Message:  "Runtime missing",
	}
}

func launchRuntimeTool(root string) error {
	name := "TrainFlow_Runtime_Tool"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(root, name)
	if !fileExists(path) {
		return fmt.Errorf("%s was not found beside TrainFlow", name)
	}
	cmd := exec.Command(path)
	cmd.Dir = root
	return cmd.Start()
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
