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

	mux.HandleFunc("/api/settings/defaults", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var settings Settings
		if err := decodeJSON(r, &settings); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		next, message := applyStableDefaults(settings)
		writeJSON(w, AutoCalcResponse{OK: true, Message: message, Settings: next})
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
		writeJSON(w, modelStatusForSettings(manager.root, manager.Settings()))
	})

	mux.HandleFunc("/api/runtime/launch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		status := runtimeStatus(manager.root)
		if status.Ready {
			if err := openRuntimeTool(manager.root); err != nil {
				writeJSON(w, StartResponse{OK: false, Message: err.Error()})
				return
			}
			writeJSON(w, StartResponse{OK: true, Message: "Runtime tool opened."})
			return
		}
		if err := openRuntimeTool(manager.root); err != nil {
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

	mux.HandleFunc("/api/dataset/prep", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Action   string   `json:"action"`
			Settings Settings `json:"settings"`
		}
		if err := decodeJSON(r, &body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp, err := manager.StartDatasetPrep(body.Action, body.Settings)
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

	mux.HandleFunc("/api/output/open", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var settings Settings
		if err := decodeJSON(r, &settings); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		settings = normalizeSettings(settings)
		if projectNameForSettings(settings) == "untitled" {
			writeJSON(w, StartResponse{OK: false, Message: "Set a project name first."})
			return
		}
		if err := manager.SaveSettings(settings); err != nil {
			writeJSON(w, StartResponse{OK: false, Message: err.Error()})
			return
		}
		path := outputProject(manager.root, settings)
		if err := os.MkdirAll(path, 0755); err != nil {
			writeJSON(w, StartResponse{OK: false, Message: err.Error()})
			return
		}
		if err := openFolder(path); err != nil {
			writeJSON(w, StartResponse{OK: false, Message: err.Error()})
			return
		}
		writeJSON(w, StartResponse{OK: true, Message: "Opened output folder."})
	})

	mux.HandleFunc("/api/images", func(w http.ResponseWriter, r *http.Request) {
		settings := manager.Settings()
		writeJSON(w, listLatestImages(filepath.Join(outputProject(manager.root, settings), "sample")))
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

func modelOverrides(settings Settings) map[string]string {
	return map[string]string{
		"dit_path":  settings.DiTPath,
		"qwen_path": settings.QwenPath,
		"vae_path":  settings.VAEPath,
	}
}

func modelStatusForSettings(root string, settings Settings) modelops.Status {
	settings = normalizeSettings(settings)
	profile := profileFor(settings)
	if profile.supportsManagedModelCheck() {
		return modelops.CheckWithOverrides(root, modelOverrides(settings))
	}

	files := []modelops.ModelFile{
		{
			Name:  "SDXL / Pony / Illustrious checkpoint",
			Key:   "checkpoint_path",
			Path:  settings.CheckpointPath,
			Found: settings.CheckpointPath,
			OK:    fileExists(settings.CheckpointPath),
		},
	}
	files = append(files, modelops.OptionalFiles(root)...)
	missing := 0
	optionalMissing := 0
	for i := range files {
		if files[i].OK {
			continue
		}
		if files[i].Optional {
			if fileExists(files[i].Path) {
				files[i].OK = true
				files[i].Found = files[i].Path
				continue
			}
			optionalMissing++
		} else {
			missing++
		}
	}
	message := "Models ready"
	if missing > 0 {
		message = "Choose an SDXL, Pony, or Illustrious checkpoint."
	} else if optionalMissing > 0 {
		message = fmt.Sprintf("Optional prep models missing: %d", optionalMissing)
	}
	return modelops.Status{
		Ready:           missing == 0,
		OptionalReady:   optionalMissing == 0,
		Missing:         missing,
		OptionalMissing: optionalMissing,
		Files:           files,
		Message:         message,
	}
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
			if err := validatePythonRuntime(candidate); err != nil {
				return RuntimeStatus{
					Ready:    false,
					OS:       runtime.GOOS,
					Path:     candidate,
					Expected: expected,
					Message:  err.Error(),
				}
			}
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

func openRuntimeTool(root string) error {
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
	if err := cmd.Start(); err != nil {
		return err
	}
	return openURL("http://127.0.0.1:7870")
}

func openFolder(path string) error {
	return openURL(path)
}

func openURL(path string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
			cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", path)
		} else {
			cmd = exec.Command("explorer", path)
		}
	case "darwin":
		cmd = exec.Command("open", path)
	default:
		cmd = exec.Command("xdg-open", path)
	}
	return cmd.Start()
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
