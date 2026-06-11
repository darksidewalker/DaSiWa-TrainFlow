package trainer

import (
	"encoding/json"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
