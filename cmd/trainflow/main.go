package main

import (
	"context"
	"embed"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"trainflow/internal/hwmon"
	"trainflow/internal/trainer"
)

//go:embed web/*
var webFS embed.FS

func main() {
	root, err := detectRoot()
	if err != nil {
		log.Fatalf("detect root: %v", err)
	}

	hub := trainer.NewHub()
	manager := trainer.NewManager(root, hub)
	monitor := hwmon.New()

	mux := http.NewServeMux()
	var server *http.Server
	var shutdownOnce sync.Once
	onQuit := func() {
		shutdownOnce.Do(func() {
			go func() {
				wasRunning := false
				if running, ok := manager.Status()["running"].(bool); ok {
					wasRunning = running
				}
				_, _ = manager.Stop()
				if wasRunning {
					time.Sleep(4 * time.Second)
				}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				_ = server.Shutdown(ctx)
			}()
		})
	}
	trainer.RegisterRoutes(mux, webFS, manager, hub, onQuit)

	server = &http.Server{
		Addr:              ":7860",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for range ticker.C {
			snapshot := monitor.Snapshot(manager.ActiveGPUActivities())
			hub.BroadcastJSON("hw_stats", snapshot)
		}
	}()

	url := "http://127.0.0.1" + server.Addr
	log.Printf("DaSiWa TrainFlow is running at %s", url)
	go openBrowser(url)

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
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
	return "", fmt.Errorf("run from the TrainFlow folder or place the binary beside training/sd-scripts")
}

func looksLikeRoot(dir string) bool {
	if dir == "" {
		return false
	}
	_, err := os.Stat(filepath.Join(dir, "training", "sd-scripts"))
	return err == nil
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
