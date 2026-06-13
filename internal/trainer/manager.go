package trainer

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"trainflow/internal/process"
)

const maxLogLines = 500

type Manager struct {
	root         string
	hub          *Hub
	mu           sync.Mutex
	settings     Settings
	trainingCmd  *exec.Cmd
	running      bool
	activeGPUs   map[string]string
	logLines     []string
	settingsPath string
}

func NewManager(root string, hub *Hub) *Manager {
	_ = os.MkdirAll(filepath.Join(root, "training", "output"), 0755)
	m := &Manager{
		root:         root,
		hub:          hub,
		settings:     DefaultSettings(root),
		activeGPUs:   make(map[string]string),
		settingsPath: filepath.Join(root, "training", "settings.json"),
	}
	_ = m.LoadSettings()
	return m
}

func (m *Manager) LoadSettings() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	data, err := os.ReadFile(m.settingsPath)
	if err != nil {
		return err
	}
	settings := DefaultSettings(m.root)
	if err := json.Unmarshal(data, &settings); err != nil {
		return err
	}
	if !jsonContains(data, "train_unet_only") {
		settings.TrainUNetOnly = true
	}
	m.settings = settings
	return nil
}

func (m *Manager) Settings() Settings {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.settings
}

func (m *Manager) SaveSettings(s Settings) error {
	m.mu.Lock()
	m.settings = s
	m.mu.Unlock()
	_ = os.MkdirAll(filepath.Dir(m.settingsPath), 0755)
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(m.settingsPath, data, 0644)
}

func (m *Manager) Start(s Settings) (StartResponse, error) {
	if err := m.SaveSettings(s); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return StartResponse{OK: false, Message: "Training is already running."}, nil
	}
	m.mu.Unlock()

	if errs := validateSettings(s); len(errs) > 0 {
		msg := strings.Join(errs, "\n")
		m.appendLog("Pre-flight failed:\n" + msg)
		return StartResponse{OK: false, Message: msg}, nil
	}

	projectName := sanitizeProjectName(s.TriggerWord)
	projectOut := filepath.Join(m.root, "training", "output", projectName)
	sampleDir := filepath.Join(projectOut, "sample")
	configDir := filepath.Join(projectOut, "configs")
	for _, dir := range []string{projectOut, sampleDir, configDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}

	baseRes, maxBucket := analyzeDatasetResolution(s.DatasetPath)
	promptPath, err := createSamplePrompts(projectName, s, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	datasetTOML, err := createDatasetTOML(projectName, s, baseRes, maxBucket, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	resumePath := resolveResumePath(s, projectOut)
	trainingTOML, err := createTrainingTOML(projectName, s, projectOut, promptPath, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	python := pythonExecutable(m.root)
	if err := validatePythonRuntime(python); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	if s.FlashAttention {
		if err := validateFlashAttentionRuntime(python); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}
	trainDir := filepath.Join(m.root, "training", "sd-scripts")
	trainScript := filepath.Join(trainDir, "anima_train_network.py")
	args := []string{
		"-m", "accelerate.commands.launch",
		"--num_processes=1",
		"--num_machines=1",
		"--mixed_precision=bf16",
		"--dynamo_backend=no",
		trainScript,
		"--config_file", trainingTOML,
		"--dataset_config", datasetTOML,
	}
	cmd := exec.Command(python, args...)
	cmd.Dir = trainDir
	cmd.Env = trainingEnv(trainDir)
	process.Prepare(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	m.mu.Lock()
	m.trainingCmd = cmd
	m.running = true
	m.activeGPUs = map[string]string{"0": "training"}
	m.logLines = nil
	m.mu.Unlock()
	m.appendLog(fmt.Sprintf("Preparing %s...", projectName))
	m.appendLog(fmt.Sprintf("Auto-resolution: base %dpx, max bucket %dpx", baseRes, maxBucket))
	if s.ResumeEnabled {
		if resumePath == "" {
			m.appendLog("Resume enabled, but no saved state was found. Starting fresh.")
		} else {
			m.appendLog("Resume state: " + resumePath)
		}
	}
	m.appendLog("Launching training process...")

	if err := cmd.Start(); err != nil {
		m.mu.Lock()
		m.running = false
		m.trainingCmd = nil
		m.activeGPUs = map[string]string{}
		m.mu.Unlock()
		m.appendLog("Launch failed: " + err.Error())
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	go m.pipeLogs(stdout, sampleDir)
	go m.pipeLogs(stderr, sampleDir)
	go m.waitForExit(cmd, sampleDir)
	return StartResponse{OK: true, Message: "Training started."}, nil
}

func (m *Manager) Stop() (StartResponse, error) {
	m.mu.Lock()
	cmd := m.trainingCmd
	running := m.running
	m.mu.Unlock()
	if !running || cmd == nil {
		return StartResponse{OK: true, Message: "Not running."}, nil
	}
	m.appendLog("Stopping training...")
	if err := process.Terminate(cmd); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	go func() {
		time.Sleep(3 * time.Second)
		m.mu.Lock()
		stillRunning := m.running && m.trainingCmd == cmd
		m.mu.Unlock()
		if stillRunning {
			_ = process.Kill(cmd)
		}
	}()
	return StartResponse{OK: true, Message: "Stopping training."}, nil
}

func (m *Manager) ActiveGPUActivities() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]string, len(m.activeGPUs))
	for k, v := range m.activeGPUs {
		out[k] = v
	}
	return out
}

func (m *Manager) Status() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	return map[string]any{
		"running": m.running,
		"logs":    strings.Join(m.logLines, "\n"),
		"images":  listLatestImages(filepath.Join(outputProject(m.root, m.settings.TriggerWord), "sample")),
	}
}

func (m *Manager) appendLog(line string) {
	m.appendLogLine(line, false)
}

func (m *Manager) appendTrainingLog(line string) {
	m.appendLogLine(line, isProgressLog(line))
}

func (m *Manager) appendLogLine(line string, replaceProgress bool) {
	m.mu.Lock()
	last := len(m.logLines) - 1
	if replaceProgress && last >= 0 && isProgressLog(m.logLines[last]) {
		m.logLines[last] = line
	} else {
		m.logLines = append(m.logLines, line)
	}
	if len(m.logLines) > maxLogLines {
		m.logLines = m.logLines[len(m.logLines)-maxLogLines:]
	}
	logs := strings.Join(m.logLines, "\n")
	running := m.running
	m.mu.Unlock()
	m.hub.BroadcastJSON("log", map[string]any{"logs": logs, "running": running})
}

func (m *Manager) pipeLogs(reader io.Reader, sampleDir string) {
	scanner := bufio.NewScanner(reader)
	scanner.Split(scanLogChunk)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.ReplaceAll(scanner.Text(), "\r", ""))
		if line == "" || isBlacklistedLog(line) {
			continue
		}
		m.appendTrainingLog(line)
		lower := strings.ToLower(line)
		if strings.Contains(lower, "saved") || strings.Contains(lower, "sample") {
			m.hub.BroadcastJSON("images", listLatestImages(sampleDir))
		}
	}
	if err := scanner.Err(); err != nil {
		m.appendLog("Log stream closed: " + err.Error())
	}
}

func (m *Manager) waitForExit(cmd *exec.Cmd, sampleDir string) {
	err := cmd.Wait()
	m.mu.Lock()
	if m.trainingCmd == cmd {
		m.trainingCmd = nil
		m.running = false
		m.activeGPUs = map[string]string{}
	}
	m.mu.Unlock()
	if err != nil {
		m.appendLog("Process exited: " + err.Error())
	} else {
		m.appendLog("Process finished.")
	}
	m.hub.BroadcastJSON("images", listLatestImages(sampleDir))
	m.hub.BroadcastJSON("training_state", map[string]bool{"running": false})
}

func trainingEnv(trainDir string) []string {
	env := os.Environ()
	env = append(env, "PYTHONPATH="+trainDir+string(os.PathListSeparator)+os.Getenv("PYTHONPATH"))
	env = append(env, "PYTHONIOENCODING=utf-8")
	env = append(env, "PYTHONWARNINGS=ignore")
	env = append(env, "TORCH_CPP_LOG_LEVEL=ERROR")
	env = append(env, "KMP_WARNINGS=0")
	env = append(env, "CUDA_VISIBLE_DEVICES=0")
	env = append(env, "ACCELERATE_USE_CPU=False")
	return env
}

func scanLogChunk(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for i, b := range data {
		if b == '\n' || b == '\r' {
			advance = i + 1
			if b == '\r' && len(data) > i+1 && data[i+1] == '\n' {
				advance++
			}
			return advance, data[:i], nil
		}
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

func isProgressLog(line string) bool {
	lower := strings.ToLower(line)
	return strings.Contains(line, "%|") && (strings.Contains(lower, "it/s") || strings.Contains(lower, "/it") || strings.Contains(lower, "steps:"))
}

func isBlacklistedLog(line string) bool {
	blacklist := []string{
		"triton not found",
		"flop counting will not work",
		"torch\\utils\\flop_counter.py",
	}
	lower := strings.ToLower(line)
	for _, skip := range blacklist {
		if strings.Contains(lower, skip) {
			return true
		}
	}
	return false
}

func decodeJSON(r *http.Request, target any) error {
	defer r.Body.Close()
	if r.Body == nil {
		return errors.New("missing request body")
	}
	return json.NewDecoder(r.Body).Decode(target)
}
