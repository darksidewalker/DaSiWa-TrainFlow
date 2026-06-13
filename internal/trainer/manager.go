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
	"strconv"
	"strings"
	"sync"
	"time"

	"trainflow/internal/modelops"
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
	if !jsonContains(data, "auto_trigger") {
		settings.AutoTrigger = true
	}
	m.settings = normalizeSettings(settings)
	return nil
}

func (m *Manager) Settings() Settings {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.settings
}

func (m *Manager) SaveSettings(s Settings) error {
	s = normalizeSettings(s)
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
	s = normalizeSettings(s)
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

	projectName := projectNameForSettings(s)
	projectOut := filepath.Join(m.root, "training", "output", projectName)
	sampleDir := filepath.Join(projectOut, "sample")
	configDir := filepath.Join(projectOut, "configs")
	for _, dir := range []string{projectOut, sampleDir, configDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}

	profile := profileFor(s)
	baseRes, maxBucket := analyzeDatasetResolution(s.DatasetPath)
	promptPath, err := createSamplePrompts(projectName, s, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	datasetTOML, err := createDatasetTOML(projectName, s, profile, baseRes, maxBucket, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	resumePath := resolveResumePath(s, projectOut)
	trainingTOML, err := createTrainingTOML(projectName, s, profile, projectOut, promptPath, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	python := pythonExecutable(m.root)
	if err := validatePythonRuntime(python); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	if profile.Architecture == ArchitectureAnima && s.FlashAttention {
		if err := validateFlashAttentionRuntime(python); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}
	trainDir := filepath.Join(m.root, "training", "sd-scripts")
	trainScript := profile.trainingScript(m.root)
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
	m.activeGPUs = map[string]string{"0": profile.Label + " training"}
	m.logLines = nil
	m.mu.Unlock()
	m.appendLog(fmt.Sprintf("Preparing %s (%s)...", projectName, profile.Label))
	m.appendLog(fmt.Sprintf("Auto-resolution: base %dpx, max bucket %dpx, bucket step %dpx", baseRes, maxBucket, profile.BucketStep))
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

func (m *Manager) StartDatasetPrep(action string, s Settings) (StartResponse, error) {
	if err := m.SaveSettings(s); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return StartResponse{OK: false, Message: "Another process is already running."}, nil
	}
	m.mu.Unlock()

	if strings.TrimSpace(s.DatasetPath) == "" || !dirExists(s.DatasetPath) {
		return StartResponse{OK: false, Message: "Dataset path not found: " + s.DatasetPath}, nil
	}
	if action != "tag" && action != "resize" && action != "all" {
		return StartResponse{OK: false, Message: "Unknown dataset prep action: " + action}, nil
	}
	if action == "tag" || action == "all" {
		if err := validatePrepModels(m.root); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}

	python := pythonExecutable(m.root)
	if err := validatePythonRuntime(python); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	trainDir := filepath.Join(m.root, "training", "sd-scripts")
	args := datasetPrepArgs(m.root, action, s)
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
	m.activeGPUs = map[string]string{"0": "dataset prep"}
	m.logLines = nil
	m.mu.Unlock()
	m.appendLog("Preparing dataset: " + datasetPrepLabel(action))
	if action == "resize" || action == "all" {
		m.appendLog("Prepared images will be written to: " + preparedDatasetPath(m.root, s))
	}

	if err := cmd.Start(); err != nil {
		m.mu.Lock()
		m.running = false
		m.trainingCmd = nil
		m.activeGPUs = map[string]string{}
		m.mu.Unlock()
		m.appendLog("Launch failed: " + err.Error())
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	go m.pipeLogs(stdout, "")
	go m.pipeLogs(stderr, "")
	go m.waitForExit(cmd, "")
	resp := StartResponse{OK: true, Message: "Dataset prep started."}
	if action == "resize" || action == "all" {
		resp.PreparedPath = filepath.ToSlash(absPath(preparedDatasetPath(m.root, s)))
	}
	return resp, nil
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
		"images":  listLatestImages(filepath.Join(outputProject(m.root, m.settings), "sample")),
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

func datasetPrepArgs(root, action string, s Settings) []string {
	tagArgs := datasetTagArgs(root, s)
	resizeArgs := datasetResizeArgs(root, s)
	switch action {
	case "tag":
		return tagArgs
	case "resize":
		return resizeArgs
	default:
		payload, _ := json.Marshal([][]string{tagArgs, resizeArgs})
		script := `
import json
import subprocess
import sys

for command in json.loads(sys.argv[1]):
    print("$ " + " ".join(command), flush=True)
    subprocess.check_call(command)
`
		return []string{"-c", script, string(payload)}
	}
}

func datasetTagArgs(root string, s Settings) []string {
	args := []string{
		filepath.Join(root, "training", "sd-scripts", "finetune", "tag_images_by_wd14_tagger.py"),
		filepath.ToSlash(absPath(s.DatasetPath)),
		"--repo_id", "wd-eva02-large-tagger-v3",
		"--model_dir", filepath.ToSlash(absPath(filepath.Join(root, "models"))),
		"--onnx",
		"--caption_extension", ".txt",
		"--general_threshold", fmt.Sprintf("%.4f", s.TaggerGenThreshold),
		"--character_threshold", fmt.Sprintf("%.4f", s.TaggerCharThreshold),
		"--batch_size", "1",
		"--max_data_loader_n_workers", "2",
	}
	if !s.TaggerOverwrite {
		args = append(args, "--append_tags")
	}
	return args
}

func datasetResizeArgs(root string, s Settings) []string {
	profile := profileFor(normalizeSettings(s))
	maxSide := s.SideMax
	if maxSide <= 0 {
		maxSide = 768
	}
	resolution := fmt.Sprintf("%dx%d", maxSide, maxSide)
	return []string{
		filepath.Join(root, "training", "sd-scripts", "tools", "resize_images_to_resolution.py"),
		filepath.ToSlash(absPath(s.DatasetPath)),
		filepath.ToSlash(absPath(preparedDatasetPath(root, s))),
		"--max_resolution", resolution,
		"--divisible_by", strconv.Itoa(profile.ResizeDivisor),
		"--copy_associated_files",
	}
}

func preparedDatasetPath(root string, s Settings) string {
	name := projectNameForSettings(s)
	if name == "untitled" {
		name = sanitizeProjectName(filepath.Base(strings.TrimRight(s.DatasetPath, string(os.PathSeparator))))
	}
	return filepath.Join(root, "training", "prepared", name)
}

func datasetPrepLabel(action string) string {
	switch action {
	case "tag":
		return "caption tagging"
	case "resize":
		return "resize/copy"
	default:
		return "caption tagging and resize/copy"
	}
}

func validatePrepModels(root string) error {
	required := map[string]bool{
		filepath.Join(root, "models", "wd-eva02-large-tagger-v3", "model.onnx"):        false,
		filepath.Join(root, "models", "wd-eva02-large-tagger-v3", "selected_tags.csv"): false,
	}
	for _, file := range modelops.OptionalFiles(root) {
		if _, ok := required[file.Path]; ok && fileExists(file.Path) {
			required[file.Path] = true
		}
	}
	var missing []string
	for path, ok := range required {
		if !ok {
			missing = append(missing, path)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("prep model files missing; click Models Tool, then Download Prep:\n%s", strings.Join(missing, "\n"))
	}
	return nil
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
