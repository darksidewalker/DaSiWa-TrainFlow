package trainer

import (
	"bufio"
	"context"
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
	root              string
	hub               *Hub
	mu                sync.Mutex
	settings          Settings
	trainingCmd       *exec.Cmd
	running           bool
	activeGPUs        map[string]string
	logLines          []string
	settingsPath      string
	sampleDirRegistry map[string]string // URL token -> actual sample directory
}

func NewManager(root string, hub *Hub) *Manager {
	_ = os.MkdirAll(filepath.Join(root, "training", "output"), 0755)
	m := &Manager{
		root:              root,
		hub:               hub,
		settings:          DefaultSettings(root),
		activeGPUs:        make(map[string]string),
		settingsPath:      filepath.Join(root, "training", "settings.json"),
		sampleDirRegistry: make(map[string]string),
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
	if err := os.WriteFile(m.settingsPath, data, 0644); err != nil {
		return err
	}
	// Auto-generate Musubi dataset TOML when saving Musubi-family settings
	_ = m.writeMusubiDatasetTOMLIfReady(s)
	return nil
}

// writeMusubiDatasetTOMLIfReady auto-generates the Musubi dataset TOML when
// settings indicate a Musubi-family architecture and the output directory exists.
func (m *Manager) writeMusubiDatasetTOMLIfReady(s Settings) error {
	profile := profileFor(s)
	if profile.Family != trainingFamilyMusubi {
		return nil
	}
	projectName := projectNameForSettings(s)
	projectOut := outputProject(m.root, s)
	configDir := filepath.Join(projectOut, "configs")
	// Ensure output and config directories exist
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return err
	}
	_, err := createMusubiDatasetTOML(projectName, s, profile, configDir)
	return err
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
	projectOut := outputProject(m.root, s)
	sampleDir := filepath.Join(projectOut, "sample")
	configDir := filepath.Join(projectOut, "configs")
	for _, dir := range []string{projectOut, sampleDir, configDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}
	sampleToken := m.registerSampleDir(sampleDir)

	profile := profileFor(s)
	if profile.Family == trainingFamilyMusubi {
		return m.startMusubiSequenced(s, profile, projectName, projectOut, configDir, sampleDir)
	}
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

	python := process.PythonExecutable(m.root)
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
	bootstrapScript, err := createTrainingBootstrap(trainDir, trainScript, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	args := []string{
		"-m", "accelerate.commands.launch",
		"--num_processes=1",
		"--num_machines=1",
		"--mixed_precision=bf16",
		"--dynamo_backend=no",
		bootstrapScript,
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

	go m.pipeLogs(stdout, sampleToken)
	go m.pipeLogs(stderr, sampleToken)
	go m.waitForExit(cmd, sampleToken)
	return StartResponse{OK: true, Message: "Training started."}, nil
}

// runPipelineStep executes a single pipeline step sequentially with progress logging.
// It blocks until the step completes (or fails), then returns.
// The context is used for cancellation (e.g., user stop).
func (m *Manager) runPipelineStep(ctx context.Context, name, activity string, buildFn func() (musubiCommand, error)) (StartResponse, error) {
	m.appendLog(fmt.Sprintf("[%s] Starting...", name))
	m.setPipelineStep(name)

	cmdSpec, err := buildFn()
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	cmd := exec.CommandContext(ctx, cmdSpec.Program, cmdSpec.Args...)
	cmd.Dir = cmdSpec.Dir
	cmd.Env = cmdSpec.Env
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
	m.activeGPUs = map[string]string{"0": activity}
	m.logLines = nil
	m.mu.Unlock()

	m.appendLog(fmt.Sprintf("Launching: %s %s", cmdSpec.Program, strings.Join(cmdSpec.Args, " ")))

	if err := cmd.Start(); err != nil {
		m.mu.Lock()
		m.running = false
		m.trainingCmd = nil
		m.activeGPUs = map[string]string{}
		m.mu.Unlock()
		m.appendLog("Launch failed: " + err.Error())
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	errCh := make(chan error, 1)
	go func() {
		if err := cmd.Wait(); err != nil {
			if ctx.Err() == context.Canceled {
				errCh <- nil // cancellation is not an error
			} else {
				errCh <- fmt.Errorf("%s failed: %w", name, err)
			}
		} else {
			errCh <- nil
		}
	}()

	go m.pipeLogs(stdout, "")
	go m.pipeLogs(stderr, "")

	err = <-errCh
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	m.mu.Lock()
	m.running = false
	m.trainingCmd = nil
	m.activeGPUs = map[string]string{}
	m.mu.Unlock()

	m.appendLog(fmt.Sprintf("[%s] Completed successfully.", name))
	return StartResponse{OK: true, Message: fmt.Sprintf("%s completed.", name)}, nil
}

func (m *Manager) setPipelineStep(step string) {
	m.appendLog(fmt.Sprintf("Pipeline step: %s", step))
}

func (m *Manager) startCommandAsync(spec musubiCommand, activity, step, message string) (StartResponse, error) {
	cmd := exec.Command(spec.Program, spec.Args...)
	cmd.Dir = spec.Dir
	cmd.Env = spec.Env
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
	m.activeGPUs = map[string]string{"0": activity}
	m.logLines = nil
	m.mu.Unlock()
	m.setPipelineStep(step)
	m.appendLog(fmt.Sprintf("Launching: %s %s", spec.Program, strings.Join(spec.Args, " ")))

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
	return StartResponse{OK: true, Message: message, Step: step}, nil
}

// startMusubiSequenced runs the full Musubi pipeline in sequence:
// 1. Dataset TOML generation
// 2. Text encoder cache
// 3. Latent cache
// 4. Training
func (m *Manager) startMusubiSequenced(s Settings, profile trainingProfile, projectName, projectOut, configDir, sampleDir string) (StartResponse, error) {
	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return StartResponse{OK: false, Message: "Training is already running."}, nil
	}
	m.mu.Unlock()

	m.mu.Lock()
	m.logLines = nil
	m.mu.Unlock()

	sampleToken := m.registerSampleDir(sampleDir)

	m.appendLog(fmt.Sprintf("Starting %s (%s) pipeline...", projectName, profile.Label))

	// Use a cancellable context for the entire pipeline
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Video normalization is now manual-only (triggered via StartDatasetPrep "normalize-video").
	// The pipeline below handles: dataset TOML, text cache, latent cache, then training.

	// Step 1: Generate dataset TOML (file operation, not a command)
	_, err := createMusubiDatasetTOML(projectName, s, profile, configDir)
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	m.appendLog("Dataset TOML generated.")
	m.setPipelineStep("Text Cache")

	// Step 3: Cache text encoder outputs
	python := process.PythonExecutable(m.root)
	if err := validatePythonRuntime(python); err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	if err := validateMusubiSource(m.root); err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	datasetTOML, err := createMusubiDatasetTOML(projectName, s, profile, configDir)
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	spec, err := buildMusubiCommand(m.root, musubiCommandCacheText, s, datasetTOML, projectOut)
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	resp, err := m.runPipelineStep(ctx, "Text Cache", "text cache", func() (musubiCommand, error) {
		return spec, nil
	})
	if err != nil {
		cancel()
		return resp, err
	}
	if !resp.OK {
		cancel()
		return resp, nil
	}

	// Step 4: Cache latents
	spec, err = buildMusubiCommand(m.root, musubiCommandCacheLatents, s, datasetTOML, projectOut)
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	resp, err = m.runPipelineStep(ctx, "Latent Cache", "latent cache", func() (musubiCommand, error) {
		return spec, nil
	})
	if err != nil {
		cancel()
		return resp, err
	}
	if !resp.OK {
		cancel()
		return resp, nil
	}

	// Step 5: Training
	spec, err = buildMusubiCommand(m.root, musubiCommandTrain, s, datasetTOML, projectOut)
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	m.setPipelineStep("Training")
	m.appendLog(fmt.Sprintf("Starting %s training...", profile.Label))
	m.appendLog("Launching training process...")

	cmd := exec.CommandContext(ctx, spec.Program, spec.Args...)
	cmd.Dir = spec.Dir
	cmd.Env = spec.Env
	process.Prepare(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	m.mu.Lock()
	m.trainingCmd = cmd
	m.running = true
	m.activeGPUs = map[string]string{"0": profile.Label + " training"}
	m.logLines = nil
	m.mu.Unlock()

	if err := cmd.Start(); err != nil {
		cancel()
		m.mu.Lock()
		m.running = false
		m.trainingCmd = nil
		m.activeGPUs = map[string]string{}
		m.mu.Unlock()
		m.appendLog("Launch failed: " + err.Error())
		return StartResponse{OK: false, Message: err.Error()}, err
	}

	go m.pipeLogs(stdout, sampleToken)
	go m.pipeLogs(stderr, sampleToken)
	go m.waitForExit(cmd, sampleToken)
	go func() {
		<-ctx.Done()
		m.mu.Lock()
		if m.trainingCmd != nil && m.running {
			_ = process.Terminate(m.trainingCmd)
		}
		m.mu.Unlock()
	}()

	return StartResponse{OK: true, Message: "Pipeline started.", Step: "training"}, nil
}

// StartDatasetPrep dispatches a dataset preparation action.
// Supported actions: normalize-video, musubi-cache-text, musubi-cache-latents,
// musubi-dataset-toml, tag, resize, all.
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

	switch action {
	case "normalize-video":
		return m.prepNormalizeVideo(s)
	case "musubi-cache-text", "musubi-cache-latents":
		return m.prepMusubiCache(action, s)
	case "musubi-dataset-toml":
		return m.prepMusubiDatasetTOML(s)
	case "tag", "resize", "all":
		return m.prepDataset(action, s)
	default:
		return StartResponse{OK: false, Message: "Unknown dataset prep action: " + action}, nil
	}
}

// prepNormalizeVideo starts video normalization as a background goroutine.
func (m *Manager) prepNormalizeVideo(s Settings) (StartResponse, error) {
	profile := profileFor(s)
	if profile.Family != trainingFamilyMusubi {
		return StartResponse{OK: false, Message: "Video normalization is only supported for LTX 2.3 and Wan 2.2 architectures"}, nil
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return StartResponse{OK: false, Message: "ffmpeg not found in PATH; install ffmpeg before video normalization"}, nil
	}
	outputDir := preparedVideoDatasetPath(s.DatasetPath)
	m.mu.Lock()
	m.trainingCmd = nil
	m.running = true
	m.activeGPUs = map[string]string{"0": "video normalization"}
	m.logLines = nil
	m.mu.Unlock()
	go m.runVideoNormalization(s, outputDir)
	return StartResponse{OK: true, Message: "Video normalization started.", PreparedPath: filepath.ToSlash(absPath(outputDir)), Step: "normalize-video"}, nil
}

// prepMusubiCache starts a Musubi text-encoder or latent caching command asynchronously.
func (m *Manager) prepMusubiCache(action string, s Settings) (StartResponse, error) {
	profile := profileFor(s)
	if profile.Family != trainingFamilyMusubi {
		return StartResponse{OK: false, Message: "Musubi caching is only supported for LTX 2.3 and Wan 2.2 architectures"}, nil
	}
	if err := validatePythonRuntime(process.PythonExecutable(m.root)); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	if err := validateMusubiSource(m.root); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	projectName := projectNameForSettings(s)
	projectOut := outputProject(m.root, s)
	configDir := filepath.Join(projectOut, "configs")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	datasetTOML, err := createMusubiDatasetTOML(projectName, s, profile, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	kind := musubiCommandCacheText
	label := "text encoder cache"
	step := "cache-text"
	if action == "musubi-cache-latents" {
		kind = musubiCommandCacheLatents
		label = "latent cache"
		step = "cache-latents"
	}
	spec, err := buildMusubiCommand(m.root, kind, s, datasetTOML, projectOut)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	return m.startCommandAsync(spec, label, step, "Musubi "+label+" started.")
}

// prepMusubiDatasetTOML generates the Musubi dataset TOML file synchronously.
func (m *Manager) prepMusubiDatasetTOML(s Settings) (StartResponse, error) {
	profile := profileFor(s)
	if profile.Family != trainingFamilyMusubi {
		return StartResponse{OK: false, Message: "Musubi dataset TOML is only supported for LTX 2.3 and Wan 2.2 architectures"}, nil
	}
	projectName := projectNameForSettings(s)
	projectOut := outputProject(m.root, s)
	configDir := filepath.Join(projectOut, "configs")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	tomlPath, err := createMusubiDatasetTOML(projectName, s, profile, configDir)
	if err != nil {
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	m.appendLog("Musubi dataset TOML generated: " + tomlPath)
	return StartResponse{OK: true, PreparedPath: tomlPath, Message: "Dataset TOML generated."}, nil
}

// prepDataset launches a tag/resize/all dataset preparation command asynchronously.
func (m *Manager) prepDataset(action string, s Settings) (StartResponse, error) {
	if action == "tag" || action == "all" {
		if err := validatePrepModels(m.root); err != nil {
			return StartResponse{OK: false, Message: err.Error()}, err
		}
	}

	python := process.PythonExecutable(m.root)
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
	sampleDir := filepath.Join(outputProject(m.root, m.settings), "sample")
	// Find the active token for the current sample directory
	var activeToken string
	for token, dir := range m.sampleDirRegistry {
		if dir == sampleDir {
			activeToken = token
			break
		}
	}
	return map[string]any{
		"running": m.running,
		"logs":    strings.Join(m.logLines, "\n"),
		"images":  listLatestImagesTokened(sampleDir, activeToken),
	}
}

// registerSampleDir registers a sample directory and returns a URL token.
// The token is used in image URLs so the /samples/ route can resolve the
// actual filesystem path regardless of whether a custom output path is used.
func (m *Manager) registerSampleDir(dir string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Reuse existing token if dir already registered
	for token, existing := range m.sampleDirRegistry {
		if existing == dir {
			return token
		}
	}
	token := fmt.Sprintf("s%d", len(m.sampleDirRegistry)+1)
	m.sampleDirRegistry[token] = dir
	return token
}

// resolveSampleDir looks up a URL token and returns the actual sample directory.
func (m *Manager) resolveSampleDir(token string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sampleDirRegistry[token]
}

// getSampleToken returns the registered token for a sample directory, or "" if unregistered.
func (m *Manager) getSampleToken(dir string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	for token, d := range m.sampleDirRegistry {
		if d == dir {
			return token
		}
	}
	return ""
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

// pipeLogs reads from a subprocess stream, filters blacklisted lines, appends
// them to the training log, and broadcasts sample images whenever a log line
// mentions "saved" or "sample". The sampleToken parameter is a URL token that
// maps to the actual sample directory; pass "" for steps that don't produce images.
func (m *Manager) pipeLogs(reader io.Reader, sampleToken string) {
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
		if sampleToken != "" && (strings.Contains(lower, "saved") || strings.Contains(lower, "sample")) {
			sampleDir := m.resolveSampleDir(sampleToken)
			m.hub.BroadcastJSON("images", listLatestImagesTokened(sampleDir, sampleToken))
		}
	}
	if err := scanner.Err(); err != nil {
		m.appendLog("Log stream closed: " + err.Error())
	}
}

// waitForExit blocks until the subprocess exits, clears the running state,
// logs the exit reason, and broadcasts the final sample images and training
// state. The sampleToken parameter is a URL token that maps to the actual
// sample directory; pass "" for steps that don't produce images.
func (m *Manager) waitForExit(cmd *exec.Cmd, sampleToken string) {
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
	if sampleToken != "" {
		sampleDir := m.resolveSampleDir(sampleToken)
		m.hub.BroadcastJSON("images", listLatestImagesTokened(sampleDir, sampleToken))
	}
	m.hub.BroadcastJSON("training_state", map[string]bool{"running": false})
}

func trainingEnv(trainDir string) []string {
	env := os.Environ()
	pythonPath := trainDir
	if existing := os.Getenv("PYTHONPATH"); existing != "" {
		pythonPath += string(os.PathListSeparator) + existing
	}
	env = append(env, "PYTHONPATH="+pythonPath)
	env = append(env, "PYTHONIOENCODING=utf-8")
	env = append(env, "PYTHONWARNINGS=ignore")
	env = append(env, "TORCH_CPP_LOG_LEVEL=ERROR")
	env = append(env, "KMP_WARNINGS=0")
	env = append(env, "CUDA_VISIBLE_DEVICES=0")
	env = append(env, "ACCELERATE_USE_CPU=False")
	return env
}

func createTrainingBootstrap(trainDir, trainScript, configDir string) (string, error) {
	bootstrapPath := filepath.Join(configDir, "trainflow_train_bootstrap.py")
	trainDirJSON, err := json.Marshal(filepath.ToSlash(absPath(trainDir)))
	if err != nil {
		return "", err
	}
	trainScriptJSON, err := json.Marshal(filepath.ToSlash(absPath(trainScript)))
	if err != nil {
		return "", err
	}
	content := fmt.Sprintf(`import os
import runpy
import sys

train_dir = %s
train_script = %s

if train_dir not in sys.path:
    sys.path.insert(0, train_dir)
os.chdir(train_dir)
sys.argv[0] = train_script
runpy.run_path(train_script, run_name="__main__")
`, trainDirJSON, trainScriptJSON)
	if err := os.WriteFile(bootstrapPath, []byte(content), 0644); err != nil {
		return "", err
	}
	return bootstrapPath, nil
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
	maxSide := s.SideMax
	if maxSide <= 0 {
		maxSide = 768
	}
	inputDir := preparedDatasetInputPath(s)
	outputDir := preparedDatasetPath(root, s)
	script := `
import shutil
import sys
import re
from pathlib import Path
from PIL import Image, ImageOps

src = Path(sys.argv[1]).resolve()
input_dir = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()
max_side = int(sys.argv[4])

image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
output_dir.mkdir(parents=True, exist_ok=True)
input_dir.mkdir(parents=True, exist_ok=True)

def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

for child in list(src.iterdir()):
    if child.name == input_dir.name or not child.is_file():
        continue
    if child.suffix.lower() in image_exts or child.suffix.lower() == ".txt":
        target = input_dir / child.name
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite existing input file while preparing dataset: {target}")
        shutil.move(str(child), str(target))

images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts], key=natural_key)
captions_by_stem = {p.stem: p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
image_stems = {p.stem for p in images}
missing_captions = [p.name for p in images if p.stem not in captions_by_stem]
orphan_captions = [p.name for p in sorted(captions_by_stem.values(), key=natural_key) if p.stem not in image_stems]
if missing_captions or orphan_captions:
    problems = []
    if missing_captions:
        problems.append("images without exact same-name .txt captions: " + ", ".join(missing_captions))
    if orphan_captions:
        problems.append(".txt captions without exact same-name images: " + ", ".join(orphan_captions))
    raise RuntimeError("Dataset prep requires exact image/caption filename pairs before renumbering; " + "; ".join(problems))

for index, image_path in enumerate(images, 1):
    base = str(index)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        if (new_width, new_height) != image.size:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        out_image = output_dir / f"{base}.jpg"
        image.save(out_image, quality=95, subsampling=0)
    caption_src = captions_by_stem[image_path.stem]
    caption_dst = output_dir / f"{base}.txt"
    shutil.copy2(caption_src, caption_dst)
    print(f"Prepared {image_path.name} + {caption_src.name} -> {out_image.name} + {caption_dst.name} ({new_width}x{new_height})", flush=True)

print(f"Dataset prep complete. Originals are in: {input_dir}", flush=True)
print(f"Prepared numbered images/captions are in: {output_dir}", flush=True)
`
	return []string{
		"-c", script,
		filepath.ToSlash(absPath(s.DatasetPath)),
		filepath.ToSlash(absPath(inputDir)),
		filepath.ToSlash(absPath(outputDir)),
		strconv.Itoa(maxSide),
	}
}

func preparedDatasetPath(root string, s Settings) string {
	return filepath.Clean(s.DatasetPath)
}

func preparedDatasetInputPath(s Settings) string {
	return filepath.Join(filepath.Clean(s.DatasetPath), "input")
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
		if _, ok := required[file.Path]; ok && process.FileExists(file.Path) {
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
