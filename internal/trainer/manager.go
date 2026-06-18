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

	go m.pipeLogs(stdout, sampleDir)
	go m.pipeLogs(stderr, sampleDir)
	go m.waitForExit(cmd, sampleDir)
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

func isVideoFile(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	return ext == ".mp4" || ext == ".mkv" || ext == ".mov" || ext == ".avi" || ext == ".webm" || ext == ".m4v"
}

// startMusubiSequenced runs the full Musubi pipeline in sequence:
// 1. Video normalization (if dataset contains video files)
// 2. Dataset TOML generation
// 3. Text encoder cache
// 4. Latent cache
// 5. Training
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

	m.appendLog(fmt.Sprintf("Starting %s (%s) pipeline...", projectName, profile.Label))

	// Use a cancellable context for the entire pipeline
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Step 1: Video normalization (only if dataset path contains video files)
	videoOutputDir := preparedVideoDatasetPath(m.root, s)
	normalizeNeeded := false
	if dirExists(s.DatasetPath) {
		if files, err := os.ReadDir(s.DatasetPath); err == nil {
			for _, f := range files {
				if !f.IsDir() && isVideoFile(f.Name()) {
					normalizeNeeded = true
					break
				}
			}
		}
	}

	if normalizeNeeded {
		if _, err := exec.LookPath("ffmpeg"); err != nil {
			return StartResponse{OK: false, Message: "ffmpeg not found in PATH; install ffmpeg before video normalization"}, nil
		}
		resp, err := m.runPipelineStep(ctx, "Video Normalization", "video normalization", func() (musubiCommand, error) {
			// Build a normalizer command using the normalizer tool
			args := []string{
				"-input", s.DatasetPath,
				"-output", videoOutputDir,
				"-codec", s.VideoCodec,
				"-quality", s.VideoQuality,
				"-encoder_preset", s.VideoEncoderPreset,
			}
			if s.VideoWidth > 0 {
				args = append(args, "-w", strconv.Itoa(s.VideoWidth))
			}
			if s.VideoHeight > 0 {
				args = append(args, "-h", strconv.Itoa(s.VideoHeight))
			}
			if s.VideoFPS > 0 {
				args = append(args, "-fps", strconv.Itoa(s.VideoFPS))
			}
			if s.VideoDuration != "" {
				args = append(args, "-len", s.VideoDuration)
			}
			if s.VideoSpeed != "" {
				args = append(args, "-speed", s.VideoSpeed)
			}
			if s.VideoSkipFrames > 0 {
				args = append(args, "-skip", strconv.Itoa(s.VideoSkipFrames))
			}
			if !s.VideoIncludeAudio {
				args = append(args, "-noaudio")
			}
			if s.VideoExtraArgs != "" {
				args = append(args, strings.Split(s.VideoExtraArgs, " ")...)
			}
			return musubiCommand{
				Program: "normalize-video",
				Args:    args,
				Dir:     m.root,
				Env:     os.Environ(),
			}, nil
		})
		if err != nil {
			cancel()
			return resp, err
		}
		if !resp.OK {
			cancel()
			return resp, nil
		}
		// Update dataset path to normalized output for subsequent steps
		s.DatasetPath = videoOutputDir
	}

	// Step 2: Generate dataset TOML (file operation, not a command)
	_, err := createMusubiDatasetTOML(projectName, s, profile, configDir)
	if err != nil {
		cancel()
		return StartResponse{OK: false, Message: err.Error()}, err
	}
	m.appendLog("Dataset TOML generated.")
	m.setPipelineStep("Text Cache")

	// Step 3: Cache text encoder outputs
	python := pythonExecutable(m.root)
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

	go m.pipeLogs(stdout, sampleDir)
	go m.pipeLogs(stderr, sampleDir)
	go m.waitForExit(cmd, sampleDir)
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
	if action == "normalize-video" {
		profile := profileFor(s)
		if profile.Family != trainingFamilyMusubi {
			return StartResponse{OK: false, Message: "Video normalization is only supported for LTX 2.3 and Wan 2.2 architectures"}, nil
		}
		if _, err := exec.LookPath("ffmpeg"); err != nil {
			return StartResponse{OK: false, Message: "ffmpeg not found in PATH; install ffmpeg before video normalization"}, nil
		}
		outputDir := preparedVideoDatasetPath(m.root, s)
		m.mu.Lock()
		m.trainingCmd = nil
		m.running = true
		m.activeGPUs = map[string]string{"0": "video normalization"}
		m.logLines = nil
		m.mu.Unlock()
		go m.runVideoNormalization(s, outputDir)
		return StartResponse{OK: true, Message: "Video normalization started.", PreparedPath: filepath.ToSlash(absPath(outputDir)), Step: "normalize-video"}, nil
	}
	if action == "musubi-cache-text" || action == "musubi-cache-latents" {
		profile := profileFor(s)
		if profile.Family != trainingFamilyMusubi {
			return StartResponse{OK: false, Message: "Musubi caching is only supported for LTX 2.3 and Wan 2.2 architectures"}, nil
		}
		if err := validatePythonRuntime(pythonExecutable(m.root)); err != nil {
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
	if action == "musubi-dataset-toml" {
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
