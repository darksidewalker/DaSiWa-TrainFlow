package runtimeops

import (
	"archive/zip"
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type Logger func(string)

const (
	pythonVersion = "3.12.10"
	pythonTag     = "312"
)

func InstallRequirements(root string, installFlashAttention bool, log Logger) error {
	python, err := ensurePython(root, log)
	if err != nil {
		return err
	}
	installer := prepareInstaller(root, python, log)
	log("Installing PyTorch CUDA 13.0 wheels...")
	if err := installer.install("--upgrade", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu130"); err != nil {
		return err
	}
	return installTrainerDeps(root, installer, installFlashAttention, log)
}

func UpdateRuntime(root string, keepBackup bool, installFlashAttention bool, log Logger) error {
	if runtime.GOOS == "windows" {
		if err := installWindowsEmbeddedPython(root, keepBackup, log); err != nil {
			return err
		}
	} else {
		if err := installLinuxLocalPython(root, keepBackup, log); err != nil {
			return err
		}
	}
	return InstallRequirements(root, installFlashAttention, log)
}

func Verify(root string, log Logger) error {
	python, err := ensurePython(root, log)
	if err != nil {
		return err
	}
	return run(log, root, python, "-c", "import sys, torch; print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())")
}

func ensurePython(root string, log Logger) (string, error) {
	python := pythonExecutable(root)
	if python != "" {
		return python, nil
	}
	if runtime.GOOS == "windows" {
		return "", errors.New("python_embeded\\windows\\python.exe was not found; run Update Runtime first")
	}
	log("Local Linux runtime not found; creating python_embeded/linux venv...")
	if err := installLinuxLocalPython(root, false, log); err != nil {
		return "", err
	}
	python = pythonExecutable(root)
	if python == "" {
		return "", errors.New("failed to create python_embeded runtime")
	}
	return python, nil
}

func installTrainerDeps(root string, installer dependencyInstaller, installFlashAttention bool, log Logger) error {
	sdScriptsDir := filepath.Join(root, "training", "sd-scripts")
	log("Upgrading pip, setuptools, and wheel...")
	if err := installer.install("--upgrade", "pip", "setuptools", "wheel"); err != nil {
		return err
	}
	req := filepath.Join(sdScriptsDir, "requirements.txt")
	log("Installing sd-scripts requirements...")
	if err := installer.installIn(sdScriptsDir, "-r", req); err != nil {
		return err
	}
	log("Installing sd-scripts editable package...")
	if err := installer.installIn(sdScriptsDir, "-e", sdScriptsDir); err != nil {
		return err
	}
	log("Installing TrainFlow UI/prep dependencies...")
	if err := installer.install("gradio", "psutil", "toml", "pillow", "onnxruntime-gpu", "pandas", "opencv-python"); err != nil {
		return err
	}
	if !installFlashAttention {
		log("Skipping optional Flash Attention install.")
		return nil
	}
	installOptionalFlashAttention(installer, log)
	return nil
}

func installOptionalFlashAttention(installer dependencyInstaller, log Logger) {
	if runtime.GOOS == "windows" {
		log("Skipping optional flash-attn install on Windows.")
		return
	}
	log("Installing optional Flash Attention support...")
	if err := installer.install("--only-binary=:all:", "flash-attn"); err == nil {
		return
	} else {
		log("No compatible prebuilt flash-attn wheel was found; trying optimized source build: " + err.Error())
	}
	log("Installing Flash Attention build helpers...")
	if err := installer.install("ninja", "packaging"); err != nil {
		log("Flash Attention build helper install failed; skipping source build: " + err.Error())
		return
	}
	jobs := flashAttentionBuildJobs()
	env := []string{
		"MAX_JOBS=" + jobs,
		"NVCC_THREADS=" + jobs,
		"FLASH_ATTENTION_FORCE_BUILD=TRUE",
	}
	log("Building Flash Attention from source with MAX_JOBS=" + jobs + "...")
	if err := installer.installWithEnv(env, "--no-build-isolation", "flash-attn"); err != nil {
		log("Optional flash-attn source build failed; Flash Attention checkbox will require a manual compatible install: " + err.Error())
	}
}

func flashAttentionBuildJobs() string {
	cpus := runtime.NumCPU()
	if cpus < 1 {
		cpus = 1
	}
	if cpus > 8 {
		cpus = 8
	}
	return fmt.Sprintf("%d", cpus)
}

type dependencyInstaller struct {
	root   string
	python string
	uv     string
	log    Logger
}

func prepareInstaller(root, python string, log Logger) dependencyInstaller {
	log("Bootstrapping uv inside the embedded runtime...")
	if err := run(log, root, python, "-m", "pip", "install", "--upgrade", "pip", "uv"); err != nil {
		log("uv bootstrap failed; falling back to pip: " + err.Error())
		return dependencyInstaller{root: root, python: python, log: log}
	}
	uv := uvExecutable(root)
	if uv == "" {
		log("uv executable was not found after install; falling back to pip.")
		return dependencyInstaller{root: root, python: python, log: log}
	}
	if err := run(log, root, uv, "--version"); err != nil {
		log("uv verification failed; falling back to pip: " + err.Error())
		return dependencyInstaller{root: root, python: python, log: log}
	}
	log("Using uv for dependency installation.")
	return dependencyInstaller{root: root, python: python, uv: uv, log: log}
}

func (d dependencyInstaller) install(args ...string) error {
	return d.installIn(d.root, args...)
}

func (d dependencyInstaller) installIn(workDir string, args ...string) error {
	return d.installInWithEnv(workDir, nil, args...)
}

func (d dependencyInstaller) installWithEnv(env []string, args ...string) error {
	return d.installInWithEnv(d.root, env, args...)
}

func (d dependencyInstaller) installInWithEnv(workDir string, env []string, args ...string) error {
	if d.uv != "" {
		uvArgs := append([]string{"pip", "install", "--python", d.python}, args...)
		if err := runWithEnv(d.log, workDir, env, d.uv, uvArgs...); err == nil {
			return nil
		} else {
			d.log("uv install failed; retrying with pip: " + err.Error())
		}
	}
	pipArgs := append([]string{"-m", "pip", "install"}, args...)
	return runWithEnv(d.log, workDir, env, d.python, pipArgs...)
}

func installWindowsEmbeddedPython(root string, keepBackup bool, log Logger) error {
	tempDir := filepath.Join(root, ".runtime-update")
	pythonDir := platformRuntimeDir(root)
	var backupDir string
	zipPath := filepath.Join(tempDir, "python-"+pythonVersion+"-embed-amd64.zip")
	getPipPath := filepath.Join(tempDir, "get-pip.py")
	pythonURL := "https://www.python.org/ftp/python/" + pythonVersion + "/python-" + pythonVersion + "-embed-amd64.zip"

	log("Preparing runtime update folder...")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return err
	}
	if _, err := os.Stat(pythonDir); err == nil {
		backupDir = filepath.Join(root, "python_embeded_windows_backup_"+time.Now().Format("20060102_150405"))
		log("Backing up existing Windows runtime to " + backupDir)
		if err := os.Rename(pythonDir, backupDir); err != nil {
			return err
		}
	}
	success := false
	defer cleanupBackup(backupDir, keepBackup, &success, log)
	if err := os.MkdirAll(pythonDir, 0755); err != nil {
		return err
	}
	if err := download(log, pythonURL, zipPath); err != nil {
		return err
	}
	log("Extracting Python " + pythonVersion + " embeddable package...")
	if err := unzip(zipPath, pythonDir); err != nil {
		return err
	}
	pth := filepath.Join(pythonDir, "python"+pythonTag+"._pth")
	if err := enableSitePackages(pth); err != nil {
		return err
	}
	if err := download(log, "https://bootstrap.pypa.io/get-pip.py", getPipPath); err != nil {
		return err
	}
	log("Installing pip...")
	if err := run(log, root, filepath.Join(pythonDir, "python.exe"), getPipPath); err != nil {
		return err
	}
	success = true
	return nil
}

func installLinuxLocalPython(root string, keepBackup bool, log Logger) error {
	pythonDir := platformRuntimeDir(root)
	var backupDir string
	if _, err := os.Stat(pythonDir); err == nil {
		backupDir = filepath.Join(root, "python_embeded_linux_backup_"+time.Now().Format("20060102_150405"))
		log("Backing up existing Linux runtime to " + backupDir)
		if err := os.Rename(pythonDir, backupDir); err != nil {
			return err
		}
	}
	success := false
	defer cleanupBackup(backupDir, keepBackup, &success, log)
	if err := os.MkdirAll(filepath.Dir(pythonDir), 0755); err != nil {
		return err
	}
	basePython := findCommand("python3.12", "python3")
	if basePython == "" {
		return errors.New("python3.12 or python3 was not found on PATH; install Python 3.12 first")
	}
	log("Creating Linux local runtime with " + basePython)
	if err := run(log, root, basePython, "-m", "venv", pythonDir); err != nil {
		return err
	}
	python := filepath.Join(pythonDir, "bin", "python")
	log("Bootstrapping pip in local runtime...")
	if err := run(log, root, python, "-m", "ensurepip", "--upgrade"); err != nil {
		return err
	}
	success = true
	return nil
}

func cleanupBackup(backupDir string, keepBackup bool, success *bool, log Logger) {
	if backupDir == "" {
		return
	}
	if !*success {
		log("Update did not complete; keeping backup at " + backupDir)
		return
	}
	if keepBackup {
		log("Keeping backup at " + backupDir)
		return
	}
	log("Deleting backup after successful update: " + backupDir)
	if err := os.RemoveAll(backupDir); err != nil {
		log("Could not delete backup: " + err.Error())
	}
}

func pythonExecutable(root string) string {
	var candidates []string
	if runtime.GOOS == "windows" {
		candidates = []string{
			filepath.Join(root, "python_embeded", "windows", "python.exe"),
			filepath.Join(root, "python_embeded", "python.exe"),
		}
	} else {
		candidates = []string{
			filepath.Join(root, "python_embeded", "linux", "bin", "python"),
			filepath.Join(root, "python_embeded", "linux", "bin", "python3"),
			filepath.Join(root, "python_embeded", "bin", "python"),
			filepath.Join(root, "python_embeded", "bin", "python3"),
			filepath.Join(root, "python_embeded", "python"),
		}
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}

func uvExecutable(root string) string {
	var candidates []string
	if runtime.GOOS == "windows" {
		candidates = []string{
			filepath.Join(root, "python_embeded", "windows", "Scripts", "uv.exe"),
			filepath.Join(root, "python_embeded", "windows", "uv.exe"),
			filepath.Join(root, "python_embeded", "Scripts", "uv.exe"),
			filepath.Join(root, "python_embeded", "uv.exe"),
		}
	} else {
		candidates = []string{
			filepath.Join(root, "python_embeded", "linux", "bin", "uv"),
			filepath.Join(root, "python_embeded", "bin", "uv"),
		}
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	if path, err := exec.LookPath("uv"); err == nil {
		return path
	}
	return ""
}

func platformRuntimeDir(root string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(root, "python_embeded", "windows")
	}
	return filepath.Join(root, "python_embeded", "linux")
}

func findCommand(names ...string) string {
	for _, name := range names {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}
	return ""
}

func download(log Logger, url, path string) error {
	log("Downloading " + url)
	client := http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("download failed: %s", resp.Status)
	}
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

func unzip(src, dest string) error {
	reader, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer reader.Close()
	for _, file := range reader.File {
		target := filepath.Join(dest, file.Name)
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(dest)+string(os.PathSeparator)) {
			return fmt.Errorf("unsafe zip path: %s", file.Name)
		}
		if file.FileInfo().IsDir() {
			if err := os.MkdirAll(target, file.Mode()); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}
		in, err := file.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, file.Mode())
		if err != nil {
			_ = in.Close()
			return err
		}
		_, copyErr := io.Copy(out, in)
		closeErr := out.Close()
		_ = in.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func enableSitePackages(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	lines := strings.Split(string(data), "\n")
	hasSitePackages := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "#import site" {
			lines[i] = "import site"
		}
		if trimmed == "Lib\\site-packages" || trimmed == "Lib/site-packages" {
			hasSitePackages = true
		}
	}
	if !hasSitePackages {
		lines = append(lines, "Lib\\site-packages")
	}
	return os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0644)
}

func run(log Logger, workDir, name string, args ...string) error {
	return runWithEnv(log, workDir, nil, name, args...)
}

func runWithEnv(log Logger, workDir string, env []string, name string, args ...string) error {
	prefix := ""
	if len(env) > 0 {
		prefix = strings.Join(env, " ") + " "
	}
	log("$ " + prefix + name + " " + strings.Join(args, " "))
	ctx := context.Background()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = workDir
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	done := make(chan struct{})
	go stream(stdout, log, done)
	go stream(stderr, log, done)
	err = cmd.Wait()
	<-done
	<-done
	if err != nil {
		return err
	}
	return nil
}

func stream(reader io.Reader, log Logger, done chan<- struct{}) {
	defer func() { done <- struct{}{} }()
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		log(scanner.Text())
	}
}
