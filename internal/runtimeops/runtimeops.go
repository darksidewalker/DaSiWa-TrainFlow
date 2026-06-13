package runtimeops

import (
	"archive/zip"
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
	"runtime"
	"strings"
	"time"
)

type Logger func(string)

const (
	pythonVersion = "3.12.10"
	pythonTag     = "312"

	flashAttentionPackage               = "flash_attn"
	flashAttentionPyPIName              = "flash-attn"
	flashAttentionMinBuildMemoryBytes   = uint64(96 * 1024 * 1024 * 1024)
	flashAttentionDefaultReleaseVersion = "2.8.3"
	flashAttentionPrebuildReleasesAPI   = "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=100"
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
	info, infoErr := detectTorchCUDAInfo(installer.python)
	if infoErr != nil {
		log("Could not detect Torch/CUDA ABI for flash-attn wheel matching: " + infoErr.Error())
	}
	log("Installing optional Flash Attention support from a prebuilt wheel only...")
	if wheelURL := strings.TrimSpace(os.Getenv("DASIWA_FLASH_ATTN_WHEEL_URL")); wheelURL != "" {
		log("Trying Flash Attention wheel from DASIWA_FLASH_ATTN_WHEEL_URL...")
		if err := installer.install(wheelURL); err == nil {
			log("Installed Flash Attention wheel from DASIWA_FLASH_ATTN_WHEEL_URL.")
			return
		} else {
			log("Flash Attention wheel URL failed: " + err.Error())
		}
	}
	if infoErr == nil {
		version := flashAttentionReleaseVersion(log)
		wheelURL, wheelName := flashAttentionWheelURL(version, info)
		log("Trying Flash Attention release wheel: " + wheelName)
		if err := installer.install(wheelURL); err == nil {
			log("Installed Flash Attention prebuilt release wheel.")
			return
		} else {
			log("No compatible Flash Attention release wheel was found: " + err.Error())
		}
	}
	if infoErr == nil {
		candidate, err := findPrebuiltFlashAttentionWheel(info)
		if err != nil {
			log("Could not search flash-attention-prebuild-wheels releases: " + err.Error())
		} else if candidate.URL != "" {
			log("Trying community prebuilt Flash Attention wheel from " + candidate.Source + ": " + candidate.Name)
			if err := installer.install(candidate.URL); err == nil {
				log("Installed community prebuilt Flash Attention wheel.")
				return
			} else {
				log("Community prebuilt Flash Attention wheel failed: " + err.Error())
			}
		} else {
			log("No matching community prebuilt Flash Attention wheel was found.")
		}
	}
	if err := installer.install("--only-binary=:all:", flashAttentionPyPIName); err == nil {
		log("Installed Flash Attention prebuilt PyPI wheel.")
		return
	} else {
		log("No compatible prebuilt Flash Attention PyPI wheel was found: " + err.Error())
	}
	if info.CUDA != "" && isCUDA13(info.CUDA) {
		log("CUDA " + info.CUDA + " detected. Skipping flash-attn source build for the CUDA 13.0 runtime; leave TrainFlow Flash Attention off and use the default torch/SDPA attention path.")
		return
	}
	if infoErr != nil {
		log("Skipping flash-attn source build because the Torch/CUDA ABI could not be verified.")
		return
	}
	if os.Getenv("DASIWA_FLASH_ATTN_SOURCE_BUILD") != "1" {
		log("Skipping flash-attn source build because it can require extreme RAM/swap. Set DASIWA_FLASH_ATTN_SOURCE_BUILD=1 to opt in manually.")
		return
	}
	if ok, detail, err := flashAttentionBuildMemoryOK(); err != nil {
		log("Skipping flash-attn source build because available RAM/swap could not be checked: " + err.Error())
		return
	} else if !ok {
		log("Skipping flash-attn source build because available RAM/swap is too low (" + detail + "); require at least 96 GiB to avoid OOM.")
		return
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
		"CMAKE_BUILD_PARALLEL_LEVEL=" + jobs,
		"NINJAFLAGS=-j" + jobs,
		"FLASH_ATTENTION_FORCE_BUILD=TRUE",
	}
	if archs := flashAttentionCUDAArchs(info.Archs); archs != "" {
		env = append(env, "FLASH_ATTN_CUDA_ARCHS="+archs, "TORCH_CUDA_ARCH_LIST="+torchCUDAArchList(info.Archs))
	}
	log("Building Flash Attention from source with MAX_JOBS=" + jobs + "...")
	if err := installer.installWithEnv(env, "--no-build-isolation", flashAttentionPyPIName); err != nil {
		log("Optional flash-attn source build failed; Flash Attention checkbox will require a manual compatible install: " + err.Error())
	}
}

func flashAttentionBuildJobs() string {
	return "1"
}

type torchCUDAInfo struct {
	CUDA      string   `json:"cuda"`
	Torch     string   `json:"torch"`
	PythonTag string   `json:"python_tag"`
	Platform  string   `json:"platform"`
	CXX11ABI  string   `json:"cxx11abi"`
	Archs     []string `json:"archs"`
}

type flashAttentionWheelCandidate struct {
	Name   string
	URL    string
	Source string
}

func detectTorchCUDAInfo(python string) (torchCUDAInfo, error) {
	script := `
import json
import platform
import sys
import torch

platform_name = "linux_" + platform.uname().machine
archs = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        archs.append(f"{major}{minor}")
torch_version = torch.__version__.split("+", 1)[0].split(".")
print(json.dumps({
    "cuda": torch.version.cuda or "",
    "torch": ".".join(torch_version[:2]),
    "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
    "platform": platform_name,
    "cxx11abi": str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(),
    "archs": sorted(set(archs)),
}))
`
	cmd := exec.Command(python, "-c", script)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return torchCUDAInfo{}, fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
	}
	var info torchCUDAInfo
	if err := json.Unmarshal(out, &info); err != nil {
		return torchCUDAInfo{}, err
	}
	if info.CUDA == "" || info.Torch == "" || info.PythonTag == "" || info.Platform == "" || info.CXX11ABI == "" {
		return torchCUDAInfo{}, errors.New("incomplete Torch/CUDA info")
	}
	return info, nil
}

func findPrebuiltFlashAttentionWheel(info torchCUDAInfo) (flashAttentionWheelCandidate, error) {
	req, err := http.NewRequest(http.MethodGet, flashAttentionPrebuildReleasesAPI, nil)
	if err != nil {
		return flashAttentionWheelCandidate{}, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "DaSiWa-TrainFlow")
	client := http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return flashAttentionWheelCandidate{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return flashAttentionWheelCandidate{}, fmt.Errorf("GitHub returned %s", resp.Status)
	}
	var releases []struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name               string `json:"name"`
			BrowserDownloadURL string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&releases); err != nil {
		return flashAttentionWheelCandidate{}, err
	}
	var best flashAttentionWheelCandidate
	bestScore := -1
	for _, release := range releases {
		for _, asset := range release.Assets {
			if !matchesPrebuiltFlashAttentionWheel(asset.Name, info) {
				continue
			}
			score := prebuiltFlashAttentionWheelScore(asset.Name)
			if score > bestScore {
				bestScore = score
				best = flashAttentionWheelCandidate{
					Name:   asset.Name,
					URL:    asset.BrowserDownloadURL,
					Source: "mjun0812/flash-attention-prebuild-wheels@" + release.TagName,
				}
			}
		}
	}
	return best, nil
}

func matchesPrebuiltFlashAttentionWheel(name string, info torchCUDAInfo) bool {
	if !strings.HasPrefix(name, flashAttentionPackage+"-") || !strings.HasSuffix(name, ".whl") {
		return false
	}
	cudaLabel := "cu" + strings.ReplaceAll(info.CUDA, ".", "")
	if !strings.Contains(name, "+"+cudaLabel+"torch"+info.Torch+"-") {
		return false
	}
	if !strings.Contains(name, "-"+info.PythonTag+"-"+info.PythonTag+"-") && !strings.Contains(name, "-abi3-") {
		return false
	}
	return wheelPlatformMatches(name, info.Platform)
}

func wheelPlatformMatches(name, platform string) bool {
	switch {
	case strings.Contains(platform, "x86_64") || strings.Contains(platform, "amd64"):
		return strings.Contains(name, "x86_64") || strings.Contains(name, "amd64")
	case strings.Contains(platform, "aarch64") || strings.Contains(platform, "arm64"):
		return strings.Contains(name, "aarch64") || strings.Contains(name, "arm64")
	default:
		return strings.Contains(name, platform)
	}
}

func prebuiltFlashAttentionWheelScore(name string) int {
	score := 0
	if strings.Contains(name, "manylinux") {
		score += 100
	}
	if strings.Contains(name, "2.8.3") {
		score += 30
	}
	if strings.Contains(name, "2.7.4") {
		score += 20
	}
	if strings.Contains(name, "2.6.3") {
		score += 10
	}
	return score
}

func flashAttentionReleaseVersion(log Logger) string {
	if version := strings.TrimSpace(os.Getenv("DASIWA_FLASH_ATTN_VERSION")); version != "" {
		log("Using Flash Attention version from DASIWA_FLASH_ATTN_VERSION=" + version)
		return version
	}
	version, err := latestFlashAttentionVersion()
	if err != nil {
		log("Could not read latest Flash Attention version from PyPI; trying " + flashAttentionDefaultReleaseVersion + ": " + err.Error())
		return flashAttentionDefaultReleaseVersion
	}
	return version
}

func latestFlashAttentionVersion() (string, error) {
	client := http.Client{Timeout: 20 * time.Second}
	resp, err := client.Get("https://pypi.org/pypi/" + flashAttentionPyPIName + "/json")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("PyPI returned %s", resp.Status)
	}
	var body struct {
		Info struct {
			Version string `json:"version"`
		} `json:"info"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	if body.Info.Version == "" {
		return "", errors.New("PyPI response did not include info.version")
	}
	return body.Info.Version, nil
}

func flashAttentionWheelURL(version string, info torchCUDAInfo) (string, string) {
	cudaMajor := strings.Split(info.CUDA, ".")[0]
	wheelName := fmt.Sprintf(
		"%s-%s+cu%storch%scxx11abi%s-%s-%s-%s.whl",
		flashAttentionPackage,
		version,
		cudaMajor,
		info.Torch,
		info.CXX11ABI,
		info.PythonTag,
		info.PythonTag,
		info.Platform,
	)
	return fmt.Sprintf(
		"https://github.com/Dao-AILab/flash-attention/releases/download/v%s/%s",
		version,
		wheelName,
	), wheelName
}

func isCUDA13(version string) bool {
	return strings.HasPrefix(strings.TrimSpace(version), "13.")
}

func flashAttentionBuildMemoryOK() (bool, string, error) {
	if runtime.GOOS != "linux" {
		return false, "", errors.New("only Linux /proc/meminfo is supported")
	}
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return false, "", err
	}
	values := map[string]uint64{}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		key := strings.TrimSuffix(fields[0], ":")
		var value uint64
		if _, err := fmt.Sscanf(fields[1], "%d", &value); err != nil {
			continue
		}
		values[key] = value * 1024
	}
	available := values["MemAvailable"] + values["SwapFree"]
	if available == 0 {
		return false, "", errors.New("MemAvailable and SwapFree were not present")
	}
	return available >= flashAttentionMinBuildMemoryBytes, humanBytes(available), nil
}

func humanBytes(value uint64) string {
	const gib = 1024 * 1024 * 1024
	if value >= gib {
		return fmt.Sprintf("%.1f GiB", float64(value)/float64(gib))
	}
	const mib = 1024 * 1024
	return fmt.Sprintf("%.1f MiB", float64(value)/float64(mib))
}

func flashAttentionCUDAArchs(archs []string) string {
	seen := map[string]bool{}
	var out []string
	for _, arch := range archs {
		arch = strings.TrimSpace(arch)
		if arch == "" || seen[arch] {
			continue
		}
		seen[arch] = true
		out = append(out, arch)
	}
	return strings.Join(out, ";")
}

func torchCUDAArchList(archs []string) string {
	seen := map[string]bool{}
	var out []string
	for _, arch := range archs {
		arch = strings.TrimSpace(arch)
		if len(arch) < 2 || seen[arch] {
			continue
		}
		seen[arch] = true
		out = append(out, arch[:len(arch)-1]+"."+arch[len(arch)-1:])
	}
	return strings.Join(out, ";")
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
