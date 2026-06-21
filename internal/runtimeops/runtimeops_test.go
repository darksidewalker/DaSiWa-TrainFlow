package runtimeops

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFlashAttentionWheelURLUsesCUDAAndTorchABI(t *testing.T) {
	info := torchCUDAInfo{
		CUDA:      "13.0",
		Torch:     "2.9",
		PythonTag: "cp312",
		Platform:  "linux_x86_64",
		CXX11ABI:  "FALSE",
	}

	url, wheel := flashAttentionWheelURL("2.8.3", info)
	wantWheel := "flash_attn-2.8.3+cu13torch2.9cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
	if wheel != wantWheel {
		t.Fatalf("wheel = %q, want %q", wheel, wantWheel)
	}
	wantURL := "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/" + wantWheel
	if url != wantURL {
		t.Fatalf("url = %q, want %q", url, wantURL)
	}
}

func TestCUDA13IsDetected(t *testing.T) {
	for _, version := range []string{"13.0", "13.1", " 13.0"} {
		if !isCUDA13(version) {
			t.Fatalf("expected %q to be treated as CUDA 13", version)
		}
	}
	for _, version := range []string{"12.8", "11.8", ""} {
		if isCUDA13(version) {
			t.Fatalf("did not expect %q to be treated as CUDA 13", version)
		}
	}
}

func TestTorchCUDAArchListSupportsThreeDigitArchitectures(t *testing.T) {
	got := torchCUDAArchList([]string{"89", "120", "89", ""})
	want := "8.9;12.0"
	if got != want {
		t.Fatalf("arch list = %q, want %q", got, want)
	}
}

func TestUnifiedRuntimeMusubiPaths(t *testing.T) {
	root := t.TempDir()

	if got, want := musubiSourceDir(root), filepath.Join(root, "training", "musubi-tuner"); got != want {
		t.Fatalf("musubi source dir = %q, want %q", got, want)
	}
	if got, want := musubiOverlayRequirementsPath(root), filepath.Join(root, "training", "requirements-musubi-overlay.txt"); got != want {
		t.Fatalf("musubi overlay requirements path = %q, want %q", got, want)
	}

	pythonPath := musubiPythonPath(root)
	for _, want := range []string{
		filepath.Join(root, "training", "musubi-tuner", "src"),
		filepath.Join(root, "training", "musubi-tuner"),
		filepath.Join(root, "training", "sd-scripts"),
	} {
		if !strings.Contains(pythonPath, want) {
			t.Fatalf("musubi python path %q does not contain %q", pythonPath, want)
		}
	}
	if strings.Contains(pythonPath, ".venv") {
		t.Fatalf("musubi python path must not reference a separate venv: %q", pythonPath)
	}
}

func TestUpdateAppBinariesSkipsNonGitCheckout(t *testing.T) {
	root := t.TempDir()
	var logs []string
	err := UpdateAppBinaries(root, func(line string) { logs = append(logs, line) })
	if err != nil {
		t.Fatalf("UpdateAppBinaries returned error for non-git checkout: %v", err)
	}
	if len(logs) != 1 || !strings.Contains(logs[0], "not a git checkout") {
		t.Fatalf("logs = %q, want non-git checkout skip message", logs)
	}
}

func TestPathExists(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "file.txt")
	if pathExists(path) {
		t.Fatalf("pathExists(%q) = true before file exists", path)
	}
	if err := os.WriteFile(path, []byte("ok"), 0644); err != nil {
		t.Fatal(err)
	}
	if !pathExists(path) {
		t.Fatalf("pathExists(%q) = false after file exists", path)
	}
}

func TestPublicGitRemoteURLConvertsGitHubSSHToHTTPS(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	root := t.TempDir()
	runGitForTest(t, root, "init")
	runGitForTest(t, root, "remote", "add", "origin", "git@github.com:darksidewalker/DaSiWa-TrainFlow.git")

	got, err := publicGitRemoteURL(root, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	want := "https://github.com/darksidewalker/DaSiWa-TrainFlow.git"
	if got != want {
		t.Fatalf("publicGitRemoteURL = %q, want %q", got, want)
	}
}

func TestCurrentGitBranchFallsBackToMain(t *testing.T) {
	root := t.TempDir()
	var logs []string
	got := currentGitBranch(root, func(line string) { logs = append(logs, line) })
	if got != "main" {
		t.Fatalf("currentGitBranch = %q, want main", got)
	}
	if len(logs) != 1 || !strings.Contains(logs[0], "using main") {
		t.Fatalf("logs = %q, want fallback message", logs)
	}
}

func runGitForTest(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, out)
	}
}

func TestMatchesPrebuiltFlashAttentionWheel(t *testing.T) {
	info := torchCUDAInfo{
		CUDA:      "13.0",
		Torch:     "2.12",
		PythonTag: "cp312",
		Platform:  "linux_x86_64",
	}
	matches := []string{
		"flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl",
		"flash_attn-2.8.3+cu130torch2.12-cp312-cp312-manylinux_2_28_x86_64.whl",
	}
	for _, name := range matches {
		if !matchesPrebuiltFlashAttentionWheel(name, info) {
			t.Fatalf("expected %q to match", name)
		}
	}

	misses := []string{
		"flash_attn-2.8.3+cu13torch2.12cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
		"flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl",
		"flash_attn-2.8.3+cu130torch2.12-cp313-cp313-linux_x86_64.whl",
		"flash_attn_3-3.0.0+cu130torch2.12-abi3-abi3-linux_x86_64.whl",
	}
	for _, name := range misses {
		if matchesPrebuiltFlashAttentionWheel(name, info) {
			t.Fatalf("did not expect %q to match", name)
		}
	}
}
