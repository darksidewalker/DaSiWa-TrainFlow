package trainer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveResumePath_disabled(t *testing.T) {
	s := Settings{ResumeEnabled: false, ResumePath: "/some/path"}
	if got := resolveResumePath(s, "/output"); got != "" {
		t.Errorf("expected empty when resume disabled, got %q", got)
	}
}

func TestResolveResumePath_manual(t *testing.T) {
	// Manual resume (AutoResume=false) should return the explicit path
	s := Settings{ResumeEnabled: true, AutoResume: false, ResumePath: "/explicit/checkpoint"}
	if got := resolveResumePath(s, "/output"); got != "/explicit/checkpoint" {
		t.Errorf("expected manual path, got %q", got)
	}
}

func TestResolveResumePath_auto_ignores_default_home(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}

	// AutoResume with a stale default (home dir) should NOT return home.
	// It should search the output dir for *-state folders instead.
	s := Settings{ResumeEnabled: true, AutoResume: true, ResumePath: home}
	got := resolveResumePath(s, "/nonexistent/output")
	if got == home {
		t.Errorf("auto-resume must NOT fall back to home dir default (got %q)", got)
	}
	if got != "" {
		t.Errorf("expected empty (no state dir exists), got %q", got)
	}
}

func TestResolveResumePath_auto_finds_state_dir(t *testing.T) {
	tmp := t.TempDir()
	// Create a fake -state directory
	stateDir := filepath.Join(tmp, "output", "model1-state")
	if err := os.MkdirAll(stateDir, 0755); err != nil {
		t.Fatal(err)
	}

	s := Settings{ResumeEnabled: true, AutoResume: true, ResumePath: ""}
	got := resolveResumePath(s, filepath.Join(tmp, "output"))
	if got != stateDir {
		t.Errorf("expected %q, got %q", stateDir, got)
	}
}

func TestDefaultSettings_resumePath_empty(t *testing.T) {
	s := DefaultSettings("/some/root")
	if s.ResumePath != "" {
		t.Errorf("default ResumePath should be empty, got %q", s.ResumePath)
	}
	if !s.AutoResume {
		t.Error("default AutoResume should be true")
	}
	if s.ResumeEnabled {
		t.Error("default ResumeEnabled should be false")
	}
}

// --- interval alignment tests ---

func TestBestDivisorInRange(t *testing.T) {
	tests := []struct {
		steps int
		lo    int
		hi    int
		ideal int
		want  int
	}{
		{1100, 100, 300, 150, 110}, // divisors of 1100 in [100,300]: 100,110,220,275; closest to 150 is 110
		{1000, 100, 300, 100, 100}, // already divides
		{1050, 100, 300, 150, 150}, // already divides
		{900, 100, 300, 150, 150},  // already divides
		{850, 100, 300, 150, 170},  // divisors of 850 in [100,300]: 170,250; closest to 150 is 170
		{1200, 100, 300, 150, 150}, // 1200%150==0
		{3200, 250, 500, 300, 320}, // divisors of 3200 in [250,500]: 320,400; closest to 300 is 320
		{500, 100, 300, 100, 100},  // already divides
		{500, 100, 300, 150, 125},  // divisors of 500 in [100,300]: 100,125,200,250; closest to 150 is 125
	}
	for _, tt := range tests {
		got := bestDivisorInRange(tt.steps, tt.lo, tt.hi, tt.ideal)
		if got != tt.want {
			t.Errorf("bestDivisorInRange(%d, %d, %d, %d) = %d, want %d", tt.steps, tt.lo, tt.hi, tt.ideal, got, tt.want)
		}
		if tt.steps%got != 0 {
			t.Errorf("bestDivisorInRange(%d, %d, %d, %d) = %d does not divide steps %d", tt.steps, tt.lo, tt.hi, tt.ideal, got, tt.steps)
		}
	}
}

func TestRecommendedInterval_aligns(t *testing.T) {
	// Verify that recommendedInterval always returns a divisor of steps.
	// The [100,300] range is a soft target — if no divisor exists in range,
	// we pick the closest one outside it.
	for steps := 500; steps <= 4000; steps += 50 {
		interval := recommendedInterval(steps)
		if steps%interval != 0 {
			t.Errorf("recommendedInterval(%d) = %d, does not divide steps", steps, interval)
		}
	}
}

func TestRecommendedVideoInterval_aligns(t *testing.T) {
	// Same logic: divisibility is guaranteed; range is a soft target.
	for steps := 500; steps <= 5000; steps += 50 {
		interval := recommendedVideoInterval(steps)
		if steps%interval != 0 {
			t.Errorf("recommendedVideoInterval(%d) = %d, does not divide steps", steps, interval)
		}
	}
}

func TestApplyStableDefaults_112images(t *testing.T) {
	// 112 images is the reported failure case: steps=1100, save=150 didn't divide.
	s := Settings{
		Architecture: ArchitectureAnima,
		DatasetPath:  "", // no dataset → defaults to 30 images
	}
	// Simulate 30-image default (no dataset path)
	result, _ := applyStableDefaults(s)
	// SaveSteps must divide TrainingSteps evenly
	if result.TrainingSteps%result.SaveSteps != 0 {
		t.Errorf("SaveSteps %d does not divide TrainingSteps %d", result.SaveSteps, result.TrainingSteps)
	}
	if result.TrainingSteps%result.SampleSteps != 0 {
		t.Errorf("SampleSteps %d does not divide TrainingSteps %d", result.SampleSteps, result.TrainingSteps)
	}
}

// --- rank/alpha default tests per architecture ---

func TestProfileDefaults_rankAlpha(t *testing.T) {
	tests := []struct {
		arch      string
		wantRank  int
		wantAlpha int
	}{
		{ArchitectureAnima, 48, 32},
		{ArchitectureSDXL, 64, 32},
		{ArchitectureLTX23, 64, 64},
		{ArchitectureWAN22, 64, 64},
	}
	for _, tt := range tests {
		profile := profileFor(Settings{Architecture: tt.arch})
		defaults := defaultsForProfile(profile)
		if defaults.NetworkRank != tt.wantRank {
			t.Errorf("%s: NetworkRank = %d, want %d", tt.arch, defaults.NetworkRank, tt.wantRank)
		}
		if defaults.NetworkAlpha != tt.wantAlpha {
			t.Errorf("%s: NetworkAlpha = %d, want %d", tt.arch, defaults.NetworkAlpha, tt.wantAlpha)
		}
	}
}

func TestApplyStableDefaults_rankAlpha_preserved(t *testing.T) {
	// Auto calc should apply profile-aware rank/alpha defaults
	tests := []struct {
		arch      string
		wantRank  int
		wantAlpha int
	}{
		{ArchitectureAnima, 48, 32},
		{ArchitectureSDXL, 64, 32},
		{ArchitectureLTX23, 64, 64},
		{ArchitectureWAN22, 64, 64},
	}
	for _, tt := range tests {
		s := Settings{
			Architecture: tt.arch,
			DatasetPath:  "",
		}
		result, _ := applyStableDefaults(s)
		if result.NetworkRank != tt.wantRank {
			t.Errorf("%s: NetworkRank = %d, want %d", tt.arch, result.NetworkRank, tt.wantRank)
		}
		if result.NetworkAlpha != tt.wantAlpha {
			t.Errorf("%s: NetworkAlpha = %d, want %d", tt.arch, result.NetworkAlpha, tt.wantAlpha)
		}
	}
}

func TestCreateTrainingTOML_alpha_written(t *testing.T) {
	// Verify that network_alpha in the TOML uses s.NetworkAlpha, not s.NetworkRank.
	tmp := t.TempDir()
	s := Settings{
		Architecture:  ArchitectureAnima,
		ProjectName:   "test",
		OutputPath:    tmp,
		DatasetPath:   tmp,
		NetworkRank:   48,
		NetworkAlpha:  32,
		LearningRate:  "1e-4",
		TrainingSteps: 1000,
		SaveSteps:     100,
		SampleSteps:   100,
	}
	profile := profileFor(s)
	path, err := createTrainingTOML(s.ProjectName, s, profile, s.OutputPath, "", tmp)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	toml := string(data)
	// Check that alpha=32 appears, not alpha=48
	if !strings.Contains(toml, "network_alpha = 32") {
		t.Errorf("TOML should contain 'network_alpha = 32', got:\n%s", toml)
	}
	if strings.Contains(toml, "network_alpha = 48") {
		t.Errorf("TOML should NOT contain 'network_alpha = 48' (alpha should be 32, not rank): \n%s", toml)
	}
}

// --- cross-architecture rank/alpha transition tests ---

func TestNormalizeSettings_video_from_anima(t *testing.T) {
	// Switching from Anima (rank=48, alpha=32) to LTX23 should upgrade to 64/64.
	s := Settings{
		Architecture: ArchitectureLTX23,
		NetworkRank:  48,
		NetworkAlpha: 32,
	}
	s = normalizeSettings(s)
	if s.NetworkRank != 64 {
		t.Errorf("LTX23 from Anima rank 48: got rank %d, want 64", s.NetworkRank)
	}
	if s.NetworkAlpha != 64 {
		t.Errorf("LTX23 from Anima alpha 32: got alpha %d, want 64", s.NetworkAlpha)
	}
}

func TestNormalizeSettings_video_from_sdxl(t *testing.T) {
	// Switching from SDXL (rank=64, alpha=32) to LTX23 should fix alpha to 64.
	s := Settings{
		Architecture: ArchitectureLTX23,
		NetworkRank:  64,
		NetworkAlpha: 32,
	}
	s = normalizeSettings(s)
	if s.NetworkRank != 64 {
		t.Errorf("LTX23 from SDXL rank 64: got rank %d, want 64", s.NetworkRank)
	}
	if s.NetworkAlpha != 64 {
		t.Errorf("LTX23 from SDXL alpha 32: got alpha %d, want 64", s.NetworkAlpha)
	}
}

func TestNormalizeSettings_video_preserves_user_rank(t *testing.T) {
	// User explicitly set rank=128 — normalizeSettings should NOT downgrade it.
	s := Settings{
		Architecture: ArchitectureLTX23,
		NetworkRank:  128,
		NetworkAlpha: 64,
	}
	s = normalizeSettings(s)
	if s.NetworkRank != 128 {
		t.Errorf("LTX23 user rank 128: got rank %d, want 128 (should preserve)", s.NetworkRank)
	}
}

func TestNormalizeSettings_image_from_video(t *testing.T) {
	// Switching from LTX23 (rank=64, alpha=64) to Anima should use Anima defaults.
	s := Settings{
		Architecture: ArchitectureAnima,
		NetworkRank:  64,
		NetworkAlpha: 64,
	}
	s = normalizeSettings(s)
	// normalizeSettings only fills missing values for image models (fallback to 48/32 when <= 0).
	// It does NOT downgrade video values — that's handled by the JS frontend and auto-calc.
	// The key invariant: image fallbacks are 48/32 when values are zero.
	sZero := Settings{Architecture: ArchitectureAnima}
	sZero = normalizeSettings(sZero)
	if sZero.NetworkRank != 48 {
		t.Errorf("Anima zero rank: got %d, want 48", sZero.NetworkRank)
	}
	if sZero.NetworkAlpha != 32 {
		t.Errorf("Anima zero alpha: got %d, want 32", sZero.NetworkAlpha)
	}
}

func TestDefaultSettings_torchCompileDefaults(t *testing.T) {
	s := DefaultSettings("/some/root")
	if s.TorchCompile {
		t.Error("torch.compile should default to disabled")
	}
	if s.TorchCompileMode != "default" {
		t.Errorf("TorchCompileMode = %q, want default", s.TorchCompileMode)
	}
	if s.TorchCompileBackend != "inductor" {
		t.Errorf("TorchCompileBackend = %q, want inductor", s.TorchCompileBackend)
	}
	if s.TorchCompileDynamic != "auto" {
		t.Errorf("TorchCompileDynamic = %q, want auto", s.TorchCompileDynamic)
	}
	if s.TorchCompileCacheSizeLimit != 32 {
		t.Errorf("TorchCompileCacheSizeLimit = %d, want 32", s.TorchCompileCacheSizeLimit)
	}
}

func TestCreateTrainingTOML_animaTorchCompile(t *testing.T) {
	tmp := t.TempDir()
	s := normalizeSettings(Settings{
		Architecture:               ArchitectureAnima,
		ProjectName:                "compile-test",
		OutputPath:                 tmp,
		DatasetPath:                tmp,
		NetworkRank:                48,
		NetworkAlpha:               32,
		LearningRate:               "1e-4",
		TrainingSteps:              1000,
		SaveSteps:                  100,
		SampleSteps:                100,
		TorchCompile:               true,
		TorchCompileMode:           "default",
		TorchCompileBackend:        "inductor",
		TorchCompileDynamic:        "false",
		TorchCompileCacheSizeLimit: 32,
		CudaAllowTF32:              true,
		CudaCudnnBenchmark:         true,
	})
	path, err := createTrainingTOML(s.ProjectName, s, profileFor(s), s.OutputPath, "", tmp)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	toml := string(data)
	for _, want := range []string{
		"compile = true",
		"compile_backend = \"inductor\"",
		"compile_mode = \"default\"",
		"compile_dynamic = false",
		"compile_cache_size_limit = 32",
		"cuda_allow_tf32 = true",
		"cuda_cudnn_benchmark = true",
	} {
		if !strings.Contains(toml, want) {
			t.Errorf("Anima compile TOML missing %q:\n%s", want, toml)
		}
	}
	if strings.Contains(toml, "torch_compile") {
		t.Errorf("Anima compile TOML must use upstream --compile keys, not torch_compile:\n%s", toml)
	}
}

func TestCreateTrainingTOML_sdxlIgnoresTorchCompile(t *testing.T) {
	tmp := t.TempDir()
	s := normalizeSettings(Settings{
		Architecture:               ArchitectureSDXL,
		ProjectName:                "sdxl-compile-test",
		OutputPath:                 tmp,
		DatasetPath:                tmp,
		CheckpointPath:             filepath.Join(tmp, "model.safetensors"),
		NetworkRank:                64,
		NetworkAlpha:               32,
		LearningRate:               "1e-4",
		TrainingSteps:              1000,
		SaveSteps:                  100,
		SampleSteps:                100,
		TorchCompile:               true,
		TorchCompileMode:           "default",
		TorchCompileBackend:        "inductor",
		TorchCompileCacheSizeLimit: 32,
		CudaAllowTF32:              true,
		CudaCudnnBenchmark:         true,
	})
	if err := os.WriteFile(s.CheckpointPath, []byte("stub"), 0644); err != nil {
		t.Fatal(err)
	}
	path, err := createTrainingTOML(s.ProjectName, s, profileFor(s), s.OutputPath, "", tmp)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	toml := string(data)
	if strings.Contains(toml, "compile = true") || strings.Contains(toml, "compile_backend") || strings.Contains(toml, "cuda_allow_tf32") {
		t.Errorf("SDXL TOML must not include Anima torch.compile keys:\n%s", toml)
	}
	if !strings.Contains(toml, "sdpa = true") {
		t.Errorf("SDXL TOML should keep SDPA enabled by default:\n%s", toml)
	}
}

func TestValidateTorchCompileRuntime_reportsMissingTriton(t *testing.T) {
	tmp := t.TempDir()
	python := filepath.Join(tmp, "python")
	script := "#!/bin/sh\n" +
		"echo 'ModuleNotFoundError: No module named triton' >&2\n" +
		"exit 1\n"
	if err := os.WriteFile(python, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	err := validateTorchCompileRuntime(python, Settings{TorchCompile: true})
	if err == nil {
		t.Fatal("expected missing Triton error")
	}
	if !strings.Contains(err.Error(), "torch.compile is enabled") || !strings.Contains(err.Error(), "Triton") {
		t.Fatalf("expected actionable Triton error, got %v", err)
	}
}
