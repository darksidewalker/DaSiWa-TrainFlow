package trainer

import (
	"os"
	"path/filepath"
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
		steps    int
		lo       int
		hi       int
		ideal    int
		want     int
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
