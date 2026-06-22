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
