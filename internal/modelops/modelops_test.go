package modelops

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRequiredFiles(t *testing.T) {
	root := t.TempDir()
	files := RequiredFiles(root)
	if len(files) != 3 {
		t.Errorf("expected 3 required files, got %d", len(files))
	}
	for _, f := range files {
		if f.Optional {
			t.Errorf("required file %q marked as optional", f.Name)
		}
		if f.Key == "" {
			t.Errorf("required file %q missing key", f.Name)
		}
		if !filepath.IsAbs(f.Path) {
			t.Errorf("required file %q has non-absolute path: %s", f.Name, f.Path)
		}
	}
}

func TestOptionalFiles(t *testing.T) {
	root := t.TempDir()
	files := OptionalFiles(root)
	if len(files) != 5 {
		t.Errorf("expected 5 optional files, got %d", len(files))
	}
	for _, f := range files {
		if !f.Optional {
			t.Errorf("optional file %q not marked as optional", f.Name)
		}
	}
}

func TestAllFiles(t *testing.T) {
	root := t.TempDir()
	all := AllFiles(root)
	req := RequiredFiles(root)
	opt := OptionalFiles(root)
	if len(all) != len(req)+len(opt) {
		t.Errorf("AllFiles returned %d, expected %d", len(all), len(req)+len(opt))
	}
}

func TestCheck(t *testing.T) {
	root := t.TempDir()
	status := Check(root)
	if status.Ready {
		t.Error("expected models to be missing in temp dir")
	}
	if status.Missing != 3 {
		t.Errorf("expected 3 missing, got %d", status.Missing)
	}
}

func TestCheckWithOverrides(t *testing.T) {
	root := t.TempDir()
	// Create fake model files
	ditPath := filepath.Join(root, "fake-dit.safetensors")
	qwenPath := filepath.Join(root, "fake-qwen.safetensors")
	vaePath := filepath.Join(root, "fake-vae.safetensors")
	for _, p := range []string{ditPath, qwenPath, vaePath} {
		if err := os.WriteFile(p, []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	overrides := map[string]string{
		"dit_path":  ditPath,
		"qwen_path": qwenPath,
		"vae_path":  vaePath,
	}

	status := CheckWithOverrides(root, overrides)
	if !status.Ready {
		t.Error("expected models to be ready with overrides")
	}
	if status.Missing != 0 {
		t.Errorf("expected 0 missing, got %d", status.Missing)
	}
}

func TestCheckMessage(t *testing.T) {
	root := t.TempDir()
	status := Check(root)
	if status.Message == "" {
		t.Error("expected non-empty message")
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		value int64
		want  string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KiB"},
		{1048576, "1.0 MiB"},
		{1073741824, "1.0 GiB"},
	}
	for _, tt := range tests {
		got := formatBytes(tt.value)
		if got != tt.want {
			t.Errorf("formatBytes(%d) = %q, want %q", tt.value, got, tt.want)
		}
	}
}
