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
	krea := Krea2Files(root)
	if len(all) != len(req)+len(krea)+len(opt) {
		t.Errorf("AllFiles returned %d, expected %d", len(all), len(req)+len(krea)+len(opt))
	}
}

func TestCheck(t *testing.T) {
	root := t.TempDir()
	status := Check(root)
	if status.Ready {
		t.Error("expected models to be missing in temp dir")
	}
	if status.Missing != 6 {
		t.Errorf("expected 6 missing, got %d", status.Missing)
	}
}

func TestKrea2Files(t *testing.T) {
	root := t.TempDir()
	files := Krea2Files(root)
	if len(files) != 3 {
		t.Fatalf("expected 3 Krea2 files, got %d", len(files))
	}
	wantCategories := map[string]bool{"Text Encoder": false, "Model / DiT": false, "VAE": false}
	for _, f := range files {
		if f.Arch != "krea2" {
			t.Errorf("expected krea2 arch, got %q", f.Arch)
		}
		if f.URL == "" || f.Path == "" {
			t.Errorf("Krea2 file %q missing URL/path", f.Name)
		}
		wantCategories[f.Category] = true
	}
	for category, seen := range wantCategories {
		if !seen {
			t.Errorf("missing category %q", category)
		}
	}
}

func TestDownloadSelectedExistingFile(t *testing.T) {
	root := t.TempDir()
	file := Krea2Files(root)[0]
	if err := os.MkdirAll(filepath.Dir(file.Path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file.Path, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}
	var logs []string
	err := DownloadSelected(root, []string{selectionKey(file)}, func(line string) {
		logs = append(logs, line)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(logs) == 0 || logs[0] == "" {
		t.Fatal("expected already-present log")
	}
}

func TestDownloadSelectedRejectsEmptySelection(t *testing.T) {
	err := DownloadSelected(t.TempDir(), nil, func(string) {})
	if err == nil {
		t.Fatal("expected empty selection error")
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
