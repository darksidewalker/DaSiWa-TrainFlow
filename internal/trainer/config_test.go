package trainer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCreateTrainingTOMLAttentionMode(t *testing.T) {
	dir := t.TempDir()
	settings := DefaultSettings(dir)

	path, err := createTrainingTOML("plain", settings, dir, filepath.Join(dir, "prompts.txt"), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "attn_mode = \"torch\"") {
		t.Fatalf("expected torch attention mode, got:\n%s", data)
	}

	settings.FlashAttention = true
	path, err = createTrainingTOML("flash", settings, dir, filepath.Join(dir, "prompts.txt"), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "attn_mode = \"flash\"") {
		t.Fatalf("expected flash attention mode, got:\n%s", data)
	}
}
