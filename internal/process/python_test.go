package process

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFileExists(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a file
	filePath := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(filePath, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		path string
		want bool
	}{
		{filePath, true},
		{filepath.Join(tmpDir, "nonexistent.txt"), false},
		{tmpDir, false}, // directory, not a file
		{"", false},
	}

	for _, tt := range tests {
		got := FileExists(tt.path)
		if got != tt.want {
			t.Errorf("FileExists(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestFileExistsNonEmpty(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a non-empty file
	nonEmptyPath := filepath.Join(tmpDir, "nonempty.txt")
	if err := os.WriteFile(nonEmptyPath, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create an empty file
	emptyPath := filepath.Join(tmpDir, "empty.txt")
	if err := os.WriteFile(emptyPath, []byte{}, 0644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		path string
		want bool
	}{
		{nonEmptyPath, true},
		{emptyPath, false},
		{filepath.Join(tmpDir, "nonexistent.txt"), false},
		{tmpDir, false}, // directory
	}

	for _, tt := range tests {
		got := FileExistsNonEmpty(tt.path)
		if got != tt.want {
			t.Errorf("FileExistsNonEmpty(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestPythonExecutable(t *testing.T) {
	root := t.TempDir()

	// Test with no python found
	got := PythonExecutable(root)
	if got == "" {
		t.Error("expected fallback to python3")
	}

	// Create a fake python executable
	pyPath := filepath.Join(root, "python_embeded", "linux", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(pyPath), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pyPath, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}

	got = PythonExecutable(root)
	if got != pyPath {
		t.Errorf("PythonExecutable(%q) = %q, want %q", root, got, pyPath)
	}
}
