package process

import (
	"os"
	"path/filepath"
	"runtime"
)

// FileExists returns true if the path exists and is a regular file (not a dir).
// Empty files are considered to exist.
func FileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// FileExistsNonEmpty returns true if the path exists, is a regular file, and
// has size greater than zero. Use this for model weights, downloads, etc.
func FileExistsNonEmpty(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Size() > 0
}

// PythonExecutable returns the path to the embedded Python interpreter,
// or "" if none is found in the given root directory.
func PythonExecutable(root string) string {
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
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}
