package trainer

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

type PathEntry struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	IsDir bool   `json:"isDir"`
}

type PathListResponse struct {
	Path    string      `json:"path"`
	Parent  string      `json:"parent"`
	Entries []PathEntry `json:"entries"`
	Roots   []PathEntry `json:"roots"`
}

func listPath(pathValue, mode string) PathListResponse {
	pathValue = strings.TrimSpace(pathValue)
	if pathValue == "" {
		pathValue = defaultBrowsePath()
	}
	if info, err := os.Stat(pathValue); err == nil && !info.IsDir() {
		pathValue = filepath.Dir(pathValue)
	} else if err != nil {
		pathValue = nearestExistingDir(pathValue)
	}
	abs, err := filepath.Abs(pathValue)
	if err == nil {
		pathValue = abs
	}

	entries, _ := os.ReadDir(pathValue)
	out := make([]PathEntry, 0, len(entries))
	for _, entry := range entries {
		isDir := entry.IsDir()
		if !isDir && mode == "directory" {
			continue
		}
		if !isDir && mode == "model" && !isModelFile(entry.Name()) {
			continue
		}
		out = append(out, PathEntry{
			Name:  entry.Name(),
			Path:  filepath.Join(pathValue, entry.Name()),
			IsDir: isDir,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].IsDir != out[j].IsDir {
			return out[i].IsDir
		}
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})

	return PathListResponse{
		Path:    pathValue,
		Parent:  filepath.Dir(pathValue),
		Entries: out,
		Roots:   browseRoots(),
	}
}

func nearestExistingDir(pathValue string) string {
	if pathValue == "" {
		return defaultBrowsePath()
	}
	for {
		if info, err := os.Stat(pathValue); err == nil && info.IsDir() {
			return pathValue
		}
		parent := filepath.Dir(pathValue)
		if parent == pathValue || parent == "." {
			return defaultBrowsePath()
		}
		pathValue = parent
	}
}

func defaultBrowsePath() string {
	if home, err := os.UserHomeDir(); err == nil {
		return home
	}
	cwd, err := os.Getwd()
	if err == nil {
		return cwd
	}
	return string(filepath.Separator)
}

func browseRoots() []PathEntry {
	if runtime.GOOS != "windows" {
		return []PathEntry{{Name: "/", Path: "/", IsDir: true}}
	}
	var roots []PathEntry
	for drive := 'A'; drive <= 'Z'; drive++ {
		path := string(drive) + `:\`
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			roots = append(roots, PathEntry{Name: path, Path: path, IsDir: true})
		}
	}
	return roots
}

func isModelFile(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx":
		return true
	default:
		return false
	}
}
