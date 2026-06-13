package trainer

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

var projectNameRe = regexp.MustCompile(`[^a-zA-Z0-9]+`)

func joinSlash(elem ...string) string {
	return filepath.ToSlash(filepath.Join(elem...))
}

func jsonContains(data []byte, key string) bool {
	quoted := []byte(strconv.Quote(key))
	return bytes.Contains(data, quoted)
}

func sanitizeProjectName(trigger string) string {
	name := strings.Trim(projectNameRe.ReplaceAllString(strings.TrimSpace(trigger), "_"), "_")
	if name == "" {
		return "untitled"
	}
	return name
}

func validImageExt(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg", ".webp", ".bmp":
		return true
	default:
		return false
	}
}

func outputProject(root, trigger string) string {
	project := sanitizeProjectName(trigger)
	return filepath.Join(root, "training", "output", project)
}

func listLatestImages(dir string) []ImageItem {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	type candidate struct {
		item ImageItem
		mod  int64
	}
	var files []candidate
	for _, entry := range entries {
		if entry.IsDir() || !validImageExt(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		files = append(files, candidate{
			item: ImageItem{
				Src:  "/samples/" + filepath.ToSlash(filepath.Base(filepath.Dir(dir))) + "/" + entry.Name(),
				Name: entry.Name(),
			},
			mod: info.ModTime().UnixNano(),
		})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].mod > files[j].mod })
	images := make([]ImageItem, 0, len(files))
	for _, file := range files {
		images = append(images, file.item)
	}
	return images
}

func analyzeDatasetResolution(datasetPath string) (int, int) {
	entries, err := os.ReadDir(datasetPath)
	if err != nil {
		return 512, 768
	}
	maxArea := 0
	maxSide := 0
	for _, entry := range entries {
		if entry.IsDir() || !validImageExt(entry.Name()) {
			continue
		}
		file, err := os.Open(filepath.Join(datasetPath, entry.Name()))
		if err != nil {
			continue
		}
		cfg, _, err := image.DecodeConfig(file)
		_ = file.Close()
		if err != nil {
			continue
		}
		area := cfg.Width * cfg.Height
		if area > maxArea {
			maxArea = area
		}
		if cfg.Width > maxSide {
			maxSide = cfg.Width
		}
		if cfg.Height > maxSide {
			maxSide = cfg.Height
		}
	}
	if maxArea == 0 {
		return 512, 768
	}
	return ceilTo64(sqrtInt(maxArea)), ceilTo64(maxSide)
}

func ceilTo64(v int) int {
	if v <= 0 {
		return 64
	}
	return ((v + 63) / 64) * 64
}

func sqrtInt(v int) int {
	x := 1
	for x*x < v {
		x++
	}
	return x
}

func validateSettings(s Settings) []string {
	var errs []string
	if !fileExists(s.DiTPath) {
		errs = append(errs, "DiT file not found: "+s.DiTPath)
	}
	if !fileExists(s.QwenPath) {
		errs = append(errs, "Qwen3 file not found: "+s.QwenPath)
	}
	if !fileExists(s.VAEPath) {
		errs = append(errs, "VAE file not found: "+s.VAEPath)
	}
	if !dirExists(s.DatasetPath) {
		errs = append(errs, "Dataset path not found: "+s.DatasetPath)
		return errs
	}
	entries, err := os.ReadDir(s.DatasetPath)
	if err != nil {
		errs = append(errs, "Dataset cannot be read: "+err.Error())
		return errs
	}
	var images []string
	var missingCaptions []string
	var oversized []string
	for _, entry := range entries {
		if entry.IsDir() || !validImageExt(entry.Name()) {
			continue
		}
		images = append(images, entry.Name())
		stem := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		if !fileExists(filepath.Join(s.DatasetPath, stem+".txt")) {
			missingCaptions = append(missingCaptions, entry.Name())
		}
		path := filepath.Join(s.DatasetPath, entry.Name())
		file, openErr := os.Open(path)
		if openErr == nil {
			cfg, _, decodeErr := image.DecodeConfig(file)
			_ = file.Close()
			if decodeErr == nil && (cfg.Width >= 2048 || cfg.Height >= 2048) {
				oversized = append(oversized, entry.Name())
			}
		}
	}
	if len(images) == 0 {
		errs = append(errs, "No valid images found in the dataset path.")
	}
	if len(missingCaptions) > 0 {
		errs = append(errs, fmt.Sprintf("%d images are missing .txt captions. Run auto-captioning first.", len(missingCaptions)))
	}
	if len(oversized) > 0 {
		errs = append(errs, fmt.Sprintf("%d images are >= 2048px. Run smart bucketing first.", len(oversized)))
	}
	if s.ResumeEnabled && !s.AutoResume && strings.TrimSpace(s.ResumePath) != "" && !dirExists(strings.TrimSpace(s.ResumePath)) {
		errs = append(errs, "Resume state folder not found: "+s.ResumePath)
	}
	return errs
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func pythonExecutable(root string) string {
	var candidates []string
	if runtime.GOOS == "windows" {
		candidates = []string{
			filepath.Join(root, "python_embeded", "windows", "python.exe"),
			filepath.Join(root, "python_embeded", "python.exe"),
		}
	} else {
		candidates = []string{
			filepath.Join(root, "python_embeded", "linux", "bin", "python3"),
			filepath.Join(root, "python_embeded", "linux", "bin", "python"),
			filepath.Join(root, "python_embeded", "bin", "python3"),
			filepath.Join(root, "python_embeded", "bin", "python"),
			filepath.Join(root, "python_embeded", "python"),
		}
	}
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate
		}
	}
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func validatePythonRuntime(python string) error {
	cmd := exec.Command(python, "-c", "import accelerate.commands.launch")
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg != "" {
			return fmt.Errorf("runtime incomplete: %s", msg)
		}
		return fmt.Errorf("runtime incomplete: %w", err)
	}
	return nil
}

func findLastStateDir(outputDir string) string {
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		return ""
	}
	type stateDir struct {
		path string
		mod  time.Time
	}
	var states []stateDir
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasSuffix(entry.Name(), "-state") {
			continue
		}
		path := filepath.Join(outputDir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue
		}
		states = append(states, stateDir{path: path, mod: info.ModTime()})
	}
	sort.Slice(states, func(i, j int) bool { return states[i].mod.After(states[j].mod) })
	if len(states) == 0 {
		return ""
	}
	return states[0].path
}
