package modelops

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type Logger func(string)

type ModelFile struct {
	Name     string `json:"name"`
	Key      string `json:"key"`
	Path     string `json:"path"`
	Found    string `json:"found"`
	URL      string `json:"url"`
	Size     string `json:"size"`
	Optional bool   `json:"optional"`
	OK       bool   `json:"ok"`
}

type Status struct {
	Ready           bool        `json:"ready"`
	OptionalReady   bool        `json:"optional_ready"`
	Missing         int         `json:"missing"`
	OptionalMissing int         `json:"optional_missing"`
	Files           []ModelFile `json:"files"`
	Message         string      `json:"message"`
}

func RequiredFiles(root string) []ModelFile {
	return []ModelFile{
		{
			Name: "Anima Base v1.0 DiT",
			Key:  "dit_path",
			Path: filepath.Join(root, "models", "anima", "dit", "anima-base-v1.0.safetensors"),
			URL:  "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
			Size: "18.2 GB",
		},
		{
			Name: "Qwen3 text encoder",
			Key:  "qwen_path",
			Path: filepath.Join(root, "models", "anima", "text_encoder", "qwen_3_06b_base.safetensors"),
			URL:  "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
		},
		{
			Name: "Qwen Image VAE",
			Key:  "vae_path",
			Path: filepath.Join(root, "models", "anima", "vae", "qwen_image_vae.safetensors"),
			URL:  "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
			Size: "254 MB",
		},
	}
}

func OptionalFiles(root string) []ModelFile {
	taggerDir := filepath.Join(root, "models", "wd-eva02-large-tagger-v3")
	return []ModelFile{
		{
			Name:     "WD EVA02 tagger ONNX",
			Path:     filepath.Join(taggerDir, "model.onnx"),
			URL:      "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx",
			Size:     "1.26 GB",
			Optional: true,
		},
		{
			Name:     "WD EVA02 selected tags",
			Path:     filepath.Join(taggerDir, "selected_tags.csv"),
			URL:      "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv",
			Size:     "308 KB",
			Optional: true,
		},
		{
			Name:     "WD EVA02 model config",
			Path:     filepath.Join(taggerDir, "config.json"),
			URL:      "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/config.json",
			Optional: true,
		},
		{
			Name:     "WD EVA02 JAX config",
			Path:     filepath.Join(taggerDir, "sw_jax_cv_config.json"),
			URL:      "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/sw_jax_cv_config.json",
			Optional: true,
		},
		{
			Name:     "U2Net background-removal model",
			Path:     filepath.Join(root, "models", "u2net", "u2net.onnx"),
			URL:      "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
			Optional: true,
		},
	}
}

func AllFiles(root string) []ModelFile {
	files := RequiredFiles(root)
	files = append(files, OptionalFiles(root)...)
	return files
}

func Check(root string) Status {
	return CheckWithOverrides(root, nil)
}

func CheckWithOverrides(root string, overrides map[string]string) Status {
	files := AllFiles(root)
	missing := 0
	optionalMissing := 0
	for i := range files {
		files[i].OK = fileExists(files[i].Path)
		if files[i].OK {
			files[i].Found = files[i].Path
		}
		if !files[i].Optional && files[i].Key != "" && overrides != nil {
			if override := overrides[files[i].Key]; override != "" && fileExists(override) {
				files[i].OK = true
				files[i].Found = override
			}
		}
		if files[i].OK {
			continue
		}
		if files[i].Optional {
			optionalMissing++
		} else {
			missing++
		}
	}
	message := "Models ready"
	if missing > 0 {
		message = fmt.Sprintf("%d model file(s) missing", missing)
	}
	if missing == 0 && optionalMissing > 0 {
		message = fmt.Sprintf("Optional prep models missing: %d", optionalMissing)
	}
	return Status{
		Ready:           missing == 0,
		OptionalReady:   optionalMissing == 0,
		Missing:         missing,
		OptionalMissing: optionalMissing,
		Files:           files,
		Message:         message,
	}
}

func DownloadRequired(root string, log Logger) error {
	for _, file := range RequiredFiles(root) {
		if fileExists(file.Path) {
			log("Already present: " + file.Path)
			continue
		}
		if err := download(log, file.URL, file.Path); err != nil {
			return err
		}
	}
	return nil
}

func DownloadOptional(root string, log Logger) error {
	for _, file := range OptionalFiles(root) {
		if fileExists(file.Path) {
			log("Already present: " + file.Path)
			continue
		}
		if err := download(log, file.URL, file.Path); err != nil {
			return err
		}
	}
	return nil
}

func download(log Logger, url, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tempPath := path + ".incomplete"
	_ = os.Remove(tempPath)

	log("Downloading " + filepath.Base(path))
	log("From: " + url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("download failed: %s", resp.Status)
	}

	out, err := os.Create(tempPath)
	if err != nil {
		return err
	}
	defer out.Close()

	var written int64
	buf := make([]byte, 1024*1024)
	nextLog := time.Now().Add(10 * time.Second)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				return err
			}
			written += int64(n)
			if time.Now().After(nextLog) {
				log(fmt.Sprintf("Downloaded %s for %s", formatBytes(written), filepath.Base(path)))
				nextLog = time.Now().Add(10 * time.Second)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if err := out.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return err
	}
	log("Saved: " + path)
	return nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Size() > 0
}

func formatBytes(value int64) string {
	const unit = 1024
	if value < unit {
		return fmt.Sprintf("%d B", value)
	}
	div, exp := int64(unit), 0
	for n := value / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(value)/float64(div), "KMGTPE"[exp])
}
