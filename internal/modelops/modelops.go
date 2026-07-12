package modelops

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"trainflow/internal/process"
)

type Logger func(string)

type ModelFile struct {
	Name     string `json:"name"`
	Key      string `json:"key"`
	Arch     string `json:"arch,omitempty"`
	Category string `json:"category,omitempty"`
	Path     string `json:"path"`
	Found    string `json:"found"`
	URL      string `json:"url"`
	Size     string `json:"size"`
	Hash     string `json:"hash,omitempty"`
	Optional bool   `json:"optional"`
	OK       bool   `json:"ok"`
}

type ModelGroup struct {
	Architecture string      `json:"architecture"`
	Label        string      `json:"label"`
	Files        []ModelFile `json:"files"`
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
			Name:     "Anima Base v1.0 DiT",
			Key:      "dit_path",
			Arch:     "anima",
			Category: "Model / DiT",
			Path:     filepath.Join(root, "models", "anima", "dit", "anima-base-v1.0.safetensors"),
			URL:      "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
			Size:     "18.2 GB",
		},
		{
			Name:     "Qwen3 text encoder",
			Key:      "qwen_path",
			Arch:     "anima",
			Category: "Text Encoder",
			Path:     filepath.Join(root, "models", "anima", "text_encoder", "qwen_3_06b_base.safetensors"),
			URL:      "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
		},
		{
			Name:     "Qwen Image VAE",
			Key:      "vae_path",
			Arch:     "anima",
			Category: "VAE",
			Path:     filepath.Join(root, "models", "anima", "vae", "qwen_image_vae.safetensors"),
			URL:      "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
			Size:     "254 MB",
		},
	}
}

func Krea2Files(root string) []ModelFile {
	return []ModelFile{
		{
			Name:     "Qwen3-VL 4B BF16 text encoder",
			Key:      "qwen_path",
			Arch:     "krea2",
			Category: "Text Encoder",
			Path:     filepath.Join(root, "models", "krea2", "text_encoder", "qwen3vl_4b_bf16.safetensors"),
			URL:      "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors",
			Size:     "BF16",
		},
		{
			Name:     "Krea 2 RAW BF16 diffusion model",
			Key:      "dit_path",
			Arch:     "krea2",
			Category: "Model / DiT",
			Path:     filepath.Join(root, "models", "krea2", "diffusion_models", "krea2_raw_bf16.safetensors"),
			URL:      "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_bf16.safetensors",
			Size:     "BF16",
		},
		{
			Name:     "Qwen Image VAE",
			Key:      "vae_path",
			Arch:     "krea2",
			Category: "VAE",
			Path:     filepath.Join(root, "models", "krea2", "vae", "qwen_image_vae.safetensors"),
			URL:      "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
			Size:     "254 MB",
		},
	}
}

func RequiredModelFiles(root string) []ModelFile {
	files := RequiredFiles(root)
	files = append(files, Krea2Files(root)...)
	return files
}

func Catalog(root string) []ModelGroup {
	return []ModelGroup{
		{Architecture: "anima", Label: "ANIMA", Files: withStatus(root, RequiredFiles(root), nil)},
		{Architecture: "krea2", Label: "Krea 2", Files: withStatus(root, Krea2Files(root), nil)},
		{Architecture: "wan22", Label: "WAN 2.2", Files: nil},
		{Architecture: "ltx23", Label: "LTX 2.3", Files: nil},
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
	files := RequiredModelFiles(root)
	files = append(files, OptionalFiles(root)...)
	return files
}

func Check(root string) Status {
	return CheckWithOverrides(root, nil)
}

func CheckWithOverrides(root string, overrides map[string]string) Status {
	files := AllFiles(root)
	files = withStatus(root, files, overrides)
	missing := 0
	optionalMissing := 0
	for _, file := range files {
		if file.OK {
			continue
		}
		if file.Optional {
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

func withStatus(root string, files []ModelFile, overrides map[string]string) []ModelFile {
	_ = root
	for i := range files {
		files[i].OK = process.FileExistsNonEmpty(files[i].Path)
		if files[i].OK {
			files[i].Found = files[i].Path
		}
		if !files[i].Optional && files[i].Key != "" && overrides != nil {
			if override := overrides[files[i].Key]; override != "" && process.FileExistsNonEmpty(override) {
				files[i].OK = true
				files[i].Found = override
			}
		}
	}
	return files
}

func DownloadRequired(root string, log Logger) error {
	for _, file := range RequiredModelFiles(root) {
		if err := downloadFileIfMissing(file, log); err != nil {
			return err
		}
	}
	return nil
}

func DownloadSelected(root string, keys []string, log Logger) error {
	selected := map[string]bool{}
	for _, key := range keys {
		selected[key] = true
	}
	if len(selected) == 0 {
		return fmt.Errorf("no models selected")
	}
	matched := 0
	for _, file := range RequiredModelFiles(root) {
		if !selected[selectionKey(file)] {
			continue
		}
		matched++
		if err := downloadFileIfMissing(file, log); err != nil {
			return err
		}
	}
	if matched == 0 {
		return fmt.Errorf("no known models selected")
	}
	return nil
}

func selectionKey(file ModelFile) string {
	return file.Arch + ":" + file.Key + ":" + filepath.Base(file.Path)
}

func downloadFileIfMissing(file ModelFile, log Logger) error {
	if process.FileExistsNonEmpty(file.Path) {
		log("Already present: " + file.Path)
		return nil
	}
	if err := download(log, file.URL, file.Path, file.Hash); err != nil {
		return err
	}
	return nil
}

func DownloadOptional(root string, log Logger) error {
	for _, file := range OptionalFiles(root) {
		if process.FileExistsNonEmpty(file.Path) {
			log("Already present: " + file.Path)
			continue
		}
		if err := download(log, file.URL, file.Path, file.Hash); err != nil {
			return err
		}
	}
	return nil
}

func download(log Logger, url, path, expectedHash string) error {
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

	hash := sha256.New()
	var written int64
	buf := make([]byte, 1024*1024)
	nextLog := time.Now().Add(10 * time.Second)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				return err
			}
			hash.Write(buf[:n])
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

	computedHash := hex.EncodeToString(hash.Sum(nil))
	log(fmt.Sprintf("SHA-256: %s", computedHash))

	if expectedHash != "" && computedHash != expectedHash {
		_ = os.Remove(tempPath)
		return fmt.Errorf("hash mismatch for %s: expected %s, got %s", filepath.Base(path), expectedHash, computedHash)
	}

	if err := os.Rename(tempPath, path); err != nil {
		return err
	}
	log("Saved: " + path)
	return nil
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
