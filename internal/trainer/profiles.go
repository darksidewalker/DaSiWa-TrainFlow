package trainer

import (
	"fmt"
	"path/filepath"
	"strings"
)

const (
	ArchitectureAnima = "anima"
	ArchitectureSDXL  = "sdxl"
)

type trainingProfile struct {
	Architecture      string
	Label             string
	Script            string
	InferenceScript   string
	RequiredPathNames []string
	BucketStep        int
	ResizeDivisor     int
}

func profileFor(s Settings) trainingProfile {
	switch normalizeArchitecture(s.Architecture) {
	case ArchitectureSDXL:
		return trainingProfile{
			Architecture:      ArchitectureSDXL,
			Label:             "SDXL / Pony / Illustrious",
			Script:            "sdxl_train_network.py",
			InferenceScript:   "sdxl_minimal_inference.py",
			RequiredPathNames: []string{"checkpoint_path"},
			BucketStep:        32,
			ResizeDivisor:     32,
		}
	default:
		return trainingProfile{
			Architecture:      ArchitectureAnima,
			Label:             "Anima",
			Script:            "anima_train_network.py",
			InferenceScript:   "anima_minimal_inference.py",
			RequiredPathNames: []string{"dit_path", "qwen_path", "vae_path"},
			BucketStep:        64,
			ResizeDivisor:     64,
		}
	}
}

func normalizeArchitecture(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ArchitectureSDXL:
		return ArchitectureSDXL
	default:
		return ArchitectureAnima
	}
}

func normalizeSettings(s Settings) Settings {
	s.Architecture = normalizeArchitecture(s.Architecture)
	if strings.TrimSpace(s.ProjectName) == "" && strings.TrimSpace(s.TriggerWord) != "" {
		s.ProjectName = strings.TrimSpace(s.TriggerWord)
	}
	if s.NetworkRank <= 0 {
		s.NetworkRank = 32
	}
	if strings.TrimSpace(s.LearningRate) == "" {
		if s.Architecture == ArchitectureSDXL {
			s.LearningRate = "1e-4"
		} else {
			s.LearningRate = "1e-4"
		}
	}
	if strings.EqualFold(strings.TrimSpace(s.Optimizer), "Prodigy") {
		s.LearningRate = "1.0"
	} else if strings.TrimSpace(s.LearningRate) == "1.0" {
		s.LearningRate = "1e-4"
	}
	if strings.TrimSpace(s.UNetLR) == "" {
		s.UNetLR = "1e-4"
	}
	if strings.TrimSpace(s.TextEncoderLR1) == "" {
		s.TextEncoderLR1 = "1e-5"
	}
	if strings.TrimSpace(s.TextEncoderLR2) == "" {
		s.TextEncoderLR2 = "1e-5"
	}
	return s
}

func (p trainingProfile) trainingScript(root string) string {
	return filepath.Join(root, "training", "sd-scripts", p.Script)
}

func (p trainingProfile) validateModelPaths(s Settings) []string {
	var errs []string
	switch p.Architecture {
	case ArchitectureSDXL:
		if !fileExists(s.CheckpointPath) {
			errs = append(errs, "SDXL checkpoint file not found: "+s.CheckpointPath)
		}
	case ArchitectureAnima:
		if !fileExists(s.DiTPath) {
			errs = append(errs, "DiT file not found: "+s.DiTPath)
		}
		if !fileExists(s.QwenPath) {
			errs = append(errs, "Qwen3 file not found: "+s.QwenPath)
		}
		if !fileExists(s.VAEPath) {
			errs = append(errs, "VAE file not found: "+s.VAEPath)
		}
	default:
		errs = append(errs, fmt.Sprintf("Unknown training architecture: %s", p.Architecture))
	}
	return errs
}

func (p trainingProfile) supportsManagedModelCheck() bool {
	return p.Architecture == ArchitectureAnima
}
