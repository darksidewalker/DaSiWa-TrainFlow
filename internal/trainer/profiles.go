package trainer

import (
	"fmt"
	"path/filepath"
	"strings"
)

const (
	ArchitectureAnima = "anima"
	ArchitectureSDXL  = "sdxl"
	ArchitectureLTX23 = "ltx23"
	ArchitectureWAN22 = "wan22"
)

type trainingFamily string

const (
	trainingFamilySDScripts trainingFamily = "sd-scripts"
	trainingFamilyMusubi    trainingFamily = "musubi"
)

type trainingProfile struct {
	Architecture      string
	Label             string
	Family            trainingFamily
	Script            string
	InferenceScript   string
	RequiredPathNames []string
	BucketStep        int
	ResizeDivisor     int
	Video             bool
}

func profileFor(s Settings) trainingProfile {
	switch normalizeArchitecture(s.Architecture) {
	case ArchitectureSDXL:
		return trainingProfile{
			Architecture:      ArchitectureSDXL,
			Label:             "SDXL / Pony / Illustrious",
			Family:            trainingFamilySDScripts,
			Script:            "sdxl_train_network.py",
			InferenceScript:   "sdxl_minimal_inference.py",
			RequiredPathNames: []string{"checkpoint_path"},
			BucketStep:        32,
			ResizeDivisor:     32,
		}
	case ArchitectureLTX23:
		return trainingProfile{
			Architecture:      ArchitectureLTX23,
			Label:             "LTX 2.3 Video LoRA",
			Family:            trainingFamilyMusubi,
			Script:            "ltx2_train_network.py",
			RequiredPathNames: []string{"checkpoint_path", "qwen_path"},
			BucketStep:        16,
			ResizeDivisor:     16,
			Video:             true,
		}
	case ArchitectureWAN22:
		return trainingProfile{
			Architecture:      ArchitectureWAN22,
			Label:             "Wan 2.2 Video LoRA",
			Family:            trainingFamilyMusubi,
			Script:            "wan_train_network.py",
			RequiredPathNames: []string{"dit_path", "qwen_path", "vae_path"},
			BucketStep:        16,
			ResizeDivisor:     16,
			Video:             true,
		}
	default:
		return trainingProfile{
			Architecture:      ArchitectureAnima,
			Label:             "Anima",
			Family:            trainingFamilySDScripts,
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
	case ArchitectureLTX23:
		return ArchitectureLTX23
	case ArchitectureWAN22:
		return ArchitectureWAN22
	default:
		return ArchitectureAnima
	}
}

func normalizeSettings(s Settings) Settings {
	s.Architecture = normalizeArchitecture(s.Architecture)
	if strings.TrimSpace(s.ProjectName) == "" && strings.TrimSpace(s.TriggerWord) != "" {
		s.ProjectName = strings.TrimSpace(s.TriggerWord)
	}
	if s.TargetEpochs <= 0 {
		s.TargetEpochs = 6
	}
	if strings.TrimSpace(s.MixedPrecision) == "" {
		s.MixedPrecision = "bf16"
	}
	if s.NumCPUThreads <= 0 {
		s.NumCPUThreads = 8
	}
	if s.VideoWidth <= 0 {
		s.VideoWidth = 768
	}
	if s.VideoHeight <= 0 {
		s.VideoHeight = 512
	}
	if s.VideoFPS <= 0 {
		s.VideoFPS = 24
	}
	if strings.TrimSpace(s.VideoDuration) == "" {
		s.VideoDuration = "5"
	}
	if strings.TrimSpace(s.VideoTargetFrames) == "" {
		s.VideoTargetFrames = "1,65,129"
	}
	if strings.TrimSpace(s.VideoFrameExtraction) == "" {
		s.VideoFrameExtraction = "full"
	}
	if s.VideoNumRepeats <= 0 {
		s.VideoNumRepeats = 1
	}
	if strings.TrimSpace(s.VideoCaptionExtension) == "" {
		s.VideoCaptionExtension = ".txt"
	}
	if strings.TrimSpace(s.VideoCodec) == "" {
		s.VideoCodec = "libx264"
	}
	if strings.TrimSpace(s.VideoQuality) == "" {
		s.VideoQuality = "19"
	}
	if strings.TrimSpace(s.VideoEncoderPreset) == "" {
		s.VideoEncoderPreset = "medium"
	}
	if strings.TrimSpace(s.VideoSpeed) == "" {
		s.VideoSpeed = "1.0"
	}
	if s.VideoSkipFrames < 0 {
		s.VideoSkipFrames = 0
	}
	if s.VideoParallelWorkers <= 0 {
		s.VideoParallelWorkers = 1
	}
	if s.Architecture == ArchitectureLTX23 || s.Architecture == ArchitectureWAN22 {
		s.Optimizer = "Prodigy"
		if s.NetworkRank <= 0 || s.NetworkRank == 32 {
			s.NetworkRank = 128
		}
		if s.NetworkAlpha <= 0 || s.NetworkAlpha == s.NetworkRank {
			s.NetworkAlpha = 1
		}
		if s.BlocksToSwap <= 0 {
			s.BlocksToSwap = 14
		}
		s.FP8Base = true
		s.FP8Scaled = true
		s.SDPA = true
		s.GradientCheckpointing = true
		s.UsePinnedMemoryBlockSwap = true
		s.PersistentWorkers = true
		s.SaveStateOnTrainEnd = true
		if s.Architecture == ArchitectureLTX23 {
			if strings.TrimSpace(s.NetworkModule) == "" {
				s.NetworkModule = "networks.lora_ltx2"
			}
			if strings.TrimSpace(s.TimestepSampling) == "" {
				s.TimestepSampling = "shifted_logit_normal"
			}
			if strings.TrimSpace(s.LTXVersion) == "" {
				s.LTXVersion = "2.3"
			}
			if strings.TrimSpace(s.LTXMode) == "" {
				s.LTXMode = "video"
			}
			if strings.TrimSpace(s.LTXVersionCheckMode) == "" {
				s.LTXVersionCheckMode = "error"
			}
		} else {
			if strings.TrimSpace(s.NetworkModule) == "" {
				s.NetworkModule = "networks.lora_wan"
			}
			if strings.TrimSpace(s.WanTask) == "" {
				s.WanTask = "i2v-A14B"
			}
			if strings.TrimSpace(s.TimestepSampling) == "" {
				s.TimestepSampling = "shift"
			}
			if strings.TrimSpace(s.DiscreteFlowShift) == "" {
				s.DiscreteFlowShift = "5.0"
			}
		}
	}
	if s.NetworkRank <= 0 {
		s.NetworkRank = 32
	}
	if s.NetworkAlpha <= 0 {
		s.NetworkAlpha = s.NetworkRank
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
	if s.TargetVRAMPercent <= 0 {
		s.TargetVRAMPercent = 90
	}
	s.TargetVRAMPercent = clampInt(s.TargetVRAMPercent, 50, 98)
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
	case ArchitectureLTX23:
		if !fileExists(s.CheckpointPath) {
			errs = append(errs, "LTX 2.3 checkpoint file not found: "+s.CheckpointPath)
		}
		if !fileExists(s.QwenPath) {
			errs = append(errs, "Gemma text encoder file not found: "+s.QwenPath)
		}
	case ArchitectureWAN22:
		if !fileExists(s.DiTPath) {
			errs = append(errs, "Wan DiT checkpoint file not found: "+s.DiTPath)
		}
		if !fileExists(s.QwenPath) {
			errs = append(errs, "Wan T5 text encoder file not found: "+s.QwenPath)
		}
		if !fileExists(s.VAEPath) {
			errs = append(errs, "Wan VAE file not found: "+s.VAEPath)
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
