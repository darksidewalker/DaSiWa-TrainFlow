package trainer

import (
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
)

type profileDefaults struct {
	NetworkRank        int
	LearningRate       string
	UNetLR             string
	TextEncoderLR1     string
	TextEncoderLR2     string
	Optimizer          string
	BaseSteps          int
	TargetRepeats      int
	VideoTargetRepeats int
	VideoTargetEpochs  int
	VideoTargetSteps   int
	VideoMinSteps      int
	VideoMaxSteps      int
	MinSteps           int
	MaxSteps           int
}

func defaultsForProfile(profile trainingProfile) profileDefaults {
	switch profile.Architecture {
	case ArchitectureSDXL:
		return profileDefaults{
			NetworkRank:        32,
			LearningRate:       "1e-4",
			UNetLR:             "1e-4",
			TextEncoderLR1:     "1e-5",
			TextEncoderLR2:     "1e-5",
			Optimizer:          "Prodigy",
			BaseSteps:          1800,
			TargetRepeats:      30,
			VideoTargetRepeats: 0,
			VideoTargetEpochs:  0,
			VideoTargetSteps:   0,
			VideoMinSteps:      0,
			VideoMaxSteps:      0,
			MinSteps:           1200,
			MaxSteps:           3600,
		}
	case ArchitectureLTX23:
		return profileDefaults{
			NetworkRank:        128,
			LearningRate:       "1.0",
			UNetLR:             "1e-4",
			Optimizer:          "Prodigy",
			BaseSteps:          1100,
			TargetRepeats:      19,
			VideoTargetRepeats: 19,
			VideoTargetEpochs:  6,
			VideoTargetSteps:   3200,
			VideoMinSteps:      3000,
			VideoMaxSteps:      4000,
			MinSteps:           800,
			MaxSteps:           2200,
		}
	case ArchitectureWAN22:
		return profileDefaults{
			NetworkRank:        128,
			LearningRate:       "1.0",
			UNetLR:             "1e-4",
			Optimizer:          "Prodigy",
			BaseSteps:          1800,
			TargetRepeats:      30,
			VideoTargetRepeats: 30,
			VideoTargetEpochs:  6,
			VideoTargetSteps:   3200,
			VideoMinSteps:      3000,
			VideoMaxSteps:      4000,
			MinSteps:           1200,
			MaxSteps:           3600,
		}
	default:
		return profileDefaults{
			NetworkRank:        32,
			LearningRate:       "1e-4",
			UNetLR:             "1e-4",
			Optimizer:          "Prodigy",
			BaseSteps:          1100,
			TargetRepeats:      19,
			VideoTargetRepeats: 0,
			VideoTargetEpochs:  0,
			VideoTargetSteps:   0,
			VideoMinSteps:      0,
			VideoMaxSteps:      0,
			MinSteps:           800,
			MaxSteps:           2200,
		}
	}
}

func applyStableDefaults(s Settings) (Settings, string) {
	return applyStableDefaultsWithVRAM(s, 0)
}

func applyStableDefaultsWithVRAM(s Settings, totalVRAMMB int) (Settings, string) {
	s = normalizeSettings(s)
	profile := profileFor(s)
	defaults := defaultsForProfile(profile)

	var imageCount int
	if profile.Video {
		imageCount = countDatasetVideos(s.DatasetPath)
	} else {
		imageCount = countDatasetImages(s.DatasetPath)
	}
	if imageCount <= 0 {
		imageCount = 30
	}

	s.NetworkRank = defaults.NetworkRank
	if !profile.Video {
		if s.Optimizer == "" {
			s.Optimizer = defaults.Optimizer
		}
		if s.LearningRate == "" {
			s.LearningRate = defaults.LearningRate
		}
	}
	s.UNetLR = defaults.UNetLR
	s.TextEncoderLR1 = defaults.TextEncoderLR1
	s.TextEncoderLR2 = defaults.TextEncoderLR2
	s.TrainBatchSize = recommendedBatchSize(profile, s, totalVRAMMB)
	s.GradientAccumulationSteps = recommendedGradAccum(profile, imageCount, totalVRAMMB)
	s.TrainUNetOnly = true
	s.FlashAttention = false
	s = normalizeSettings(s)

	effectiveBatch := s.TrainBatchSize * s.GradientAccumulationSteps

	if profile.Video && defaults.VideoTargetRepeats > 0 {
		return applyVideoDefaults(s, imageCount, profile, defaults, totalVRAMMB)
	}

	steps := int(math.Ceil(float64(imageCount*defaults.TargetRepeats) / float64(effectiveBatch)))
	steps = clampInt(roundUpTo(steps, 50), defaults.MinSteps, defaults.MaxSteps)
	s.TrainingSteps = steps
	s.SaveSteps = recommendedInterval(steps)
	s.SampleSteps = recommendedInterval(steps)

	actualRepeats := float64(steps*effectiveBatch) / float64(imageCount)
	message := fmt.Sprintf(
		"%s auto calc: %d images, target %d repeats/image, batch %d x grad %d = effective %d, %d steps.",
		profile.Label,
		imageCount,
		defaults.TargetRepeats,
		s.TrainBatchSize,
		s.GradientAccumulationSteps,
		effectiveBatch,
		steps,
	)
	message = fmt.Sprintf("%s Actual exposure: %.1f repeats/image after rounding.", message, actualRepeats)
	if totalVRAMMB > 0 {
		message = fmt.Sprintf("%s VRAM target: %d%% of %d MB.", message, s.TargetVRAMPercent, totalVRAMMB)
	} else {
		message += " VRAM auto-detect unavailable; using safe batch defaults."
	}
	return s, message
}

func recommendedBatchSize(profile trainingProfile, s Settings, totalVRAMMB int) int {
	if profile.Video {
		return 1
	}
	if totalVRAMMB <= 0 {
		return 1
	}
	baseMB, perBatchMB, maxBatch := vramEstimate(profile, s.NetworkRank)
	targetMB := totalVRAMMB * s.TargetVRAMPercent / 100
	batch := (targetMB - baseMB) / perBatchMB
	return clampInt(batch, 1, maxBatch)
}

// applyVideoDefaults calculates video-specific training defaults
// (repeats, epochs, steps, save/sample intervals) and returns a descriptive message.
func applyVideoDefaults(s Settings, videoCount int, profile trainingProfile, defaults profileDefaults, totalVRAMMB int) (Settings, string) {
	targetEpochs := s.TargetEpochs
	if targetEpochs <= 0 {
		targetEpochs = defaults.VideoTargetEpochs
	}
	s.VideoNumRepeats = recommendedVideoRepeats(videoCount, targetEpochs, s.TrainBatchSize, defaults)
	if s.TargetEpochs <= 0 {
		s.TargetEpochs = targetEpochs
	}
	effectiveVideoSteps := s.TargetEpochs * videoCount * s.VideoNumRepeats / maxInt(s.TrainBatchSize, 1)
	steps := effectiveVideoSteps / maxInt(s.GradientAccumulationSteps, 1)
	if steps < 1 {
		steps = 1
	}
	s.TrainingSteps = steps
	s.SaveSteps = recommendedVideoInterval(steps)
	s.SampleSteps = recommendedVideoInterval(steps)

	message := fmt.Sprintf(
		"%s auto calc: %d videos, chose %d repeats/video x %d epochs = %d effective steps; batch %d x grad %d gives %d optimizer steps.",
		profile.Label,
		videoCount,
		s.VideoNumRepeats,
		s.TargetEpochs,
		effectiveVideoSteps,
		s.TrainBatchSize,
		s.GradientAccumulationSteps,
		s.TrainingSteps,
	)
	if defaults.VideoMinSteps > 0 && defaults.VideoMaxSteps > 0 {
		message = fmt.Sprintf("%s Target window: %d-%d effective steps.", message, defaults.VideoMinSteps, defaults.VideoMaxSteps)
	}
	if totalVRAMMB > 0 {
		message = fmt.Sprintf("%s VRAM target: %d%% of %d MB.", message, s.TargetVRAMPercent, totalVRAMMB)
	} else {
		message += " VRAM auto-detect unavailable; using safe batch defaults."
	}
	return s, message
}

func vramEstimate(profile trainingProfile, rank int) (baseMB, perBatchMB, maxBatch int) {
	rankOverDefault := maxInt(rank-32, 0)
	switch profile.Architecture {
	case ArchitectureSDXL:
		return 10000 + rankOverDefault*40, 4500, 8
	default:
		return 15000 + rankOverDefault*60, 7500, 4
	}
}

func detectLargestGPUMemoryMB() int {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	maxMemory := 0
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		value, err := strconv.Atoi(strings.TrimSpace(line))
		if err == nil && value > maxMemory {
			maxMemory = value
		}
	}
	return maxMemory
}

func recommendedGradAccum(profile trainingProfile, imageCount, totalVRAMMB int) int {
	if profile.Video {
		switch {
		case totalVRAMMB >= 30000:
			return 4
		case totalVRAMMB >= 20000:
			return 2
		default:
			return 1
		}
	}
	if imageCount >= 80 {
		return 2
	}
	return 1
}

func recommendedVideoRepeats(videoCount, epochs, batchSize int, defaults profileDefaults) int {
	if videoCount <= 0 || epochs <= 0 {
		return maxInt(defaults.VideoTargetRepeats, 1)
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	exactRepeats := float64(defaults.VideoTargetSteps*batchSize) / float64(videoCount*epochs)
	base := int(math.Floor(exactRepeats))
	if base < 1 {
		base = 1
	}
	bestRepeats := maxInt(defaults.VideoTargetRepeats, 1)
	bestScore := math.Inf(1)
	for _, repeats := range []int{base - 1, base, base + 1, base + 2} {
		if repeats < 1 {
			continue
		}
		steps := videoCount * epochs * repeats / batchSize
		score := math.Abs(float64(steps - defaults.VideoTargetSteps))
		if steps < defaults.VideoMinSteps {
			score += float64(defaults.VideoMinSteps-steps) * 3
		}
		if defaults.VideoMaxSteps > 0 && steps > defaults.VideoMaxSteps {
			score += float64(steps-defaults.VideoMaxSteps) * 3
		}
		if score < bestScore {
			bestScore = score
			bestRepeats = repeats
		}
	}
	return maxInt(bestRepeats, 1)
}

func recommendedVideoInterval(steps int) int {
	return clampInt(roundUpTo(steps/12, 50), 250, 500)
}

func recommendedInterval(steps int) int {
	return clampInt(roundUpTo(steps/10, 50), 100, 300)
}

func roundUpTo(value, step int) int {
	if step <= 0 {
		return value
	}
	if value <= 0 {
		return step
	}
	return ((value + step - 1) / step) * step
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
