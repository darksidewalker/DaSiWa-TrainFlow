package trainer

import (
	"fmt"
	"math"
)

type profileDefaults struct {
	NetworkRank    int
	LearningRate   string
	UNetLR         string
	TextEncoderLR1 string
	TextEncoderLR2 string
	Optimizer      string
	BaseSteps      int
	TargetRepeats  int
	MinSteps       int
	MaxSteps       int
}

func defaultsForProfile(profile trainingProfile) profileDefaults {
	switch profile.Architecture {
	case ArchitectureSDXL:
		return profileDefaults{
			NetworkRank:    32,
			LearningRate:   "1e-4",
			UNetLR:         "1e-4",
			TextEncoderLR1: "1e-5",
			TextEncoderLR2: "1e-5",
			Optimizer:      "AdamW8bit",
			BaseSteps:      1800,
			TargetRepeats:  30,
			MinSteps:       1200,
			MaxSteps:       3600,
		}
	default:
		return profileDefaults{
			NetworkRank:   32,
			LearningRate:  "1e-4",
			UNetLR:        "1e-4",
			Optimizer:     "AdamW8bit",
			BaseSteps:     1100,
			TargetRepeats: 19,
			MinSteps:      800,
			MaxSteps:      2200,
		}
	}
}

func applyStableDefaults(s Settings) (Settings, string) {
	s = normalizeSettings(s)
	profile := profileFor(s)
	defaults := defaultsForProfile(profile)
	imageCount := countDatasetImages(s.DatasetPath)
	if imageCount <= 0 {
		imageCount = 30
	}

	s.NetworkRank = defaults.NetworkRank
	s.LearningRate = defaults.LearningRate
	s.UNetLR = defaults.UNetLR
	s.TextEncoderLR1 = defaults.TextEncoderLR1
	s.TextEncoderLR2 = defaults.TextEncoderLR2
	s.Optimizer = defaults.Optimizer
	s.TrainBatchSize = 1
	s.GradientAccumulationSteps = recommendedGradAccum(imageCount)
	s.TrainUNetOnly = true
	s.FlashAttention = false

	effectiveBatch := s.TrainBatchSize * s.GradientAccumulationSteps
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
	return s, message
}

func recommendedGradAccum(imageCount int) int {
	if imageCount >= 80 {
		return 2
	}
	return 1
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

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
