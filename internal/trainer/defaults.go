package trainer

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

type profileDefaults struct {
	NetworkRank        int
	NetworkAlpha       int
	LearningRate       string
	UNetLR             string
	TextEncoderLR1     string
	TextEncoderLR2     string
	Optimizer          string
	BaseSteps          int
	TargetRepeats      int // baseline (AdamW); Prodigy applies a multiplier
	VideoTargetRepeats int
	VideoTargetEpochs  int
	VideoTargetSteps   int
	VideoMinSteps      int
	VideoMaxSteps      int
	MinSteps           int // baseline (AdamW); Prodigy applies a multiplier
	MaxSteps           int // baseline (AdamW); Prodigy applies a multiplier
}

// optimizerSteps holds optimizer-specific step-count adjustments.
// Prodigy converges ~25% faster than AdamW (fewer steps needed) and
// needs ~100 steps of warmup for its LR estimation to stabilize.
// AdamW needs a full cosine decay cycle (~800+ steps minimum).
type optimizerSteps struct {
	StepMultiplier float64 // applied to raw step count; Prodigy=0.75, AdamW=1.0
	AbsoluteMin    int     // hard floor regardless of profile scaling
	WarmupSteps    int     // informational; used in message
}

func optimizerStepsFor(optimizer string) optimizerSteps {
	switch {
	case strings.EqualFold(optimizer, "Prodigy"):
		return optimizerSteps{StepMultiplier: 0.75, AbsoluteMin: 600, WarmupSteps: 100}
	default:
		// AdamW, AdamW8bit, Adafactor, Lion, DAdapt*, etc.
		return optimizerSteps{StepMultiplier: 1.0, AbsoluteMin: 800, WarmupSteps: 50}
	}
}

func defaultsForProfile(profile trainingProfile) profileDefaults {
	switch profile.Architecture {
	case ArchitectureSDXL:
		return profileDefaults{
			NetworkRank:        64,
			NetworkAlpha:       32,
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
			NetworkRank:        64,
			NetworkAlpha:       64,
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
			NetworkRank:        64,
			NetworkAlpha:       64,
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
			NetworkRank:        48,
			NetworkAlpha:       32,
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

	// Textual Inversion mode: apply TI-specific defaults
	if s.TrainingMode == string(TrainingModeTI) {
		return applyTIDefaults(s, profile, imageCount, defaults, totalVRAMMB)
	}

	s.NetworkRank = defaults.NetworkRank
	s.NetworkAlpha = defaults.NetworkAlpha
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

	// Optimizer-aware step adjustment: Prodigy converges ~25% faster.
	optSteps := optimizerStepsFor(s.Optimizer)

	effectiveBatch := s.TrainBatchSize * s.GradientAccumulationSteps

	if profile.Video && defaults.VideoTargetRepeats > 0 {
		return applyVideoDefaults(s, imageCount, profile, defaults, totalVRAMMB)
	}

	// Scale target repeats by optimizer multiplier so Prodigy trains fewer steps.
	targetRepeats := int(math.Round(float64(defaults.TargetRepeats) * optSteps.StepMultiplier))
	if targetRepeats < 1 {
		targetRepeats = 1
	}
	steps := int(math.Ceil(float64(imageCount*targetRepeats) / float64(effectiveBatch)))

	// Scale min/max bounds by optimizer multiplier.
	optMin := int(math.Round(float64(defaults.MinSteps) * optSteps.StepMultiplier))
	optMax := int(math.Round(float64(defaults.MaxSteps) * optSteps.StepMultiplier))
	// Enforce optimizer absolute floor.
	if optMin < optSteps.AbsoluteMin {
		optMin = optSteps.AbsoluteMin
	}

	// LoRA needs a minimum number of optimizer steps to converge properly
	// (scheduler warmup, AdamW/Prodigy buffers, cosine decay). When effective
	// batch is so high that target-repeat steps fall below that, cap the
	// effective batch to preserve step count. Better to leave VRAM headroom
	// than train a useless model.
	if steps < optMin {
		idealEffective := int(math.Floor(float64(imageCount*targetRepeats) / float64(optMin)))
		if idealEffective < effectiveBatch {
			effectiveBatch = max(idealEffective, 1)
			if effectiveBatch < s.TrainBatchSize {
				s.TrainBatchSize = effectiveBatch
			} else if effectiveBatch < s.TrainBatchSize*s.GradientAccumulationSteps {
				s.GradientAccumulationSteps = max(effectiveBatch/s.TrainBatchSize, 1)
			}
			effectiveBatch = s.TrainBatchSize * s.GradientAccumulationSteps
			steps = int(math.Ceil(float64(imageCount*targetRepeats) / float64(effectiveBatch)))
		}
	}
	steps = clampInt(roundUpTo(steps, 50), optMin, optMax)
	s.TrainingSteps = steps
	s.SaveSteps = recommendedInterval(steps)
	s.SampleSteps = recommendedInterval(steps)

	actualRepeats := float64(steps*effectiveBatch) / float64(imageCount)
	message := fmt.Sprintf(
		"%s auto calc: %d images, optimizer %s (warmup ~%d steps), target %d repeats/image, batch %d x grad %d = effective %d, %d steps.",
		profile.Label,
		imageCount,
		s.Optimizer,
		optSteps.WarmupSteps,
		targetRepeats,
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
	case ArchitectureAnima:
		// Anima (DiT + Qwen3 + VAE) is much lighter per-batch than SDXL.
		// Real-world: batch 1 ≈ 9GB, batch 10 ≈ 31GB on RTX 5090.
		// Base ≈ 7GB, per-batch ≈ 2.4GB, safe max 16.
		return 7000 + rankOverDefault*30, 2400, 16
	default:
		return 15000 + rankOverDefault*60, 7500, 4
	}
}

func detectLargestGPUMemoryMB() int {
	maxMemory := 0

	// Try NVIDIA via nvidia-smi
	if out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output(); err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			value, err := strconv.Atoi(strings.TrimSpace(line))
			if err == nil && value > maxMemory {
				maxMemory = value
			}
		}
	}

	// Try AMD via amdgpu sysfs
	if v := detectAMDGPUMemoryMB(); v > maxMemory {
		maxMemory = v
	}

	return maxMemory
}

// detectAMDGPUMemoryMB reads VRAM from amdgpu sysfs.
func detectAMDGPUMemoryMB() int {
	maxMemory := 0
	entries, err := os.ReadDir("/sys/class/drm")
	if err != nil {
		return 0
	}
	seenPCI := make(map[string]bool)
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "card") {
			continue
		}
		if strings.Count(entry.Name(), "-") > 0 {
			continue
		}
		cardDir := filepath.Join("/sys/class/drm", entry.Name(), "device")

		vendorData, err := os.ReadFile(filepath.Join(cardDir, "vendor"))
		if err != nil || strings.TrimSpace(string(vendorData)) != "0x1002" {
			continue
		}

		// Deduplicate
		ueventData, _ := os.ReadFile(filepath.Join(cardDir, "uevent"))
		var pciSlot string
		for _, line := range strings.Split(string(ueventData), "\n") {
			if strings.HasPrefix(line, "PCI_SLOT_NAME=") {
				pciSlot = strings.TrimPrefix(line, "PCI_SLOT_NAME=")
				break
			}
		}
		if pciSlot == "" || seenPCI[pciSlot] {
			continue
		}
		seenPCI[pciSlot] = true

		data, err := os.ReadFile(filepath.Join(cardDir, "mem_info_vram_total"))
		if err != nil {
			continue
		}
		val, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
		if err != nil {
			continue
		}
		mb := int(val / (1024 * 1024))
		if mb > maxMemory {
			maxMemory = mb
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
	if profile.Architecture == ArchitectureAnima {
		// Anima batch size already scales with VRAM (see vramEstimate).
		// Keep grad accum at 1 so step count stays proportional to
		// target repeats. Only bump grad accum when VRAM detection
		// failed and we're stuck at batch 1 with a large dataset.
		if totalVRAMMB > 0 {
			return 1
		}
		if imageCount >= 80 {
			return 2
		}
		return 1
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
	ideal := clampInt(roundUpTo(steps/12, 50), 250, 500)
	return bestDivisorInRange(steps, 250, 500, ideal)
}

func recommendedInterval(steps int) int {
	ideal := clampInt(roundUpTo(steps/8, 50), 200, 600)
	return bestDivisorInRange(steps, 200, 600, ideal)
}

// applyTIDefaults calculates TI-specific defaults.
// TI loads the full model (UNet + text encoder + VAE) into VRAM for training,
// so batch size depends on VRAM just like LoRA — even though only the tiny
// embedding vectors (~8KB) are being updated.
func applyTIDefaults(s Settings, profile trainingProfile, imageCount int, defaults profileDefaults, totalVRAMMB int) (Settings, string) {
	if s.TIPlaceholderToken == "" {
		s.TIPlaceholderToken = "*test*"
	}
	if s.TINumVectors <= 0 {
		s.TINumVectors = 16
	}

	// TI batch size: based on VRAM. TI loads the full model for forward passes,
	// so memory usage is similar to LoRA training (just fewer trainable params).
	if s.TIPerDeviceBatchSz <= 0 {
		s.TIPerDeviceBatchSz = recommendedTIBatchSize(profile, totalVRAMMB)
	}

	// TI learning rate: scaled by num_vectors. More vectors = lower LR to prevent
	// overfitting. Base LR of 0.01 at 8 vectors, scaled linearly.
	// Reference: https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_textual_inversion.md
	if s.TILearningRate == "" {
		s.TILearningRate = recommendedTILearningRate(s.TINumVectors)
	}

	// TI uses similar step logic to LoRA (target repeats per image)
	targetRepeats := defaults.TargetRepeats
	if targetRepeats <= 0 {
		targetRepeats = 19
	}
	steps := int(math.Ceil(float64(imageCount*targetRepeats) / float64(s.TIPerDeviceBatchSz)))
	tidMin := 800
	tidMax := 3000
	steps = clampInt(roundUpTo(steps, 50), tidMin, tidMax)
	s.TrainingSteps = steps
	s.SaveSteps = recommendedInterval(steps)
	s.SampleSteps = recommendedInterval(steps)

	message := fmt.Sprintf(
		"%s TI auto calc: %d images, %d vectors, LR %s, target %d repeats/image, batch %d, %d steps.",
		profile.Label,
		imageCount,
		s.TINumVectors,
		s.TILearningRate,
		targetRepeats,
		s.TIPerDeviceBatchSz,
		steps,
	)
	if totalVRAMMB > 0 {
		message = fmt.Sprintf("%s VRAM: %d MB detected.", message, totalVRAMMB)
	} else {
		message += " VRAM auto-detect unavailable."
	}
	return s, message
}

// recommendedTIBatchSize picks a TI batch size based on VRAM.
// TI loads the full model for forward passes, so memory is similar to LoRA.
func recommendedTIBatchSize(profile trainingProfile, totalVRAMMB int) int {
	if totalVRAMMB <= 0 {
		return 1
	}
	// TI memory estimates: base model load + per-batch overhead.
	// SDXL TI: ~9GB base + ~3.5GB/batch, max 8
	// SD 1.5 TI: ~5GB base + ~1.5GB/batch, max 8
	// Anima TI: ~7GB base + ~2GB/batch, max 8
	baseMB, perBatchMB, maxBatch := tiVramEstimate(profile)
	targetMB := totalVRAMMB * 85 / 100 // use 85% of VRAM for TI
	batch := (targetMB - baseMB) / perBatchMB
	return clampInt(batch, 1, maxBatch)
}

// tiVramEstimate returns base memory, per-batch memory, and max batch for TI training.
func tiVramEstimate(profile trainingProfile) (baseMB, perBatchMB, maxBatch int) {
	switch profile.Architecture {
	case ArchitectureSDXL:
		// SDXL TI: ~9GB base model + ~3.5GB per batch
		return 9000, 3500, 8
	case ArchitectureAnima:
		// Anima TI: ~7GB base + ~2GB per batch
		return 7000, 2000, 8
	default:
		// SD 1.5 / other: ~5GB base + ~1.5GB per batch
		return 5000, 1500, 8
	}
}

// recommendedTILearningRate scales LR by num_vectors.
// More vectors = lower LR to prevent overfitting.
// Base: 0.01 at 8 vectors, scaled linearly.
func recommendedTILearningRate(numVectors int) string {
	if numVectors <= 0 {
		numVectors = 16
	}
	// lr = 0.01 * (8 / num_vectors), clamped to [0.001, 0.1]
	lr := 0.01 * float64(8) / float64(numVectors)
	lr = math.Max(0.001, math.Min(0.1, lr))
	return fmt.Sprintf("%.4g", lr)
}

// bestDivisorInRange finds the divisor of `steps` closest to `ideal` within [lo, hi].
// If no divisor exists in range, returns the divisor closest to the range boundaries.
func bestDivisorInRange(steps, lo, hi, ideal int) int {
	// Collect all divisors
	divisors := divisorsOf(steps)

	// Prefer: divisor in [lo, hi] closest to ideal
	best := -1
	bestDist := math.MaxInt64
	for _, d := range divisors {
		if d < lo || d > hi {
			continue
		}
		dist := absInt(d - ideal)
		if dist < bestDist {
			bestDist = dist
			best = d
		}
	}
	if best > 0 {
		return best
	}

	// Fallback: closest divisor outside the range
	best = -1
	bestDist = math.MaxInt64
	for _, d := range divisors {
		if d >= lo && d <= hi {
			continue
		}
		dist := absInt(d - ideal)
		if dist < bestDist {
			bestDist = dist
			best = d
		}
	}
	if best > 0 {
		return best
	}

	// Absolute last resort (shouldn't happen; every number has divisor 1)
	return lo
}

// divisorsOf returns all positive divisors of n in ascending order.
func divisorsOf(n int) []int {
	var small, large []int
	for i := 1; i*i <= n; i++ {
		if n%i == 0 {
			small = append(small, i)
			if i*i != n {
				large = append(large, n/i)
			}
		}
	}
	// large is in descending order; reverse and append
	result := make([]int, 0, len(small)+len(large))
	result = append(result, small...)
	for i := len(large) - 1; i >= 0; i-- {
		result = append(result, large[i])
	}
	return result
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
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
