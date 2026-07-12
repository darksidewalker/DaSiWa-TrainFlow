package trainer

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"trainflow/internal/process"
)

type musubiCommandKind string

const (
	musubiCommandCacheText    musubiCommandKind = "cache-text"
	musubiCommandCacheLatents musubiCommandKind = "cache-latents"
	musubiCommandTrain        musubiCommandKind = "train"
)

type musubiCommand struct {
	Program string
	Args    []string
	Dir     string
	Env     []string
}

var requiredMusubiFiles = []string{
	"pyproject.toml",
	"src/musubi_tuner/__init__.py",
	"ltx2_train_network.py",
	"ltx2_cache_latents.py",
	"ltx2_cache_text_encoder_outputs.py",
	"wan_train_network.py",
	"wan_cache_latents.py",
	"wan_cache_text_encoder_outputs.py",
	"src/musubi_tuner/krea2_train_network.py",
	"src/musubi_tuner/krea2_cache_latents.py",
	"src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
}

func musubiRoot(root string) string {
	return filepath.Join(root, "training", "musubi-tuner")
}

func validateMusubiSource(root string) error {
	base := musubiRoot(root)
	for _, rel := range requiredMusubiFiles {
		if !process.FileExists(filepath.Join(base, filepath.FromSlash(rel))) && !dirExists(filepath.Join(base, filepath.FromSlash(rel))) {
			return fmt.Errorf("Musubi trainer source is missing from training/musubi-tuner (%s). Run scripts/sync-musubi-tuner.sh /home/darksidewalker/GitHub/musubi-tuner, then run Runtime Tool -> Install", rel)
		}
	}
	return nil
}

func buildMusubiCommand(root string, kind musubiCommandKind, s Settings, datasetTOML, outputDir string) (musubiCommand, error) {
	s = normalizeSettings(s)
	switch s.Architecture {
	case ArchitectureLTX23:
		return buildLTX23MusubiCommand(root, kind, s, datasetTOML, outputDir)
	case ArchitectureWAN22:
		return buildWAN22MusubiCommand(root, kind, s, datasetTOML, outputDir)
	case ArchitectureKrea2:
		return buildKrea2MusubiCommand(root, kind, s, datasetTOML, outputDir)
	default:
		return musubiCommand{}, fmt.Errorf("not a Musubi architecture: %s", s.Architecture)
	}
}

func buildLTX23MusubiCommand(root string, kind musubiCommandKind, s Settings, datasetTOML, outputDir string) (musubiCommand, error) {
	args := []string{}
	switch kind {
	case musubiCommandCacheText:
		args = []string{"ltx2_cache_text_encoder_outputs.py",
			"--dataset_config", datasetTOML,
			"--ltx2_checkpoint", s.CheckpointPath,
			"--gemma_safetensors", s.QwenPath,
			"--ltx2_mode", nonEmpty(s.LTXMode, "video"),
			"--ltx_version", nonEmpty(s.LTXVersion, "2.3"),
			"--mixed_precision", nonEmpty(s.MixedPrecision, "bf16"),
		}
		args = appendFields(args, s.ExtraCacheTextArgs)
	case musubiCommandCacheLatents:
		args = []string{"ltx2_cache_latents.py",
			"--dataset_config", datasetTOML,
			"--ltx2_checkpoint", s.CheckpointPath,
			"--device", "cuda",
			"--vae_dtype", nonEmpty(s.MixedPrecision, "bf16"),
			"--ltx2_mode", nonEmpty(s.LTXMode, "video"),
		}
		args = appendFields(args, s.ExtraCacheLatentsArgs)
	case musubiCommandTrain:
		args = []string{"-m", "accelerate.commands.launch"}
		args = append(args, commonAccelerateArgs(s)...)
		args = append(args,
			"ltx2_train_network.py",
			"--mixed_precision", nonEmpty(s.MixedPrecision, "bf16"),
			"--optimizer_type", musubiOptimizer(s.Optimizer),
			"--learning_rate", nonEmpty(s.LearningRate, "1.0"),
			"--optimizer_args", "decouple=True", "weight_decay=0.01", "d_coef=2.0", "use_bias_correction=True", "safeguard_warmup=True",
			"--lr_scheduler", "constant",
			"--timestep_sampling", nonEmpty(s.TimestepSampling, "shifted_logit_normal"),
			"--dataset_config", datasetTOML,
			"--output_dir", outputDir,
			"--output_name", projectNameForSettings(s),
			"--ltx2_checkpoint", s.CheckpointPath,
			"--gemma_safetensors", s.QwenPath,
			"--ltx_version", nonEmpty(s.LTXVersion, "2.3"),
			"--ltx_version_check_mode", nonEmpty(s.LTXVersionCheckMode, "error"),
			"--ltx2_mode", nonEmpty(s.LTXMode, "video"),
		)
		args = appendBoolArg(args, "--fp8_base", s.FP8Base)
		args = appendBoolArg(args, "--fp8_scaled", s.FP8Scaled)
		args = appendBoolArg(args, "--full_ft_train_text_encoder", s.FullFTTrainTextEncoder)
		args = appendBoolArg(args, "--full_ft_text_encoder_fallback", s.FullFTTextEncoderFallback)
		args = append(args, "--blocks_to_swap", strconv.Itoa(defaultInt(s.BlocksToSwap, 14)))
		args = appendBoolArg(args, "--use_pinned_memory_for_block_swap", s.UsePinnedMemoryBlockSwap)
		args = appendCommonMusubiTrainArgs(args, s)
	default:
		return musubiCommand{}, fmt.Errorf("unknown Musubi command kind: %s", kind)
	}
	return newMusubiCommand(root, args), nil
}

func buildWAN22MusubiCommand(root string, kind musubiCommandKind, s Settings, datasetTOML, outputDir string) (musubiCommand, error) {
	args := []string{}
	switch kind {
	case musubiCommandCacheText:
		args = []string{"wan_cache_text_encoder_outputs.py", "--dataset_config", datasetTOML, "--t5", s.QwenPath, "--batch_size", "16"}
		args = appendFields(args, s.ExtraCacheTextArgs)
	case musubiCommandCacheLatents:
		args = []string{"wan_cache_latents.py", "--dataset_config", datasetTOML, "--vae", s.VAEPath, "--i2v"}
		args = appendFields(args, s.ExtraCacheLatentsArgs)
	case musubiCommandTrain:
		args = []string{"-m", "accelerate.commands.launch"}
		args = append(args, commonAccelerateArgs(s)...)
		args = append(args,
			"wan_train_network.py",
			"--task", nonEmpty(s.WanTask, "i2v-A14B"),
			"--mixed_precision", nonEmpty(s.MixedPrecision, "bf16"),
			"--optimizer_type", musubiOptimizer(s.Optimizer),
			"--learning_rate", nonEmpty(s.LearningRate, "1.0"),
			"--optimizer_args", "decouple=True", "weight_decay=0.01", "d_coef=2.0", "use_bias_correction=True", "safeguard_warmup=True",
			"--lr_scheduler", "constant",
			"--timestep_sampling", nonEmpty(s.TimestepSampling, "shift"),
			"--discrete_flow_shift", nonEmpty(s.DiscreteFlowShift, "5.0"),
			"--dataset_config", datasetTOML,
			"--output_dir", outputDir,
			"--output_name", projectNameForSettings(s),
			"--dit", s.DiTPath,
			"--t5", s.QwenPath,
			"--vae", s.VAEPath,
		)
		args = appendBoolArg(args, "--force_v2_1_time_embedding", true)
		args = appendCommonMusubiTrainArgs(args, s)
	default:
		return musubiCommand{}, fmt.Errorf("unknown Musubi command kind: %s", kind)
	}
	return newMusubiCommand(root, args), nil
}

func buildKrea2MusubiCommand(root string, kind musubiCommandKind, s Settings, datasetTOML, outputDir string) (musubiCommand, error) {
	args := []string{}
	switch kind {
	case musubiCommandCacheText:
		args = []string{"src/musubi_tuner/krea2_cache_text_encoder_outputs.py", "--dataset_config", datasetTOML, "--text_encoder", s.QwenPath, "--batch_size", "4"}
		args = appendFields(args, s.ExtraCacheTextArgs)
	case musubiCommandCacheLatents:
		args = []string{"src/musubi_tuner/krea2_cache_latents.py", "--dataset_config", datasetTOML, "--vae", s.VAEPath}
		args = appendFields(args, s.ExtraCacheLatentsArgs)
	case musubiCommandTrain:
		args = []string{"-m", "accelerate.commands.launch"}
		args = append(args, commonAccelerateArgs(s)...)
		args = append(args,
			"src/musubi_tuner/krea2_train_network.py",
			"--mixed_precision", nonEmpty(s.MixedPrecision, "bf16"),
			"--save_precision", "bf16",
			"--optimizer_type", musubiOptimizer(s.Optimizer),
			"--learning_rate", nonEmpty(s.LearningRate, "1.0"),
			"--timestep_sampling", nonEmpty(s.TimestepSampling, "krea2_shift"),
			"--discrete_flow_shift", nonEmpty(s.DiscreteFlowShift, "2.5"),
			"--dataset_config", datasetTOML,
			"--output_dir", outputDir,
			"--output_name", projectNameForSettings(s),
			"--dit", s.DiTPath,
			"--text_encoder", s.QwenPath,
			"--vae", s.VAEPath,
		)
		args = append(args, musubiOptimizerArgs(s)...)
		args = append(args, musubiSchedulerArgs(s)...)
		args = appendBoolArg(args, "--fp8_base", s.FP8Base)
		args = appendBoolArg(args, "--fp8_scaled", s.FP8Scaled)
		if s.BlocksToSwap > 0 {
			args = append(args, "--blocks_to_swap", strconv.Itoa(s.BlocksToSwap))
		}
		args = appendBoolArg(args, "--use_pinned_memory_for_block_swap", s.UsePinnedMemoryBlockSwap)
		args = appendCommonMusubiTrainArgs(args, s)
	default:
		return musubiCommand{}, fmt.Errorf("unknown Musubi command kind: %s", kind)
	}
	return newMusubiCommand(root, args), nil
}

func newMusubiCommand(root string, args []string) musubiCommand {
	return musubiCommand{
		Program: process.PythonExecutable(root),
		Args:    args,
		Dir:     musubiRoot(root),
		Env:     musubiTrainingEnv(root),
	}
}

func musubiTrainingEnv(root string) []string {
	base := musubiRoot(root)
	return append(trainingEnv(base),
		"PYTHONUNBUFFERED=1",
		"PYTHONPATH="+strings.Join([]string{filepath.Join(base, "src"), base, filepath.Join(root, "training", "sd-scripts")}, string(os.PathListSeparator)),
	)
}

func appendCommonMusubiTrainArgs(args []string, s Settings) []string {
	args = appendBoolArg(args, "--gradient_checkpointing", s.GradientCheckpointing)
	args = appendBoolArg(args, "--sdpa", s.SDPA)
	args = append(args,
		"--network_module", s.NetworkModule,
		"--network_dim", strconv.Itoa(defaultInt(s.NetworkRank, 128)),
		"--network_alpha", strconv.Itoa(defaultInt(s.NetworkAlpha, 1)),
		"--max_data_loader_n_workers", "4",
	)
	args = appendBoolArg(args, "--persistent_data_loader_workers", s.PersistentWorkers)
	args = append(args, "--save_every_n_epochs", "1")
	args = appendBoolArg(args, "--save_state", true)
	args = appendBoolArg(args, "--save_state_on_train_end", s.SaveStateOnTrainEnd)
	args = append(args, "--max_train_epochs", strconv.Itoa(musubiMaxTrainEpochs(s)))
	args = append(args, "--metadata_title", projectNameForSettings(s))
	if strings.TrimSpace(s.MetadataAuthor) != "" {
		args = append(args, "--metadata_author", s.MetadataAuthor)
	}
	if strings.TrimSpace(s.MetadataTags) != "" {
		args = append(args, "--metadata_tags", s.MetadataTags)
	}
	return appendFields(args, s.ExtraTrainArgs)
}

func musubiOptimizerArgs(s Settings) []string {
	if strings.EqualFold(strings.TrimSpace(s.Optimizer), "Prodigy") || strings.TrimSpace(s.Optimizer) == "" {
		return []string{"--optimizer_args", "decouple=True", "weight_decay=0.01", "d_coef=2.0", "use_bias_correction=True", "safeguard_warmup=True"}
	}
	return nil
}

func musubiSchedulerArgs(s Settings) []string {
	if strings.EqualFold(strings.TrimSpace(s.Optimizer), "Prodigy") {
		return []string{"--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "100"}
	}
	return []string{"--lr_scheduler", "constant", "--lr_warmup_steps", "0"}
}

func musubiMaxTrainEpochs(s Settings) int {
	targetSteps := s.TrainingSteps
	if targetSteps <= 0 {
		return defaultInt(s.TargetEpochs, 6)
	}
	profile := profileFor(s)
	items := 0
	if profile.Video {
		items = countDatasetVideos(s.DatasetPath)
	} else {
		items = countDatasetImages(s.DatasetPath)
	}
	if items <= 0 {
		return defaultInt(s.TargetEpochs, 6)
	}
	batch := maxInt(s.TrainBatchSize, 1)
	grad := maxInt(s.GradientAccumulationSteps, 1)
	repeats := 1
	if profile.Video {
		repeats = maxInt(s.VideoNumRepeats, 1)
	}
	stepsPerEpoch := int(math.Ceil(float64(items*repeats) / float64(batch*grad)))
	if stepsPerEpoch <= 0 {
		stepsPerEpoch = 1
	}
	return maxInt(1, int(math.Ceil(float64(targetSteps)/float64(stepsPerEpoch))))
}

func commonAccelerateArgs(s Settings) []string {
	return []string{
		"--num_processes=1",
		"--num_machines=1",
		"--num_cpu_threads_per_process", strconv.Itoa(defaultInt(s.NumCPUThreads, 8)),
		"--mixed_precision", nonEmpty(s.MixedPrecision, "bf16"),
		"--dynamo_backend=no",
	}
}

func createMusubiDatasetTOML(projectName string, s Settings, profile trainingProfile, outDir string) (string, error) {
	if !profile.Video {
		return createMusubiImageDatasetTOML(projectName, s, profile, outDir)
	}
	frames, err := parseIntCSV(nonEmpty(s.VideoTargetFrames, "1,65,129"))
	if err != nil {
		return "", err
	}
	width := defaultInt(s.VideoWidth, s.Width)
	height := defaultInt(s.VideoHeight, s.Height)
	repeats := defaultInt(s.VideoNumRepeats, 1)
	content := strings.Builder{}
	content.WriteString("[general]\n")
	content.WriteString(fmt.Sprintf("resolution = [%d, %d]\n", width, height))
	content.WriteString(fmt.Sprintf("batch_size = %d\n", defaultInt(s.TrainBatchSize, 1)))
	content.WriteString(fmt.Sprintf("caption_extension = %s\n", tomlString(nonEmpty(s.VideoCaptionExtension, ".txt"))))
	content.WriteString(fmt.Sprintf("enable_bucket = %t\n\n", s.VideoEnableBucket))
	content.WriteString("[[datasets]]\n")
	content.WriteString(fmt.Sprintf("video_directory = %s\n", tomlString(filepath.ToSlash(absPath(s.DatasetPath)))))
	content.WriteString(fmt.Sprintf("target_frames = [%s]\n", joinInts(frames)))
	content.WriteString(fmt.Sprintf("frame_extraction = %s\n", tomlString(nonEmpty(s.VideoFrameExtraction, "full"))))
	content.WriteString(fmt.Sprintf("num_repeats = %d\n", repeats))
	path := filepath.Join(outDir, projectName+"_musubi_dataset.toml")
	return path, os.WriteFile(path, []byte(content.String()), 0644)
}

func createMusubiImageDatasetTOML(projectName string, s Settings, profile trainingProfile, outDir string) (string, error) {
	if profile.Family != trainingFamilyMusubi {
		return "", fmt.Errorf("profile %s is not a Musubi profile", profile.Architecture)
	}
	width := defaultInt(s.Width, 1024)
	height := defaultInt(s.Height, 1024)
	content := strings.Builder{}
	content.WriteString("[general]\n")
	content.WriteString(fmt.Sprintf("resolution = [%d, %d]\n", width, height))
	content.WriteString(fmt.Sprintf("batch_size = %d\n", defaultInt(s.TrainBatchSize, 1)))
	content.WriteString("caption_extension = \".txt\"\n")
	content.WriteString("enable_bucket = true\n")
	content.WriteString("bucket_no_upscale = false\n\n")
	content.WriteString("[[datasets]]\n")
	content.WriteString(fmt.Sprintf("image_directory = %s\n", tomlString(filepath.ToSlash(absPath(s.DatasetPath)))))
	content.WriteString(fmt.Sprintf("cache_directory = %s\n", tomlString(filepath.ToSlash(absPath(filepath.Join(outDir, "cache"))))))
	content.WriteString("num_repeats = 1\n")
	path := filepath.Join(outDir, projectName+"_musubi_dataset.toml")
	return path, os.WriteFile(path, []byte(content.String()), 0644)
}

func parseIntCSV(value string) ([]int, error) {
	parts := strings.Split(value, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := strconv.Atoi(part)
		if err != nil || parsed <= 0 {
			return nil, fmt.Errorf("invalid positive integer %q", part)
		}
		out = append(out, parsed)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no target frames configured")
	}
	return out, nil
}

func joinInts(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.Itoa(value)
	}
	return strings.Join(parts, ", ")
}

func defaultInt(value, fallback int) int {
	if value > 0 {
		return value
	}
	return fallback
}

func musubiOptimizer(value string) string {
	trimmed := strings.TrimSpace(value)
	if strings.EqualFold(trimmed, "Prodigy") || trimmed == "" {
		return "prodigyopt.Prodigy"
	}
	if strings.EqualFold(trimmed, "AdamW8bit") {
		return "bitsandbytes.optim.AdamW8bit"
	}
	return trimmed
}

func appendBoolArg(args []string, name string, enabled bool) []string {
	if enabled {
		return append(args, name)
	}
	return args
}

func appendFields(args []string, extra string) []string {
	for _, field := range strings.Fields(extra) {
		args = append(args, field)
	}
	return args
}
