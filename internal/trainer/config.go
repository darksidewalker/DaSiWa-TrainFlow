package trainer

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func createSamplePrompts(projectName string, s Settings, outDir string) (string, error) {
	trigger := strings.TrimSpace(s.TriggerWord)
	prompt := strings.TrimSpace(strings.ReplaceAll(s.PositivePrompt, "\n", " "))
	if trigger != "" && !strings.HasPrefix(prompt, trigger) {
		if prompt == "" {
			prompt = trigger
		} else {
			prompt = trigger + ", " + prompt
		}
	}
	neg := strings.TrimSpace(strings.ReplaceAll(s.NegativePrompt, "\n", " "))
	content := fmt.Sprintf("%s --n %s --w %d --h %d --l %s --s %d --d %d",
		prompt,
		neg,
		s.Width,
		s.Height,
		strconv.FormatFloat(s.SampleCFG, 'f', -1, 64),
		s.SampleStepsGen,
		s.SampleSeed,
	)
	path := filepath.Join(outDir, projectName+"_prompts.txt")
	return path, os.WriteFile(path, []byte(content), 0644)
}

func createDatasetTOML(projectName string, s Settings, baseRes, maxBucket int, outDir string) (string, error) {
	numImages := countDatasetImages(s.DatasetPath)
	if numImages == 0 {
		numImages = 1
	}
	effectiveBatch := s.TrainBatchSize * s.GradientAccumulationSteps
	repeats := (s.TrainingSteps*effectiveBatch + numImages - 1) / numImages
	if repeats < 1 {
		repeats = 1
	}
	prefix := ""
	if strings.TrimSpace(s.TriggerWord) != "" {
		prefix = strings.TrimSpace(s.TriggerWord) + ", "
	}

	content := strings.Builder{}
	content.WriteString("[general]\n")
	content.WriteString("enable_bucket = true\n")
	content.WriteString("min_bucket_reso = 256\n")
	content.WriteString(fmt.Sprintf("max_bucket_reso = %d\n", maxBucket))
	content.WriteString("bucket_reso_steps = 64\n")
	content.WriteString("bucket_no_upscale = true\n\n")
	content.WriteString("[[datasets]]\n")
	content.WriteString(fmt.Sprintf("resolution = %d\n\n", baseRes))
	content.WriteString("[[datasets.subsets]]\n")
	content.WriteString(fmt.Sprintf("image_dir = %s\n", tomlString(filepath.ToSlash(absPath(s.DatasetPath)))))
	content.WriteString("caption_extension = \".txt\"\n")
	content.WriteString(fmt.Sprintf("num_repeats = %d\n", repeats))
	if prefix == "" {
		content.WriteString("caption_prefix = \"\"\n")
	} else {
		content.WriteString(fmt.Sprintf("caption_prefix = %s\n", tomlString(prefix)))
	}
	content.WriteString("keep_tokens = 1\n")
	content.WriteString("caption_dropout_rate = 0.05\n")

	path := filepath.Join(outDir, projectName+"_dataset.toml")
	return path, os.WriteFile(path, []byte(content.String()), 0644)
}

func createTrainingTOML(projectName string, s Settings, outputDir, promptPath, outDir string) (string, error) {
	scheduler := "cosine"
	optArgs := []string{"weight_decay=0.01"}
	if s.Optimizer == "Prodigy" {
		scheduler = "constant"
		optArgs = []string{
			"decouple=True",
			"weight_decay=0.01",
			"d_coef=1.0",
			"use_bias_correction=True",
			"safeguard_warmup=True",
			"betas=0.9,0.99",
		}
	}

	content := strings.Builder{}
	content.WriteString(fmt.Sprintf("pretrained_model_name_or_path = %s\n", tomlString(filepath.ToSlash(absPath(s.DiTPath)))))
	content.WriteString(fmt.Sprintf("qwen3 = %s\n", tomlString(filepath.ToSlash(absPath(s.QwenPath)))))
	content.WriteString(fmt.Sprintf("vae = %s\n", tomlString(filepath.ToSlash(absPath(s.VAEPath)))))
	content.WriteString("network_module = \"networks.lora_anima\"\n")
	content.WriteString(fmt.Sprintf("network_dim = %d\n", s.NetworkRank))
	content.WriteString(fmt.Sprintf("network_alpha = %d\n", s.NetworkRank))
	content.WriteString(fmt.Sprintf("network_train_unet_only = %t\n", s.TrainUNetOnly))
	content.WriteString("gradient_checkpointing = true\n")
	content.WriteString("max_grad_norm = 1.0\n")
	content.WriteString(fmt.Sprintf("learning_rate = %s\n", s.LearningRate))
	content.WriteString(fmt.Sprintf("optimizer_type = %s\n", tomlString(s.Optimizer)))
	content.WriteString("optimizer_args = [")
	for i, arg := range optArgs {
		if i > 0 {
			content.WriteString(", ")
		}
		content.WriteString(tomlString(arg))
	}
	content.WriteString("]\n")
	content.WriteString(fmt.Sprintf("lr_scheduler = %s\n", tomlString(scheduler)))
	content.WriteString(fmt.Sprintf("max_train_steps = %d\n", s.TrainingSteps))
	content.WriteString(fmt.Sprintf("train_batch_size = %d\n", s.TrainBatchSize))
	content.WriteString(fmt.Sprintf("gradient_accumulation_steps = %d\n", s.GradientAccumulationSteps))
	content.WriteString("mixed_precision = \"bf16\"\n")
	content.WriteString(fmt.Sprintf("output_dir = %s\n", tomlString(filepath.ToSlash(absPath(outputDir)))))
	content.WriteString(fmt.Sprintf("output_name = %s\n", tomlString(projectName)))
	content.WriteString(fmt.Sprintf("save_every_n_steps = %d\n", s.SaveSteps))
	content.WriteString(fmt.Sprintf("sample_every_n_steps = %d\n", s.SampleSteps))
	content.WriteString(fmt.Sprintf("sample_prompts = %s\n", tomlString(filepath.ToSlash(absPath(promptPath)))))
	content.WriteString("save_state = true\n")
	content.WriteString("save_last_n_steps_state = 1\n")
	content.WriteString("save_last_n_epochs_state = 1\n")
	if resumePath := resolveResumePath(s, outputDir); resumePath != "" {
		content.WriteString(fmt.Sprintf("resume = %s\n", tomlString(filepath.ToSlash(absPath(resumePath)))))
	}
	content.WriteString("sample_sampler = \"euler\"\n")
	content.WriteString("timestep_sampling = \"sigmoid\"\n")
	content.WriteString("discrete_flow_shift = 1.0\n")
	content.WriteString("sigmoid_scale = 1.3\n")
	content.WriteString("weighting_scheme = \"logit_normal\"\n")
	content.WriteString("cache_latents = true\n")
	content.WriteString("cache_latents_to_disk = true\n")
	content.WriteString("cache_text_encoder_outputs = true\n")
	content.WriteString("cache_text_encoder_outputs_to_disk = true\n")
	attnMode := "torch"
	if s.FlashAttention {
		attnMode = "flash"
	}
	content.WriteString(fmt.Sprintf("attn_mode = %s\n", tomlString(attnMode)))
	content.WriteString("save_model_as = \"safetensors\"\n")
	content.WriteString("save_precision = \"bf16\"\n")
	content.WriteString("max_data_loader_n_workers = 4\n")
	content.WriteString("vae_chunk_size = 32\n")
	content.WriteString("vae_disable_cache = true\n")
	content.WriteString(fmt.Sprintf("seed = %d\n", s.TrainSeed))

	path := filepath.Join(outDir, projectName+"_training.toml")
	return path, os.WriteFile(path, []byte(content.String()), 0644)
}

func resolveResumePath(s Settings, outputDir string) string {
	if !s.ResumeEnabled {
		return ""
	}
	if !s.AutoResume && strings.TrimSpace(s.ResumePath) != "" {
		return strings.TrimSpace(s.ResumePath)
	}
	if strings.TrimSpace(s.ResumePath) != "" && dirExists(strings.TrimSpace(s.ResumePath)) {
		return strings.TrimSpace(s.ResumePath)
	}
	return findLastStateDir(outputDir)
}

func countDatasetImages(datasetPath string) int {
	entries, err := os.ReadDir(datasetPath)
	if err != nil {
		return 0
	}
	count := 0
	for _, entry := range entries {
		if !entry.IsDir() && validImageExt(entry.Name()) {
			count++
		}
	}
	return count
}

func tomlString(value string) string {
	return strconv.Quote(value)
}

func absPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	return abs
}
