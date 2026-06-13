package trainer

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestCreateTrainingTOMLAttentionMode(t *testing.T) {
	dir := t.TempDir()
	settings := DefaultSettings(dir)
	profile := profileFor(settings)

	path, err := createTrainingTOML("plain", settings, profile, dir, filepath.Join(dir, "prompts.txt"), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "attn_mode = \"torch\"") {
		t.Fatalf("expected torch attention mode, got:\n%s", data)
	}

	settings.FlashAttention = true
	path, err = createTrainingTOML("flash", settings, profile, dir, filepath.Join(dir, "prompts.txt"), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "attn_mode = \"flash\"") {
		t.Fatalf("expected flash attention mode, got:\n%s", data)
	}
}

func TestApplyStableDefaultsAnimaTypicalDataset(t *testing.T) {
	dir := testImageDataset(t, 60)
	settings := DefaultSettings(dir)
	settings.DatasetPath = dir

	next, message := applyStableDefaults(settings)

	if next.Architecture != ArchitectureAnima {
		t.Fatalf("expected Anima architecture, got %q", next.Architecture)
	}
	if next.TrainingSteps != 1150 {
		t.Fatalf("expected rounded Anima steps near 1100, got %d (%s)", next.TrainingSteps, message)
	}
	if next.NetworkRank != 32 || next.Optimizer != "AdamW8bit" || next.LearningRate != "1e-4" {
		t.Fatalf("unexpected Anima defaults: %+v", next)
	}
	if next.TrainBatchSize != 1 || next.GradientAccumulationSteps != 1 {
		t.Fatalf("expected safe batch defaults, got batch=%d grad=%d", next.TrainBatchSize, next.GradientAccumulationSteps)
	}
}

func TestApplyStableDefaultsSDXLTypicalDataset(t *testing.T) {
	dir := testImageDataset(t, 60)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureSDXL
	settings.DatasetPath = dir

	next, message := applyStableDefaults(settings)

	if next.TrainingSteps != 1800 {
		t.Fatalf("expected SDXL 1800 steps, got %d (%s)", next.TrainingSteps, message)
	}
	if next.NetworkRank != 32 || next.Optimizer != "AdamW8bit" || next.UNetLR != "1e-4" || next.TextEncoderLR1 != "1e-5" {
		t.Fatalf("unexpected SDXL defaults: %+v", next)
	}
	if !next.TrainUNetOnly {
		t.Fatalf("expected UNet-only SDXL default")
	}
}

func TestApplyStableDefaultsLargeDatasetUsesGradAccum(t *testing.T) {
	dir := testImageDataset(t, 90)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureSDXL
	settings.DatasetPath = dir

	next, _ := applyStableDefaults(settings)
	if next.GradientAccumulationSteps != 2 {
		t.Fatalf("expected grad accumulation for larger dataset, got %d", next.GradientAccumulationSteps)
	}
	if next.TrainingSteps != 1350 {
		t.Fatalf("expected large dataset SDXL steps to account for effective batch, got %d", next.TrainingSteps)
	}
}

func testImageDataset(t *testing.T, count int) string {
	t.Helper()
	dir := t.TempDir()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for i := 0; i < count; i++ {
		path := filepath.Join(dir, strconv.Itoa(i)+".png")
		file, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(file, img); err != nil {
			_ = file.Close()
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestCreateSDXLTrainingTOML(t *testing.T) {
	dir := t.TempDir()
	checkpoint := filepath.Join(dir, "illustrious.safetensors")
	if err := os.WriteFile(checkpoint, []byte("checkpoint"), 0644); err != nil {
		t.Fatal(err)
	}
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureSDXL
	settings.CheckpointPath = checkpoint
	settings.LearningRate = "1e-4"
	settings.Optimizer = "AdamW8bit"
	profile := profileFor(settings)

	path, err := createTrainingTOML("sdxl", settings, profile, dir, filepath.Join(dir, "prompts.txt"), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, expected := range []string{
		"pretrained_model_name_or_path = ",
		"network_module = \"networks.lora\"",
		"network_train_unet_only = true",
		"cache_text_encoder_outputs = true",
		"sdpa = true",
		"max_token_length = 225",
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("expected %q in SDXL TOML, got:\n%s", expected, text)
		}
	}
	if strings.Contains(text, "qwen3 = ") || strings.Contains(text, "lora_anima") {
		t.Fatalf("SDXL TOML contains Anima-only settings:\n%s", text)
	}
}

func TestCreateSDXLDatasetTOMLUsesBucketStep32(t *testing.T) {
	dir := t.TempDir()
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureSDXL
	settings.DatasetPath = dir
	profile := profileFor(settings)

	path, err := createDatasetTOML("sdxl", settings, profile, 1024, 1536, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, "bucket_reso_steps = 32") {
		t.Fatalf("expected SDXL bucket step 32, got:\n%s", text)
	}
	if strings.Contains(text, "caption_dropout_rate") {
		t.Fatalf("SDXL dataset TOML should not include caption dropout when text-encoder caching is enabled:\n%s", text)
	}
}

func TestProjectNameIsSeparateFromTrigger(t *testing.T) {
	settings := DefaultSettings(t.TempDir())
	settings.ProjectName = "Stable Output Name"
	settings.TriggerWord = "rare_trigger"

	if got := projectNameForSettings(settings); got != "Stable_Output_Name" {
		t.Fatalf("expected sanitized project name, got %q", got)
	}
}

func TestAutoTriggerControlsCaptionPrefix(t *testing.T) {
	dir := t.TempDir()
	settings := DefaultSettings(dir)
	settings.ProjectName = "project"
	settings.TriggerWord = "rare_trigger"
	settings.DatasetPath = dir
	profile := profileFor(settings)

	path, err := createDatasetTOML("project", settings, profile, 1024, 1536, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "caption_prefix = \"rare_trigger, \"") {
		t.Fatalf("expected trigger caption prefix, got:\n%s", data)
	}

	settings.AutoTrigger = false
	path, err = createDatasetTOML("project_no_trigger", settings, profile, 1024, 1536, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "caption_prefix = \"\"") {
		t.Fatalf("expected empty caption prefix with auto trigger disabled, got:\n%s", data)
	}
}

func TestNormalizeSettingsProdigyLearningRate(t *testing.T) {
	settings := DefaultSettings(t.TempDir())
	settings.Optimizer = "Prodigy"
	settings.LearningRate = "1e-4"

	next := normalizeSettings(settings)
	if next.LearningRate != "1.0" {
		t.Fatalf("expected Prodigy learning rate 1.0, got %q", next.LearningRate)
	}

	next.Optimizer = "AdamW8bit"
	next = normalizeSettings(next)
	if next.LearningRate != "1e-4" {
		t.Fatalf("expected AdamW8bit learning rate 1e-4 after Prodigy, got %q", next.LearningRate)
	}
}
