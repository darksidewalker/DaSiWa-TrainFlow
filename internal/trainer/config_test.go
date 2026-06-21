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
	if next.NetworkRank != 32 || next.Optimizer != "Prodigy" || next.LearningRate != "1.0" {
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
	if next.NetworkRank != 32 || next.Optimizer != "Prodigy" || next.UNetLR != "1e-4" || next.TextEncoderLR1 != "1e-5" {
		t.Fatalf("unexpected SDXL defaults: %+v", next)
	}
	if !next.TrainUNetOnly {
		t.Fatalf("expected UNet-only SDXL default")
	}
}

func TestApplyStableDefaultsPreservesProdigySchedulerMath(t *testing.T) {
	dir := testImageDataset(t, 60)
	settings := DefaultSettings(dir)
	settings.DatasetPath = dir
	settings.Optimizer = "Prodigy"
	settings.LearningRate = "1e-4"

	next, _ := applyStableDefaults(settings)
	if next.Optimizer != "Prodigy" {
		t.Fatalf("expected Auto Calc to preserve optimizer, got %q", next.Optimizer)
	}
	if next.LearningRate != "1.0" {
		t.Fatalf("expected Prodigy learning rate math, got %q", next.LearningRate)
	}

	profile := profileFor(next)
	path, err := createTrainingTOML("prodigy", next, profile, dir, filepath.Join(dir, "prompts.txt"), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "lr_scheduler = \"constant\"") {
		t.Fatalf("expected Prodigy scheduler to remain constant, got:\n%s", data)
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

func TestApplyStableDefaultsUsesVRAMTargetForBatch(t *testing.T) {
	dir := testImageDataset(t, 60)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureSDXL
	settings.DatasetPath = dir
	settings.TargetVRAMPercent = 90

	next, message := applyStableDefaultsWithVRAM(settings, 24576)
	if next.TrainBatchSize != 2 {
		t.Fatalf("expected 24 GB SDXL target to raise batch to 2, got %d (%s)", next.TrainBatchSize, message)
	}
	if next.TrainingSteps != 1200 {
		t.Fatalf("expected SDXL min steps after larger batch, got %d", next.TrainingSteps)
	}
	if !strings.Contains(message, "VRAM target: 90% of 24576 MB") {
		t.Fatalf("expected VRAM target in message, got %q", message)
	}
}

func TestApplyStableDefaultsLowerVRAMTargetKeepsSaferBatch(t *testing.T) {
	dir := testImageDataset(t, 60)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureSDXL
	settings.DatasetPath = dir
	settings.TargetVRAMPercent = 70

	next, _ := applyStableDefaultsWithVRAM(settings, 24576)
	if next.TrainBatchSize != 1 {
		t.Fatalf("expected lower VRAM target to keep batch 1, got %d", next.TrainBatchSize)
	}
}

func TestNormalizeSettingsDefaultsAndClampsTargetVRAM(t *testing.T) {
	settings := DefaultSettings(t.TempDir())
	settings.TargetVRAMPercent = 0
	if got := normalizeSettings(settings).TargetVRAMPercent; got != 90 {
		t.Fatalf("default target VRAM = %d, want 90", got)
	}
	settings.TargetVRAMPercent = 120
	if got := normalizeSettings(settings).TargetVRAMPercent; got != 98 {
		t.Fatalf("high target VRAM clamp = %d, want 98", got)
	}
	settings.TargetVRAMPercent = 10
	if got := normalizeSettings(settings).TargetVRAMPercent; got != 50 {
		t.Fatalf("low target VRAM clamp = %d, want 50", got)
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

func TestApplyStableDefaultsLTX23SetsVideoRepeats(t *testing.T) {
	dir := testVideoDataset(t, 10)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureLTX23
	settings.DatasetPath = dir

	next, message := applyStableDefaults(settings)

	if next.Architecture != ArchitectureLTX23 {
		t.Fatalf("expected LTX 2.3 architecture, got %q", next.Architecture)
	}
	if next.VideoNumRepeats != 53 {
		t.Fatalf("expected VideoNumRepeats=53 for LTX 2.3, got %d (%s)", next.VideoNumRepeats, message)
	}
	if next.TargetEpochs != 6 {
		t.Fatalf("expected TargetEpochs=6 for LTX 2.3, got %d", next.TargetEpochs)
	}
	if next.TrainingSteps != 3180 {
		t.Fatalf("expected estimated optimizer steps=3180 for LTX 2.3, got %d (%s)", next.TrainingSteps, message)
	}
	if next.NetworkRank != 128 || next.Optimizer != "Prodigy" || next.LearningRate != "1.0" {
		t.Fatalf("unexpected LTX 2.3 defaults: %+v", next)
	}
	if !strings.Contains(message, "10 videos") || !strings.Contains(message, "effective steps") {
		t.Fatalf("expected video message, got %q", message)
	}
}

func TestApplyStableDefaultsWAN22SetsVideoRepeats(t *testing.T) {
	dir := testVideoDataset(t, 15)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureWAN22
	settings.DatasetPath = dir

	next, message := applyStableDefaults(settings)

	if next.Architecture != ArchitectureWAN22 {
		t.Fatalf("expected Wan 2.2 architecture, got %q", next.Architecture)
	}
	if next.VideoNumRepeats != 36 {
		t.Fatalf("expected VideoNumRepeats=36 for Wan 2.2, got %d (%s)", next.VideoNumRepeats, message)
	}
	if next.TargetEpochs != 6 {
		t.Fatalf("expected TargetEpochs=6 for Wan 2.2, got %d", next.TargetEpochs)
	}
	if next.TrainingSteps != 3240 {
		t.Fatalf("expected estimated optimizer steps=3240 for Wan 2.2, got %d (%s)", next.TrainingSteps, message)
	}
	if next.NetworkRank != 128 || next.Optimizer != "Prodigy" || next.LearningRate != "1.0" {
		t.Fatalf("unexpected Wan 2.2 defaults: %+v", next)
	}
	if !strings.Contains(message, "15 videos") || !strings.Contains(message, "effective steps") {
		t.Fatalf("expected video message, got %q", message)
	}
}

func TestApplyStableDefaultsVideoPreservesUserEpochs(t *testing.T) {
	dir := testVideoDataset(t, 10)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureLTX23
	settings.DatasetPath = dir
	settings.TargetEpochs = 12

	next, _ := applyStableDefaults(settings)

	if next.TargetEpochs != 12 {
		t.Fatalf("expected user TargetEpochs preserved, got %d", next.TargetEpochs)
	}
	if next.VideoNumRepeats != 27 {
		t.Fatalf("expected VideoNumRepeats=27, got %d", next.VideoNumRepeats)
	}
	if next.TrainingSteps != 3240 {
		t.Fatalf("expected estimated optimizer steps=3240, got %d", next.TrainingSteps)
	}
}

func TestApplyStableDefaultsVideoUsesGradAccumForHighVRAM(t *testing.T) {
	dir := testVideoDataset(t, 72)
	settings := DefaultSettings(dir)
	settings.Architecture = ArchitectureLTX23
	settings.DatasetPath = dir
	settings.TargetVRAMPercent = 90

	next, message := applyStableDefaultsWithVRAM(settings, 32607)

	if next.TrainBatchSize != 1 {
		t.Fatalf("expected Musubi video auto calc to keep batch size 1, got %d", next.TrainBatchSize)
	}
	if next.GradientAccumulationSteps != 4 {
		t.Fatalf("expected high VRAM video auto calc to use grad accumulation 4, got %d", next.GradientAccumulationSteps)
	}
	if next.VideoNumRepeats != 7 {
		t.Fatalf("expected VideoNumRepeats=7 for 72-video dataset, got %d (%s)", next.VideoNumRepeats, message)
	}
	if next.TrainingSteps != 756 {
		t.Fatalf("expected optimizer steps=756 after grad accumulation, got %d (%s)", next.TrainingSteps, message)
	}
	if next.SaveSteps != 250 || next.SampleSteps != 250 {
		t.Fatalf("expected video save/sample interval 250, got save=%d sample=%d", next.SaveSteps, next.SampleSteps)
	}
	if !strings.Contains(message, "3024 effective steps") || !strings.Contains(message, "grad 4") {
		t.Fatalf("expected effective-step video message, got %q", message)
	}
}

func testVideoDataset(t *testing.T, count int) string {
	t.Helper()
	dir := t.TempDir()
	for i := 0; i < count; i++ {
		path := filepath.Join(dir, strconv.Itoa(i)+".mp4")
		if err := os.WriteFile(path, []byte("fake video data"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}
