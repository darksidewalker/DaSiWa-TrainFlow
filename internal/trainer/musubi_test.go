package trainer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMusubiRootUsesTrainingSubfolder(t *testing.T) {
	root := t.TempDir()
	want := filepath.Join(root, "training", "musubi-tuner")
	if got := musubiRoot(root); got != want {
		t.Fatalf("musubi root = %q, want %q", got, want)
	}
}

func TestValidateMusubiSourceFilesReportsMissingSource(t *testing.T) {
	err := validateMusubiSource(t.TempDir())
	if err == nil {
		t.Fatal("expected missing musubi source error")
	}
	if !strings.Contains(err.Error(), "training/musubi-tuner") || !strings.Contains(err.Error(), "sync-musubi-tuner.sh") {
		t.Fatalf("missing source error should be actionable, got: %v", err)
	}
}

func TestProfileForVideoArchitectures(t *testing.T) {
	ltx := profileFor(Settings{Architecture: ArchitectureLTX23})
	if ltx.Family != trainingFamilyMusubi || !ltx.Video || ltx.Script != "ltx2_train_network.py" {
		t.Fatalf("bad LTX23 profile: %#v", ltx)
	}
	wan := profileFor(Settings{Architecture: ArchitectureWAN22})
	if wan.Family != trainingFamilyMusubi || !wan.Video || wan.Script != "wan_train_network.py" {
		t.Fatalf("bad WAN22 profile: %#v", wan)
	}
	krea := profileFor(Settings{Architecture: ArchitectureKrea2})
	if krea.Family != trainingFamilyMusubi || krea.Video || krea.Script != "krea2_train_network.py" {
		t.Fatalf("bad Krea2 profile: %#v", krea)
	}
}

func TestBuildMusubiCommandsUseVendoredSourceAndSharedPython(t *testing.T) {
	root := t.TempDir()
	settings := normalizeSettings(Settings{
		Architecture:   ArchitectureLTX23,
		CheckpointPath: "/models/ltx.safetensors",
		QwenPath:       "/models/gemma.safetensors",
		ProjectName:    "ltx_project",
	})
	cmd, err := buildMusubiCommand(root, musubiCommandTrain, settings, "/tmp/dataset.toml", "/tmp/out")
	if err != nil {
		t.Fatal(err)
	}
	if cmd.Dir != filepath.Join(root, "training", "musubi-tuner") {
		t.Fatalf("command dir = %q", cmd.Dir)
	}
	if strings.Contains(cmd.Program, ".venv") {
		t.Fatalf("command program must use shared runtime, got %q", cmd.Program)
	}
	joined := strings.Join(cmd.Args, " ")
	for _, want := range []string{"-m accelerate.commands.launch", "ltx2_train_network.py", "--network_module networks.lora_ltx2", "--fp8_base", "--dataset_config /tmp/dataset.toml"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("LTX train command missing %q in %q", want, joined)
		}
	}
}

func TestBuildMusubiWAN22Command(t *testing.T) {
	settings := normalizeSettings(Settings{Architecture: ArchitectureWAN22, DiTPath: "/models/wan.safetensors", QwenPath: "/models/t5.pth", VAEPath: "/models/vae.safetensors", ProjectName: "wan_project"})
	cmd, err := buildMusubiCommand(t.TempDir(), musubiCommandTrain, settings, "/tmp/dataset.toml", "/tmp/out")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(cmd.Args, " ")
	for _, want := range []string{"wan_train_network.py", "--task i2v-A14B", "--dit /models/wan.safetensors", "--t5 /models/t5.pth", "--vae /models/vae.safetensors", "--network_module networks.lora_wan"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("WAN train command missing %q in %q", want, joined)
		}
	}
}

func TestBuildMusubiKrea2Commands(t *testing.T) {
	settings := normalizeSettings(Settings{Architecture: ArchitectureKrea2, DiTPath: "/models/krea2-raw.safetensors", QwenPath: "/models/qwen3-vl.safetensors", VAEPath: "/models/qwen-image-vae.safetensors", ProjectName: "krea_project", TriggerWord: "trigger"})
	settings.ExtraTrainArgs = appendMusubiSamplingExtraArgs(settings.ExtraTrainArgs, "/tmp/krea_project_prompts.txt", 250)
	cmd, err := buildMusubiCommand(t.TempDir(), musubiCommandTrain, settings, "/tmp/dataset.toml", "/tmp/out")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(cmd.Args, " ")
	for _, want := range []string{"--num_processes=1", "--num_machines=1", "--dynamo_backend=no", "src/musubi_tuner/krea2_train_network.py", "--dit /models/krea2-raw.safetensors", "--text_encoder /models/qwen3-vl.safetensors", "--vae /models/qwen-image-vae.safetensors", "--network_module networks.lora_krea2", "--fp8_scaled", "--sample_every_n_steps 250", "--sample_prompts /tmp/krea_project_prompts.txt"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("Krea2 train command missing %q in %q", want, joined)
		}
	}
	if strings.Contains(joined, "--metadata_trigger_phrase") {
		t.Fatalf("Musubi train command must not pass unsupported --metadata_trigger_phrase: %q", joined)
	}
}

func TestKrea2SamplePromptsUseRawRecommendedSteps(t *testing.T) {
	dir := t.TempDir()
	settings := normalizeSettings(Settings{
		Architecture:   ArchitectureKrea2,
		ProjectName:    "krea_project",
		TriggerWord:    "trigger",
		SamplePrompts:  []string{"portrait photo"},
		NegativePrompt: "blurry",
		Width:          1024,
		Height:         1024,
		SampleStepsGen: 30,
		SampleCFG:      4,
		SampleSeed:     42,
	})
	path, err := createSamplePrompts("krea_project", settings, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, "--s 52") {
		t.Fatalf("Krea2 RAW sample prompts must use 52 steps, got: %q", text)
	}
}

func TestBuildMusubiKrea2LatentCacheOmitsVAEDtype(t *testing.T) {
	settings := normalizeSettings(Settings{Architecture: ArchitectureKrea2, VAEPath: "/models/qwen-image-vae.safetensors", MixedPrecision: "bf16"})
	cmd, err := buildMusubiCommand(t.TempDir(), musubiCommandCacheLatents, settings, "/tmp/dataset.toml", "/tmp/out")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(cmd.Args, " ")
	if strings.Contains(joined, "--vae_dtype") {
		t.Fatalf("Krea2 latent cache must not pass --vae_dtype: %q", joined)
	}
	for _, want := range []string{"src/musubi_tuner/krea2_cache_latents.py", "--dataset_config /tmp/dataset.toml", "--vae /models/qwen-image-vae.safetensors"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("Krea2 latent cache command missing %q in %q", want, joined)
		}
	}
}

func TestCreateMusubiKrea2DatasetTOML(t *testing.T) {
	dir := t.TempDir()
	settings := normalizeSettings(Settings{Architecture: ArchitectureKrea2, DatasetPath: dir, TrainBatchSize: 1})
	path, err := createMusubiDatasetTOML("krea", settings, profileFor(settings), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{"[general]", "resolution = [1024, 1024]", "image_directory = ", "cache_directory = ", "enable_bucket = true"} {
		if !strings.Contains(text, want) {
			t.Fatalf("Krea2 dataset TOML missing %q:\n%s", want, text)
		}
	}
}

func TestCreateMusubiDatasetTOML(t *testing.T) {
	dir := t.TempDir()
	settings := normalizeSettings(Settings{Architecture: ArchitectureLTX23, DatasetPath: dir, TrainBatchSize: 1, VideoWidth: 768, VideoHeight: 512})
	path, err := createMusubiDatasetTOML("video", settings, profileFor(settings), dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{"[general]", "resolution = [768, 512]", "video_directory = ", "target_frames = [1, 65, 129]", "frame_extraction = \"full\""} {
		if !strings.Contains(text, want) {
			t.Fatalf("dataset TOML missing %q:\n%s", want, text)
		}
	}
}
