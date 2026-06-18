package trainer

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScanLogChunkSplitsProgressCarriageReturns(t *testing.T) {
	scanner := bufio.NewScanner(strings.NewReader("8%| step 199\r8%| step 200\nsample saved"))
	scanner.Split(scanLogChunk)

	var got []string
	for scanner.Scan() {
		got = append(got, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}

	want := []string{"8%| step 199", "8%| step 200", "sample saved"}
	if len(got) != len(want) {
		t.Fatalf("got %d chunks %q, want %d %q", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("chunk %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestAppendTrainingLogReplacesProgressLine(t *testing.T) {
	m := &Manager{hub: NewHub()}

	m.appendTrainingLog("8%| step 199 [12:16<2:15:48, 3.70s/it]")
	m.appendTrainingLog("8%| step 200 [12:20<2:15:47, 3.70s/it]")
	m.appendTrainingLog("2026-06-13 13:46:36 INFO Generating sample images")

	got := strings.Join(m.logLines, "\n")
	want := "8%| step 200 [12:20<2:15:47, 3.70s/it]\n2026-06-13 13:46:36 INFO Generating sample images"
	if got != want {
		t.Fatalf("logs = %q, want %q", got, want)
	}
}

func TestSampleStepFromName(t *testing.T) {
	got := sampleStepFromName("untitled_001100_00_20260613144549_42.png")
	if got != 1100 {
		t.Fatalf("step = %d, want 1100", got)
	}
}

func TestStartDatasetPrepWritesMusubiDatasetTOML(t *testing.T) {
	root := t.TempDir()
	dataset := filepath.Join(root, "videos")
	if err := os.MkdirAll(dataset, 0755); err != nil {
		t.Fatal(err)
	}
	m := NewManager(root, NewHub())
	settings := DefaultSettings(root)
	settings.Architecture = ArchitectureLTX23
	settings.ProjectName = "video_project"
	settings.DatasetPath = dataset
	settings.TrainBatchSize = 1

	resp, err := m.StartDatasetPrep("musubi-dataset-toml", settings)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.OK {
		t.Fatalf("expected response ok, got %#v", resp)
	}
	if resp.PreparedPath == "" {
		t.Fatal("expected generated TOML path in response")
	}
	data, err := os.ReadFile(resp.PreparedPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "video_directory = ") {
		t.Fatalf("expected Musubi dataset TOML, got:\n%s", data)
	}
}

func TestBuildNormalizeVideoCommandUsesConfigurableFlags(t *testing.T) {
	root := t.TempDir()
	settings := DefaultSettings(root)
	settings.ProjectName = "video_project"
	settings.DatasetPath = filepath.Join(root, "raw")
	settings.VideoWidth = 1024
	settings.VideoHeight = 576
	settings.VideoFPS = 25
	settings.VideoDuration = "5.16"
	settings.VideoCodec = "hevc_nvenc"
	settings.VideoQuality = "19"
	settings.VideoEncoderPreset = "p6"
	settings.VideoSpeed = "1.5"
	settings.VideoSkipFrames = 10
	settings.VideoIncludeAudio = true
	settings.VideoExtraArgs = "-pix_fmt yuv420p"

	cmd := buildNormalizeVideoCommand(root, settings, filepath.Join(settings.DatasetPath, "clip.mp4"), filepath.Join(root, "training", "prepared", "video_project-video", "clip.mp4"))
	joined := strings.Join(cmd.Args, " ")
	for _, want := range []string{"-vf", "select='gte(n\\,10)',setpts=PTS/1.5", "fps=25", "pad=1024:576", "-t 5.16", "-c:v hevc_nvenc", "-cq 19", "-preset p6", "-pix_fmt yuv420p"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("normalize command missing %q in %q", want, joined)
		}
	}
	if strings.Contains(joined, " -an ") {
		t.Fatalf("audio should be preserved when VideoIncludeAudio=true: %q", joined)
	}
}

func TestSaveSettingsWritesMusubiDatasetTOML(t *testing.T) {
	root := t.TempDir()
	dataset := filepath.Join(root, "videos")
	if err := os.MkdirAll(dataset, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataset, "clip.mp4"), []byte("video"), 0644); err != nil {
		t.Fatal(err)
	}
	m := NewManager(root, NewHub())
	settings := DefaultSettings(root)
	settings.Architecture = ArchitectureLTX23
	settings.ProjectName = "video_project"
	settings.DatasetPath = dataset

	if err := m.SaveSettings(settings); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "training", "output", "video_project", "configs", "video_project_musubi_dataset.toml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "video_directory = ") {
		t.Fatalf("expected auto-written Musubi TOML, got:\n%s", data)
	}
}

func TestMusubiOutputPathDrivesTOMLAndCheckpointDir(t *testing.T) {
	root := t.TempDir()
	dataset := filepath.Join(root, "videos")
	if err := os.MkdirAll(dataset, 0755); err != nil {
		t.Fatal(err)
	}
	m := NewManager(root, NewHub())
	settings := DefaultSettings(root)
	settings.Architecture = ArchitectureLTX23
	settings.ProjectName = "video_project"
	settings.DatasetPath = dataset
	settings.OutputPath = filepath.Join(root, "custom-scope-output")

	if err := m.SaveSettings(settings); err != nil {
		t.Fatal(err)
	}
	tomlPath := filepath.Join(settings.OutputPath, "configs", "video_project_musubi_dataset.toml")
	if _, err := os.Stat(tomlPath); err != nil {
		t.Fatalf("expected TOML under custom output path: %v", err)
	}

	datasetTOML, err := createMusubiDatasetTOML("video_project", settings, profileFor(settings), filepath.Join(settings.OutputPath, "configs"))
	if err != nil {
		t.Fatal(err)
	}
	cmd, err := buildMusubiCommand(root, musubiCommandTrain, settings, datasetTOML, outputProject(root, settings))
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(cmd.Args, " ")
	if !strings.Contains(joined, "--output_dir "+settings.OutputPath) {
		t.Fatalf("expected training output_dir to use custom output path, got %q", joined)
	}
}

func TestCreateTrainingBootstrapAddsSdScriptsToSysPath(t *testing.T) {
	root := t.TempDir()
	trainDir := filepath.Join(root, "training", "sd-scripts")
	configDir := filepath.Join(root, "training", "output", "project", "configs")
	trainScript := filepath.Join(trainDir, "sdxl_train_network.py")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		t.Fatal(err)
	}

	bootstrapPath, err := createTrainingBootstrap(trainDir, trainScript, configDir)
	if err != nil {
		t.Fatal(err)
	}
	if bootstrapPath != filepath.Join(configDir, "trainflow_train_bootstrap.py") {
		t.Fatalf("bootstrap path = %q, want file in config dir", bootstrapPath)
	}

	data, err := os.ReadFile(bootstrapPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	checks := []string{
		"if train_dir not in sys.path:",
		"sys.path.insert(0, train_dir)",
		"os.chdir(train_dir)",
		"sys.argv[0] = train_script",
		"runpy.run_path(train_script, run_name=\"__main__\")",
	}
	for _, check := range checks {
		if !strings.Contains(content, check) {
			t.Fatalf("bootstrap content missing %q:\n%s", check, content)
		}
	}
}
