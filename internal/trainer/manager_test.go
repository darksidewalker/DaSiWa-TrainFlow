package trainer

import (
	"bufio"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"os"
	"os/exec"
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

func TestDatasetResizePrepPreservesAspectAndWritesInDataset(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "from PIL import Image").Run(); err != nil {
		t.Skip("Pillow not available")
	}

	root := t.TempDir()
	dataset := filepath.Join(root, "dataset")
	if err := os.MkdirAll(dataset, 0755); err != nil {
		t.Fatal(err)
	}
	img := image.NewRGBA(image.Rect(0, 0, 800, 400))
	for y := 0; y < 400; y++ {
		for x := 0; x < 800; x++ {
			img.Set(x, y, color.RGBA{R: 128, G: 64, B: 32, A: 255})
		}
	}
	imagePath := filepath.Join(dataset, "wide.png")
	imageFile, err := os.Create(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(imageFile, img); err != nil {
		_ = imageFile.Close()
		t.Fatal(err)
	}
	if err := imageFile.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataset, "wide.txt"), []byte("caption"), 0644); err != nil {
		t.Fatal(err)
	}

	settings := DefaultSettings(root)
	settings.DatasetPath = dataset
	settings.SideMax = 240
	args := datasetResizeArgs(root, settings)
	cmd := exec.Command(python, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("resize prep failed: %v\n%s", err, out)
	}

	if _, err := os.Stat(filepath.Join(dataset, "input", "wide.png")); err != nil {
		t.Fatalf("expected original moved to input folder: %v", err)
	}
	preparedCaption := filepath.Join(dataset, "1.txt")
	if _, err := os.Stat(preparedCaption); err != nil {
		t.Fatalf("expected numbered caption in dataset folder: %v", err)
	}
	prepared, err := os.Open(filepath.Join(dataset, "1.jpg"))
	if err != nil {
		t.Fatalf("expected numbered image in dataset folder: %v", err)
	}
	decoded, _, err := image.Decode(prepared)
	_ = prepared.Close()
	if err != nil {
		t.Fatal(err)
	}
	if got := decoded.Bounds().Size(); got.X != 240 || got.Y != 120 {
		t.Fatalf("expected aspect-preserving 240x120 output, got %dx%d", got.X, got.Y)
	}
}

func TestDatasetResizePrepKeepsExactCaptionPairsWhenRenumbering(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "from PIL import Image").Run(); err != nil {
		t.Skip("Pillow not available")
	}

	root := t.TempDir()
	dataset := filepath.Join(root, "dataset")
	if err := os.MkdirAll(dataset, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestPNG(t, filepath.Join(dataset, "zebra.png"), 64, 32)
	writeTestPNG(t, filepath.Join(dataset, "apple.png"), 32, 64)
	if err := os.WriteFile(filepath.Join(dataset, "zebra.txt"), []byte("caption for zebra"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataset, "apple.txt"), []byte("caption for apple"), 0644); err != nil {
		t.Fatal(err)
	}

	settings := DefaultSettings(root)
	settings.DatasetPath = dataset
	settings.SideMax = 64
	cmd := exec.Command(python, datasetResizeArgs(root, settings)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("resize prep failed: %v\n%s", err, out)
	}

	firstCaption, err := os.ReadFile(filepath.Join(dataset, "1.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(firstCaption) != "caption for apple" {
		t.Fatalf("1.txt = %q, want apple caption matching sorted apple.png", firstCaption)
	}
	secondCaption, err := os.ReadFile(filepath.Join(dataset, "2.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(secondCaption) != "caption for zebra" {
		t.Fatalf("2.txt = %q, want zebra caption matching sorted zebra.png", secondCaption)
	}
}

func TestDatasetResizePrepRejectsMismatchedCaptionPairs(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "from PIL import Image").Run(); err != nil {
		t.Skip("Pillow not available")
	}

	root := t.TempDir()
	dataset := filepath.Join(root, "dataset")
	if err := os.MkdirAll(dataset, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestPNG(t, filepath.Join(dataset, "image.png"), 64, 64)
	if err := os.WriteFile(filepath.Join(dataset, "wrong.txt"), []byte("wrong caption"), 0644); err != nil {
		t.Fatal(err)
	}

	settings := DefaultSettings(root)
	settings.DatasetPath = dataset
	settings.SideMax = 64
	cmd := exec.Command(python, datasetResizeArgs(root, settings)...)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected resize prep to reject mismatched image/caption names; output:\n%s", out)
	}
	if !strings.Contains(string(out), "exact image/caption filename pairs") {
		t.Fatalf("expected exact-pairing error, got:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dataset, "1.txt")); !os.IsNotExist(err) {
		t.Fatalf("mismatched prep must not write numbered caption, stat err: %v", err)
	}
}

func writeTestPNG(t *testing.T, path string, width, height int) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: 128, G: 64, B: 32, A: 255})
		}
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if err := png.Encode(file, img); err != nil {
		t.Fatal(err)
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

func TestStartMusubiPipelineSkipsAutoNormalization(t *testing.T) {
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
	settings.TrainBatchSize = 1

	resp, err := m.Start(settings)
	if err != nil {
		t.Fatalf("start returned unexpected error: %v", err)
	}
	if resp.OK {
		t.Fatalf("start should fail without python runtime: got ok=true, message=%q", resp.Message)
	}
	msg := strings.ToLower(resp.Message)
	if strings.Contains(msg, "normalize") {
		t.Fatalf("start should not auto-trigger video normalization: message=%q", resp.Message)
	}
	if strings.Contains(msg, "normalize-video") {
		t.Fatalf("start should not reference normalize-video binary: message=%q", resp.Message)
	}
	if strings.Contains(msg, "ffmpeg") {
		t.Fatalf("start should not check for ffmpeg during training: message=%q", resp.Message)
	}
}
