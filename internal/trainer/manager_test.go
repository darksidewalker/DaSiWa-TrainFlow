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
