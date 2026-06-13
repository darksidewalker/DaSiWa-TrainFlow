package trainer

import (
	"bufio"
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
