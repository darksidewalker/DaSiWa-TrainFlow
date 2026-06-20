package hwmon

import (
	"os"
	"path/filepath"
	"testing"
)

func TestClamp(t *testing.T) {
	tests := []struct {
		v, min, max, want int
	}{
		{50, 0, 100, 50},
		{-10, 0, 100, 0},
		{150, 0, 100, 100},
		{0, 0, 0, 0},
		{5, 5, 5, 5},
	}
	for _, tt := range tests {
		got := clamp(tt.v, tt.min, tt.max)
		if got != tt.want {
			t.Errorf("clamp(%d, %d, %d) = %d, want %d", tt.v, tt.min, tt.max, got, tt.want)
		}
	}
}

func TestReadCPUTimes(t *testing.T) {
	// Create a temporary /proc/stat mock
	tmpDir := t.TempDir()
	testStat := filepath.Join(tmpDir, "stat")
	if err := os.WriteFile(testStat, []byte("cpu  1000 200 300 400 50 60 70 80\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Test with real /proc/stat if available
	ct := readCPUTimes()
	if ct.total == 0 {
		// If /proc/stat doesn't exist or is unreadable, skip
		t.Skip("No /proc/stat available")
	}
	if ct.idle > ct.total {
		t.Errorf("idle (%d) > total (%d)", ct.idle, ct.total)
	}
}

func TestRAMStats(t *testing.T) {
	rs := ramStats()
	if rs.Total == 0 {
		t.Skip("No RAM stats available")
	}
	if rs.Used > rs.Total {
		t.Errorf("used (%d) > total (%d)", rs.Used, rs.Total)
	}
}

func TestNew(t *testing.T) {
	m := New()
	if m == nil {
		t.Fatal("New() returned nil")
	}
}

func TestSnapshot(t *testing.T) {
	m := New()
	s := m.Snapshot(nil)
	if s.CPU < 0 || s.CPU > 100 {
		t.Errorf("CPU usage out of range: %d", s.CPU)
	}
}
