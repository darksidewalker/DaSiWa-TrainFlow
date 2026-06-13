package runtimeops

import "testing"

func TestFlashAttentionWheelURLUsesCUDAAndTorchABI(t *testing.T) {
	info := torchCUDAInfo{
		CUDA:      "13.0",
		Torch:     "2.9",
		PythonTag: "cp312",
		Platform:  "linux_x86_64",
		CXX11ABI:  "FALSE",
	}

	url, wheel := flashAttentionWheelURL("2.8.3", info)
	wantWheel := "flash_attn-2.8.3+cu13torch2.9cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
	if wheel != wantWheel {
		t.Fatalf("wheel = %q, want %q", wheel, wantWheel)
	}
	wantURL := "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/" + wantWheel
	if url != wantURL {
		t.Fatalf("url = %q, want %q", url, wantURL)
	}
}

func TestCUDA13IsDetected(t *testing.T) {
	for _, version := range []string{"13.0", "13.1", " 13.0"} {
		if !isCUDA13(version) {
			t.Fatalf("expected %q to be treated as CUDA 13", version)
		}
	}
	for _, version := range []string{"12.8", "11.8", ""} {
		if isCUDA13(version) {
			t.Fatalf("did not expect %q to be treated as CUDA 13", version)
		}
	}
}

func TestTorchCUDAArchListSupportsThreeDigitArchitectures(t *testing.T) {
	got := torchCUDAArchList([]string{"89", "120", "89", ""})
	want := "8.9;12.0"
	if got != want {
		t.Fatalf("arch list = %q, want %q", got, want)
	}
}
