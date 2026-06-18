package trainer

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"trainflow/internal/process"
)

var validVideoExtensions = map[string]bool{
	".mp4":  true,
	".mkv":  true,
	".mov":  true,
	".webm": true,
	".avi":  true,
	".m4v":  true,
}

func validVideoExt(name string) bool {
	return validVideoExtensions[strings.ToLower(filepath.Ext(name))]
}

func preparedVideoDatasetPath(datasetPath string) string {
	return filepath.Join(datasetPath, "input")
}

func listDatasetVideos(datasetPath string) ([]string, error) {
	var videos []string
	err := filepath.WalkDir(datasetPath, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !validVideoExt(entry.Name()) {
			return nil
		}
		videos = append(videos, path)
		return nil
	})
	return videos, err
}

func buildNormalizeVideoCommand(root string, s Settings, inputPath, outputPath string) musubiCommand {
	width := defaultInt(s.VideoWidth, 768)
	height := defaultInt(s.VideoHeight, 512)
	fps := defaultInt(s.VideoFPS, 24)
	filters := []string{}
	if s.VideoSkipFrames > 0 {
		filters = append(filters, fmt.Sprintf("select='gte(n\\,%d)'", s.VideoSkipFrames))
	}
	if strings.TrimSpace(s.VideoSpeed) != "" && strings.TrimSpace(s.VideoSpeed) != "1" && strings.TrimSpace(s.VideoSpeed) != "1.0" {
		filters = append(filters, "setpts=PTS/"+strings.TrimSpace(s.VideoSpeed))
	}
	filters = append(filters,
		fmt.Sprintf("fps=%d", fps),
		fmt.Sprintf("scale=w=%d:h=%d:force_original_aspect_ratio=decrease", width, height),
		fmt.Sprintf("pad=%d:%d:(ow-iw)/2:(oh-ih)/2", width, height),
	)
	args := []string{"-y", "-i", inputPath, "-vf", strings.Join(filters, ",")}
	if strings.TrimSpace(s.VideoDuration) != "" {
		args = append(args, "-t", strings.TrimSpace(s.VideoDuration))
	}
	if !s.VideoIncludeAudio {
		args = append(args, "-an")
	}
	codec := nonEmpty(s.VideoCodec, "libx264")
	quality := nonEmpty(s.VideoQuality, "19")
	preset := nonEmpty(s.VideoEncoderPreset, "medium")
	args = append(args, "-c:v", codec)
	if strings.Contains(codec, "nvenc") {
		args = append(args, "-cq", quality)
	} else {
		args = append(args, "-crf", quality)
	}
	args = append(args, "-preset", preset)
	args = appendFields(args, s.VideoExtraArgs)
	args = append(args, outputPath)
	return musubiCommand{Program: "ffmpeg", Args: args, Dir: root}
}

func (m *Manager) runVideoNormalization(s Settings, outputDir string) {
	defer func() {
		m.mu.Lock()
		m.running = false
		m.trainingCmd = nil
		m.activeGPUs = map[string]string{}
		m.mu.Unlock()
		m.appendLog("Process finished.")
	}()
	videos, err := listDatasetVideos(s.DatasetPath)
	if err != nil {
		m.appendLog("Video normalization failed: " + err.Error())
		return
	}
	if len(videos) == 0 {
		m.appendLog("Video normalization failed: no supported video files found.")
		return
	}
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		m.appendLog("Video normalization failed: " + err.Error())
		return
	}
	workers := defaultInt(s.VideoParallelWorkers, 1)
	if workers > len(videos) {
		workers = len(videos)
	}
	m.appendLog(fmt.Sprintf("Normalizing %d video(s) with %d worker(s).", len(videos), workers))

	jobs := make(chan string)
	errCh := make(chan error, 1)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for input := range jobs {
				output := filepath.Join(outputDir, strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))+".mp4")
				spec := buildNormalizeVideoCommand(m.root, s, input, output)
				m.appendLog(fmt.Sprintf("[video worker %d] Normalizing: %s", workerID+1, input))
				cmd := exec.Command(spec.Program, spec.Args...)
				cmd.Dir = spec.Dir
				processPrepare(cmd)
				data, err := cmd.CombinedOutput()
				for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
					if strings.TrimSpace(line) != "" {
						m.appendTrainingLog(strings.TrimSpace(line))
					}
				}
				if err != nil {
					select {
					case errCh <- fmt.Errorf("%s: %w", input, err):
					default:
					}
					return
				}
				copyCaptionForVideo(input, output)
			}
		}(i)
	}
	go func() {
		defer close(jobs)
		for _, input := range videos {
			jobs <- input
		}
	}()
	wg.Wait()
	select {
	case err := <-errCh:
		m.appendLog("Video normalization failed: " + err.Error())
		return
	default:
	}
	m.appendLog("Normalized videos written to: " + outputDir)
}

func copyCaptionForVideo(inputVideo, outputVideo string) {
	inputCaption := strings.TrimSuffix(inputVideo, filepath.Ext(inputVideo)) + ".txt"
	data, err := os.ReadFile(inputCaption)
	if err != nil {
		return
	}
	_ = os.WriteFile(strings.TrimSuffix(outputVideo, filepath.Ext(outputVideo))+".txt", data, 0644)
}

func processPrepare(cmd *exec.Cmd) {
	process.Prepare(cmd)
}
