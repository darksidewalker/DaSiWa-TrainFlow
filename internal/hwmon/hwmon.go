package hwmon

import (
	"bufio"
	"bytes"
	"context"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Monitor struct {
	mu      sync.Mutex
	prevCPU cpuTimes
	hasPrev bool
}

type Snapshot struct {
	CPU     int       `json:"cpu"`
	CPUTemp *int      `json:"cpuTemp"`
	RAM     RAMStats  `json:"ram"`
	GPUs    []GPUStat `json:"gpus"`
}

type RAMStats struct {
	Total uint64 `json:"total"`
	Used  uint64 `json:"used"`
}

type GPUStat struct {
	Index      int     `json:"index"`
	Name       string  `json:"name"`
	Util       int     `json:"util"`
	MemUsed    int     `json:"memUsed"`
	MemTotal   int     `json:"memTotal"`
	Temp       int     `json:"temp"`
	PowerDraw  int     `json:"powerDraw"`
	PowerLimit int     `json:"powerLimit"`
	Activity   *string `json:"activity"`
}

type cpuTimes struct {
	total uint64
	idle  uint64
}

func New() *Monitor {
	return &Monitor{}
}

func (m *Monitor) Snapshot(active map[string]string) Snapshot {
	m.mu.Lock()
	cpuPct := m.cpuUsageLocked()
	m.mu.Unlock()

	gpus := gpuStats()
	for i := range gpus {
		if activity, ok := active[strconv.Itoa(gpus[i].Index)]; ok {
			v := activity
			gpus[i].Activity = &v
		}
	}
	return Snapshot{
		CPU:     cpuPct,
		CPUTemp: cpuTemp(),
		RAM:     ramStats(),
		GPUs:    gpus,
	}
}

func (m *Monitor) cpuUsageLocked() int {
	if runtime.GOOS == "windows" {
		return windowsCPUUsage()
	}
	current := readCPUTimes()
	if current.total == 0 {
		return 0
	}
	if !m.hasPrev {
		m.prevCPU = current
		m.hasPrev = true
		return 0
	}
	totalDelta := current.total - m.prevCPU.total
	idleDelta := current.idle - m.prevCPU.idle
	m.prevCPU = current
	if totalDelta == 0 || idleDelta > totalDelta {
		return 0
	}
	return int((100 * (totalDelta - idleDelta)) / totalDelta)
}

func readCPUTimes() cpuTimes {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return cpuTimes{}
	}
	line := strings.SplitN(string(data), "\n", 2)[0]
	fields := strings.Fields(line)
	if len(fields) < 5 || fields[0] != "cpu" {
		return cpuTimes{}
	}
	var total uint64
	var values []uint64
	for _, field := range fields[1:] {
		v, _ := strconv.ParseUint(field, 10, 64)
		values = append(values, v)
		total += v
	}
	idle := values[3]
	if len(values) > 4 {
		idle += values[4]
	}
	return cpuTimes{total: total, idle: idle}
}

func windowsCPUUsage() int {
	out := runCommand(2*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average")
	v, _ := strconv.Atoi(strings.TrimSpace(out))
	return clamp(v, 0, 100)
}

func ramStats() RAMStats {
	if runtime.GOOS == "windows" {
		out := runCommand(2*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", "$m=Get-CimInstance Win32_OperatingSystem; [string]::Format('{0},{1}', $m.TotalVisibleMemorySize*1024, ($m.TotalVisibleMemorySize-$m.FreePhysicalMemory)*1024)")
		parts := strings.Split(strings.TrimSpace(out), ",")
		if len(parts) == 2 {
			total, _ := strconv.ParseUint(strings.TrimSpace(parts[0]), 10, 64)
			used, _ := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
			return RAMStats{Total: total, Used: used}
		}
		return RAMStats{}
	}
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return RAMStats{}
	}
	defer file.Close()
	values := map[string]uint64{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		v, _ := strconv.ParseUint(fields[1], 10, 64)
		values[strings.TrimSuffix(fields[0], ":")] = v * 1024
	}
	total := values["MemTotal"]
	available := values["MemAvailable"]
	return RAMStats{Total: total, Used: total - available}
}

func cpuTemp() *int {
	if runtime.GOOS == "windows" {
		out := runCommand(3*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature")
		lines := strings.Fields(out)
		sum := 0.0
		count := 0
		for _, line := range lines {
			v, err := strconv.ParseFloat(strings.TrimSpace(line), 64)
			if err == nil && v > 0 {
				sum += v/10 - 273.15
				count++
			}
		}
		if count == 0 {
			return nil
		}
		temp := int(sum/float64(count) + 0.5)
		return &temp
	}
	entries, err := os.ReadDir("/sys/class/thermal")
	if err != nil {
		return nil
	}
	sum := 0
	count := 0
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "thermal_zone") {
			continue
		}
		data, err := os.ReadFile("/sys/class/thermal/" + entry.Name() + "/temp")
		if err != nil {
			continue
		}
		v, err := strconv.Atoi(strings.TrimSpace(string(data)))
		if err == nil && v > 0 {
			sum += v / 1000
			count++
		}
	}
	if count == 0 {
		return nil
	}
	temp := sum / count
	return &temp
}

func gpuStats() []GPUStat {
	out := runCommand(3*time.Second, "nvidia-smi",
		"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
		"--format=csv,noheader,nounits",
	)
	if strings.TrimSpace(out) == "" {
		return nil
	}
	var gpus []GPUStat
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		parts := strings.Split(line, ",")
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		if len(parts) < 8 {
			continue
		}
		index, _ := strconv.Atoi(parts[0])
		util, _ := strconv.Atoi(parts[2])
		memUsed, _ := strconv.Atoi(parts[3])
		memTotal, _ := strconv.Atoi(parts[4])
		temp, _ := strconv.Atoi(parts[5])
		powerDraw, _ := strconv.ParseFloat(parts[6], 64)
		powerLimit, _ := strconv.ParseFloat(parts[7], 64)
		gpus = append(gpus, GPUStat{
			Index:      index,
			Name:       parts[1],
			Util:       util,
			MemUsed:    memUsed,
			MemTotal:   memTotal,
			Temp:       temp,
			PowerDraw:  int(powerDraw + 0.5),
			PowerLimit: int(powerLimit + 0.5),
		})
	}
	return gpus
}

func runCommand(timeout time.Duration, name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return ""
	}
	return stdout.String()
}

func clamp(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
