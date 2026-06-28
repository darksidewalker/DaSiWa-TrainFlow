package hwmon

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
	Util       *int    `json:"util,omitempty"` // nil when unavailable
	MemUsed    int     `json:"memUsed"`
	MemTotal   int     `json:"memTotal"`
	Temp       *int    `json:"temp,omitempty"`       // nil when unavailable
	PowerDraw  *int    `json:"powerDraw,omitempty"`  // nil when unavailable
	PowerLimit *int    `json:"powerLimit,omitempty"` // nil when unavailable
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
	if scanner.Err() != nil {
		return RAMStats{}
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
	var all []GPUStat

	// NVIDIA via nvidia-smi
	if nvidia := nvidiaGPUStats(); nvidia != nil {
		all = append(all, nvidia...)
	}

	// AMD on Windows via WMI (basic: name + VRAM only)
	if runtime.GOOS == "windows" {
		if amd := amdGPUWindowsStats(); amd != nil {
			all = append(all, amd...)
		}
	}

	// AMD on Linux via amdgpu sysfs (works with kernel driver alone, no ROCm needed)
	if runtime.GOOS == "linux" {
		if amd := amdGPUSysfsStats(); amd != nil {
			all = append(all, amd...)
		}
	}

	return all
}

// nvidiaGPUStats queries GPU stats via nvidia-smi.
func nvidiaGPUStats() []GPUStat {
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
			Util:       ptrInt(util),
			MemUsed:    memUsed,
			MemTotal:   memTotal,
			Temp:       ptrInt(temp),
			PowerDraw:  ptrInt(int(powerDraw + 0.5)),
			PowerLimit: ptrInt(int(powerLimit + 0.5)),
		})
	}
	return gpus
}

// amdGPUSysfsStats reads GPU stats from amdgpu sysfs.
// Works on any Linux system with the amdgpu kernel driver loaded.
func amdGPUSysfsStats() []GPUStat {
	entries, err := os.ReadDir("/sys/class/drm")
	if err != nil {
		return nil
	}
	var gpus []GPUStat
	seenPCI := make(map[string]bool) // deduplicate: card0, card1-DP-1 etc all map to same GPU

	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "card") {
			continue
		}
		// card0 -> GPU, card0-DP-1 -> connector (skip)
		if strings.Count(entry.Name(), "-") > 0 {
			continue
		}

		cardDir := filepath.Join("/sys/class/drm", entry.Name(), "device")

		// Check vendor ID (0x1002 = AMD)
		vendorData, err := os.ReadFile(filepath.Join(cardDir, "vendor"))
		if err != nil {
			continue
		}
		if strings.TrimSpace(string(vendorData)) != "0x1002" {
			continue
		}

		// Deduplicate by PCI bus ID
		pciData, _ := os.ReadFile(filepath.Join(cardDir, "uevent"))
		var pciSlot string
		for _, line := range strings.Split(string(pciData), "\n") {
			if strings.HasPrefix(line, "PCI_SLOT_NAME=") {
				pciSlot = strings.TrimPrefix(line, "PCI_SLOT_NAME=")
				break
			}
		}
		if pciSlot == "" || seenPCI[pciSlot] {
			continue
		}
		seenPCI[pciSlot] = true

		// GPU name from lspci
		name := amdGPUName(cardDir)

		// GPU utilization
		util := 0
		if data, err := os.ReadFile(filepath.Join(cardDir, "gpu_busy_percent")); err == nil {
			util, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}

		// VRAM (bytes -> MB)
		memUsed := 0
		memTotal := 0
		if data, err := os.ReadFile(filepath.Join(cardDir, "mem_info_vram_used")); err == nil {
			memUsed64, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			memUsed = int(memUsed64 / (1024 * 1024))
		}
		if data, err := os.ReadFile(filepath.Join(cardDir, "mem_info_vram_total")); err == nil {
			memTotal64, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			memTotal = int(memTotal64 / (1024 * 1024))
		}

		// Temperature from hwmon (millidegrees C -> degrees C)
		temp := 0
		hwmonDir := findAMDHwmon(cardDir)
		if hwmonDir != "" {
			if data, err := os.ReadFile(filepath.Join(hwmonDir, "temp1_input")); err == nil {
				tempMilli, _ := strconv.Atoi(strings.TrimSpace(string(data)))
				temp = tempMilli / 1000
			}
		}

		// Power draw from hwmon (microwatts -> watts)
		powerDraw := 0
		powerLimit := 0
		if hwmonDir != "" {
			if data, err := os.ReadFile(filepath.Join(hwmonDir, "power1_input")); err == nil {
				pwrMicro, _ := strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
				powerDraw = int(pwrMicro/1e6 + 0.5)
			}
		}

		// Derive GPU index from PCI slot
		index := 0
		for _, existing := range gpus {
			if existing.Index >= index {
				index++
			}
		}

		gpus = append(gpus, GPUStat{
			Index:      index,
			Name:       name,
			Util:       ptrInt(util),
			MemUsed:    memUsed,
			MemTotal:   memTotal,
			Temp:       ptrInt(temp),
			PowerDraw:  ptrInt(powerDraw),
			PowerLimit: ptrInt(powerLimit),
		})
	}
	return gpus
}

// findAMDHwmon locates the hwmon directory under an amdgpu device.
func findAMDHwmon(deviceDir string) string {
	hwmonBase := filepath.Join(deviceDir, "hwmon")
	entries, err := os.ReadDir(hwmonBase)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "hwmon") {
			namePath := filepath.Join(hwmonBase, entry.Name(), "name")
			if nameData, err := os.ReadFile(namePath); err == nil {
				if strings.TrimSpace(string(nameData)) == "amdgpu" {
					return filepath.Join(hwmonBase, entry.Name())
				}
			}
		}
	}
	return ""
}

// amdGPUName tries lspci first, falls back to PCI_ID parsing.
func amdGPUName(cardDir string) string {
	// Try lspci
	out := runCommand(3*time.Second, "lspci", "-nn")
	if out != "" {
		// Find the line matching this device's PCI slot
		ueventData, _ := os.ReadFile(filepath.Join(cardDir, "uevent"))
		var pciSlot string
		for _, line := range strings.Split(string(ueventData), "\n") {
			if strings.HasPrefix(line, "PCI_SLOT_NAME=") {
				pciSlot = strings.TrimPrefix(line, "PCI_SLOT_NAME=")
				break
			}
		}
		if pciSlot != "" {
			// PCI_SLOT_NAME like "0000:79:00.0" -> lspci shows "79:00.0"
			shortSlot := pciSlot[strings.LastIndex(pciSlot, ":")+1:]
			for _, line := range strings.Split(out, "\n") {
				if strings.HasPrefix(line, shortSlot+":") {
					// Extract name between ] and [
					start := strings.Index(line, "] ")
					if start >= 0 {
						name := strings.TrimSpace(line[start+2:])
						end := strings.Index(name, " [")
						if end >= 0 {
							return name[:end]
						}
						return name
					}
				}
			}
		}
	}

	// Fallback: read PCI_ID from uevent and format as "AMD GPU <device_id>"
	ueventData, _ := os.ReadFile(filepath.Join(cardDir, "uevent"))
	for _, line := range strings.Split(string(ueventData), "\n") {
		if strings.HasPrefix(line, "PCI_ID=") {
			pciID := strings.TrimPrefix(line, "PCI_ID=")
			parts := strings.Split(pciID, ":")
			if len(parts) == 2 {
				return fmt.Sprintf("AMD GPU (0x%s)", strings.TrimSpace(parts[1]))
			}
		}
	}
	return "AMD GPU"
}

// amdGPUWindowsStats queries AMD GPUs via Windows WMI.
// WMI only provides name and total VRAM — utilization, temp, and power are not
// exposed without the AMD Adrenalin CLI tools, so those fields are nil.
func amdGPUWindowsStats() []GPUStat {
	out := runCommand(3*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command",
		"Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -like 'PCI\\VEN_1002*' } | ForEach-Object { [string]::Format('{0}|{1}', $_.Name, $_.AdapterRAM) }")
	if strings.TrimSpace(out) == "" {
		return nil
	}
	var gpus []GPUStat
	for i, line := range strings.Split(strings.TrimSpace(out), "\n") {
		parts := strings.SplitN(line, "|", 2)
		if len(parts) < 2 {
			continue
		}
		name := strings.TrimSpace(parts[0])
		memTotalBytes, _ := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
		memTotalMB := int(memTotalBytes / (1024 * 1024))
		gpus = append(gpus, GPUStat{
			Index:    i,
			Name:     name,
			MemTotal: memTotalMB,
			// Util, Temp, PowerDraw, PowerLimit are nil (not available via WMI)
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

// ptrInt returns a pointer to the given int value.
func ptrInt(v int) *int {
	return &v
}
