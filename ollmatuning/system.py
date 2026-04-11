"""Hardware and driver detection for Windows, Linux, and macOS."""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GPU:
    name: str
    vendor: str           # nvidia | amd | intel | apple | unknown
    vram_mb: int          # 0 if unknown
    driver_version: str = ""
    driver_ok: bool = True
    driver_note: str = ""


@dataclass
class SystemInfo:
    os: str
    os_version: str
    arch: str
    cpu: str
    cpu_cores: int
    ram_mb: int
    gpus: list[GPU] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def primary_gpu(self) -> Optional[GPU]:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_mb)

    @property
    def usable_vram_mb(self) -> int:
        g = self.primary_gpu
        return g.vram_mb if g else 0


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return p.returncode, p.stdout, p.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1, "", ""


def _detect_nvidia() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    rc, out, _ = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0 or not out.strip():
        return []
    gpus: list[GPU] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem, drv = parts[0], parts[1], parts[2]
        try:
            vram = int(float(mem))
        except ValueError:
            vram = 0
        gpus.append(GPU(name=name, vendor="nvidia", vram_mb=vram, driver_version=drv))
    return gpus


def _detect_windows_gpus() -> list[GPU]:
    gpus = _detect_nvidia()
    if gpus:
        return gpus
    # PowerShell CIM fallback — covers AMD/Intel/etc.
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    rc, out, _ = _run(["powershell", "-NoProfile", "-Command", ps], timeout=15)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    for item in data:
        name = str(item.get("Name", "")).strip()
        if not name:
            continue
        vram_bytes = item.get("AdapterRAM") or 0
        try:
            vram_mb = int(vram_bytes) // (1024 * 1024)
        except (TypeError, ValueError):
            vram_mb = 0
        drv = str(item.get("DriverVersion", "") or "")
        lower = name.lower()
        if "nvidia" in lower or "geforce" in lower or "rtx" in lower or "quadro" in lower:
            vendor = "nvidia"
        elif "amd" in lower or "radeon" in lower:
            vendor = "amd"
        elif "intel" in lower:
            vendor = "intel"
        else:
            vendor = "unknown"
        gpus.append(GPU(name=name, vendor=vendor, vram_mb=vram_mb, driver_version=drv))
    return gpus


def _detect_linux_gpus() -> list[GPU]:
    gpus = _detect_nvidia()
    if gpus:
        return gpus
    if shutil.which("lspci"):
        rc, out, _ = _run(["lspci", "-nn"])
        if rc == 0:
            for line in out.splitlines():
                low = line.lower()
                if "vga" in low or "3d controller" in low or "display" in low:
                    name = line.split(":", 2)[-1].strip()
                    if "nvidia" in low:
                        vendor = "nvidia"
                    elif "amd" in low or "ati" in low or "radeon" in low:
                        vendor = "amd"
                    elif "intel" in low:
                        vendor = "intel"
                    else:
                        vendor = "unknown"
                    gpus.append(GPU(name=name, vendor=vendor, vram_mb=0))
    return gpus


def _detect_macos_gpus() -> list[GPU]:
    rc, out, _ = _run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=20)
    gpus: list[GPU] = []
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            for item in data.get("SPDisplaysDataType", []):
                name = item.get("sppci_model") or item.get("_name") or "Apple GPU"
                vram_str = item.get("spdisplays_vram") or item.get("spdisplays_vram_shared") or ""
                vram_mb = 0
                m = re.search(r"(\d+)\s*(GB|MB)", str(vram_str), re.I)
                if m:
                    val = int(m.group(1))
                    vram_mb = val * 1024 if m.group(2).upper() == "GB" else val
                vendor = "apple" if "apple" in str(name).lower() else "unknown"
                gpus.append(GPU(name=str(name), vendor=vendor, vram_mb=vram_mb))
        except json.JSONDecodeError:
            pass
    # Apple Silicon uses unified memory — fall back to total RAM if VRAM missing.
    if gpus and platform.machine().lower() in ("arm64", "aarch64"):
        for g in gpus:
            if g.vendor != "apple":
                g.vendor = "apple"
            if g.vram_mb == 0:
                g.vram_mb = _macos_total_ram_mb()
    return gpus


def _macos_total_ram_mb() -> int:
    rc, out, _ = _run(["sysctl", "-n", "hw.memsize"])
    if rc == 0 and out.strip().isdigit():
        return int(out.strip()) // (1024 * 1024)
    return 0


def _total_ram_mb() -> int:
    system = platform.system()
    if system == "Windows":
        rc, out, _ = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        if rc == 0 and out.strip().isdigit():
            return int(out.strip()) // (1024 * 1024)
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except OSError:
            pass
    elif system == "Darwin":
        return _macos_total_ram_mb()
    return 0


def _cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        rc, out, _ = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ])
        if rc == 0 and out.strip():
            return out.strip().splitlines()[0]
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        rc, out, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if rc == 0:
            return out.strip()
    return platform.processor() or "unknown"


def check_driver(gpu: GPU) -> None:
    """Annotate a GPU with driver health notes."""
    if gpu.vendor == "nvidia":
        if not gpu.driver_version:
            gpu.driver_ok = False
            gpu.driver_note = "NVIDIA driver not detected (nvidia-smi missing?)"
            return
        try:
            major = int(gpu.driver_version.split(".")[0])
        except ValueError:
            major = 0
        if major and major < 525:
            gpu.driver_ok = False
            gpu.driver_note = f"Driver {gpu.driver_version} is old — >=525 recommended for CUDA 12."
        else:
            gpu.driver_note = f"NVIDIA driver {gpu.driver_version} OK."
    elif gpu.vendor == "amd":
        gpu.driver_note = "AMD detected — ROCm on Linux recommended; limited Ollama support on Windows (Vulkan/DirectML)."
    elif gpu.vendor == "intel":
        gpu.driver_note = "Intel GPU — Ollama falls back to CPU, no real GPU acceleration."
    elif gpu.vendor == "apple":
        gpu.driver_note = "Apple Silicon — Metal acceleration active."
    else:
        gpu.driver_note = "Unknown GPU type."


def detect_system() -> SystemInfo:
    system = platform.system()
    if system == "Windows":
        gpus = _detect_windows_gpus()
    elif system == "Linux":
        gpus = _detect_linux_gpus()
    elif system == "Darwin":
        gpus = _detect_macos_gpus()
    else:
        gpus = []

    for g in gpus:
        check_driver(g)

    import os
    info = SystemInfo(
        os=system,
        os_version=platform.version(),
        arch=platform.machine(),
        cpu=_cpu_name(),
        cpu_cores=os.cpu_count() or 0,
        ram_mb=_total_ram_mb(),
        gpus=gpus,
    )
    return info
