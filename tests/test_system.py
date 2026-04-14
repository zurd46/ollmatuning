"""Tests for ollmatuning.system — hardware detection."""
import json
from unittest.mock import patch, MagicMock

from ollmatuning.system import (
    GPU,
    SystemInfo,
    is_apple_silicon,
    _run,
    _macos_total_ram_mb,
    _detect_nvidia,
    _detect_macos_gpus,
    check_driver,
)


class TestGPU:
    def test_gpu_creation(self):
        g = GPU(name="RTX 4090", vendor="nvidia", vram_mb=24576, driver_version="550.0")
        assert g.name == "RTX 4090"
        assert g.vendor == "nvidia"
        assert g.vram_mb == 24576
        assert g.driver_ok is True

    def test_gpu_defaults(self):
        g = GPU(name="Unknown", vendor="unknown", vram_mb=0)
        assert g.driver_version == ""
        assert g.driver_ok is True
        assert g.driver_note == ""


class TestSystemInfo:
    def test_primary_gpu_returns_largest(self):
        gpus = [
            GPU(name="GPU0", vendor="nvidia", vram_mb=8192),
            GPU(name="GPU1", vendor="nvidia", vram_mb=24576),
        ]
        info = SystemInfo(
            os="Linux", os_version="6.1", arch="x86_64",
            cpu="AMD", cpu_cores=16, ram_mb=32768, gpus=gpus,
        )
        assert info.primary_gpu == gpus[1]

    def test_primary_gpu_none(self):
        info = SystemInfo(
            os="Linux", os_version="6.1", arch="x86_64",
            cpu="AMD", cpu_cores=16, ram_mb=32768, gpus=[],
        )
        assert info.primary_gpu is None

    def test_usable_vram_mb(self):
        gpus = [GPU(name="GPU", vendor="nvidia", vram_mb=8192)]
        info = SystemInfo(
            os="Linux", os_version="6.1", arch="x86_64",
            cpu="AMD", cpu_cores=16, ram_mb=32768, gpus=gpus,
        )
        assert info.usable_vram_mb == 8192

    def test_to_dict(self):
        info = SystemInfo(
            os="Darwin", os_version="23.0", arch="arm64",
            cpu="Apple M2", cpu_cores=8, ram_mb=16384, gpus=[],
        )
        d = info.to_dict()
        assert d["os"] == "Darwin"
        assert d["ram_mb"] == 16384


class TestIsAppleSilicon:
    def test_macos_arm64(self):
        with patch("ollmatuning.system.platform.system", return_value="Darwin"), \
             patch("ollmatuning.system.platform.machine", return_value="arm64"):
            assert is_apple_silicon() is True

    def test_macos_x86_64(self):
        with patch("ollmatuning.system.platform.system", return_value="Darwin"), \
             patch("ollmatuning.system.platform.machine", return_value="x86_64"):
            assert is_apple_silicon() is False

    def test_linux(self):
        with patch("ollmatuning.system.platform.system", return_value="Linux"):
            assert is_apple_silicon() is False


class TestCheckDriver:
    def test_nvidia_good_driver(self):
        g = GPU(name="RTX 4090", vendor="nvidia", vram_mb=24576, driver_version="550.0")
        check_driver(g)
        assert g.driver_ok is True
        assert "OK" in g.driver_note

    def test_nvidia_old_driver(self):
        g = GPU(name="RTX 3090", vendor="nvidia", vram_mb=24576, driver_version="520.0")
        check_driver(g)
        assert g.driver_ok is False
        assert "old" in g.driver_note.lower()

    def test_nvidia_no_driver(self):
        g = GPU(name="RTX 3090", vendor="nvidia", vram_mb=24576)
        check_driver(g)
        assert g.driver_ok is False

    def test_amd_driver(self):
        g = GPU(name="RX 7900", vendor="amd", vram_mb=24576)
        check_driver(g)
        assert "ROCm" in g.driver_note

    def test_apple_driver(self):
        g = GPU(name="Apple M2", vendor="apple", vram_mb=0)
        check_driver(g)
        assert "Metal" in g.driver_note


class TestMacOSRam:
    def test_valid_output(self):
        with patch("ollmatuning.system._run", return_value=(0, "17179869184\n", "")):
            result = _macos_total_ram_mb()
            assert result == 16384

    def test_invalid_output(self):
        with patch("ollmatuning.system._run", return_value=(1, "", "")):
            result = _macos_total_ram_mb()
            assert result == 0


class TestDetectNvidia:
    def test_no_nvidia_smi(self):
        with patch("ollmatuning.system.shutil.which", return_value=None):
            result = _detect_nvidia()
            assert result == []

    def test_parse_nvidia_smi(self):
        output = "NVIDIA RTX 4090, 24576, 550.0\n"
        with patch("ollmatuning.system.shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("ollmatuning.system._run", return_value=(0, output, "")):
            result = _detect_nvidia()
            assert len(result) == 1
            assert result[0].name == "NVIDIA RTX 4090"
            assert result[0].vram_mb == 24576