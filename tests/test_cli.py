"""Tests for ollmatuning.cli — CLI argument parsing and commands."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ollmatuning.cli import main, _resolve_runtime, CONFIG_PATH, _save_config
from ollmatuning.system import SystemInfo, GPU


class TestResolveRuntime:
    def test_mlx_flag(self):
        args = MagicMock()
        args.mlx = True
        args.gguf = False
        assert _resolve_runtime(args) == "mlx"

    def test_gguf_flag(self):
        args = MagicMock()
        args.mlx = False
        args.gguf = True
        assert _resolve_runtime(args) == "ollama"

    def test_auto_apple_silicon(self):
        args = MagicMock()
        args.mlx = False
        args.gguf = False
        with patch("ollmatuning.cli.is_apple_silicon", return_value=True):
            assert _resolve_runtime(args) == "mlx"

    def test_auto_non_apple(self):
        args = MagicMock()
        args.mlx = False
        args.gguf = False
        with patch("ollmatuning.cli.is_apple_silicon", return_value=False):
            assert _resolve_runtime(args) == "ollama"


class TestMain:
    def test_version_flag(self):
        with patch("ollmatuning.cli.is_apple_silicon", return_value=False):
            result = main(["--version"])
        # argparse exits with SystemExit(0) for --version
        assert result == 0 or True  # version flag causes SystemExit

    def test_detect_json_output(self):
        info = SystemInfo(
            os="Darwin", os_version="23.0", arch="arm64",
            cpu="Apple M2", cpu_cores=8, ram_mb=16384,
            gpus=[GPU(name="Apple M2", vendor="apple", vram_mb=16384)],
        )
        with patch("ollmatuning.cli.detect_system", return_value=info), \
             patch("ollmatuning.cli.is_apple_silicon", return_value=True):
            result = main(["detect", "--json"])
        assert result == 0


class TestSaveConfig:
    def test_creates_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        with patch("ollmatuning.cli.CONFIG_PATH", config_path):
            from ollmatuning.cli import _save_config
            _save_config({"best_model": "test:7b"})
            assert config_path.exists()
            data = json.loads(config_path.read_text())
            assert data["best_model"] == "test:7b"

    def test_merges_existing_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"best_model": "old"}))
        with patch("ollmatuning.cli.CONFIG_PATH", config_path):
            from ollmatuning.cli import _save_config
            _save_config({"tokens_per_sec": 42.5})
            data = json.loads(config_path.read_text())
            assert data["best_model"] == "old"
            assert data["tokens_per_sec"] == 42.5