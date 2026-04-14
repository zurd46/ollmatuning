"""Tests for ollmatuning.mlx_models — MLX model discovery."""
from ollmatuning.mlx_models import _parse_quant_bits, _normalize_base_name


class TestParseQuantBits:
    def test_4bit_suffix(self):
        assert _parse_quant_bits("Qwen-Coder-4bit") == 4

    def test_8bit_suffix(self):
        assert _parse_quant_bits("Qwen-Coder-8bit") == 8

    def test_4_bit_hyphen(self):
        assert _parse_quant_bits("Qwen-Coder-4-bit") == 4

    def test_w4a16(self):
        assert _parse_quant_bits("Qwen-Coder-w4a16") == 4

    def test_no_quant(self):
        assert _parse_quant_bits("Qwen-Coder") == 0

    def test_3bit(self):
        assert _parse_quant_bits("model-3bit") == 3


class TestNormalizeBaseName:
    def test_strips_4bit(self):
        result = _normalize_base_name("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
        assert result == "qwen3-coder-30b-a3b-instruct"

    def test_strips_mlx(self):
        result = _normalize_base_name("lmstudio-community/Qwen3-Coder-MLX")
        assert result == "qwen3-coder"

    def test_strips_mlx_4bit(self):
        result = _normalize_base_name("org/Model-MLX-4bit")
        assert result == "model"

    def test_strips_fp16(self):
        result = _normalize_base_name("org/Model-fp16")
        assert result == "model"

    def test_strips_bf16(self):
        result = _normalize_base_name("org/Model-bf16")
        assert result == "model"

    def test_lowercase(self):
        result = _normalize_base_name("org/SomeModel-4bit")
        assert result == "somemodel"