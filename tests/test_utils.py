"""Tests for ollmatuning.utils — shared constants and helpers."""
from ollmatuning.utils import (
    guess_param_size,
    detect_categories,
    compute_vram_budget,
    estimate_vram_gguf,
    estimate_vram_mlx,
    estimate_vram_ollama,
    CODE_KEYWORDS,
    TOOL_KEYWORDS,
    VRAM_OVERHEAD_GGUF_MB,
    VRAM_OVERHEAD_MLX_MB,
    VRAM_OVERHEAD_OLLAMA_MB,
    DEFAULT_VRAM_FALLBACK_MB,
    DEFAULT_MLX_BUDGET_MB,
)


class TestGuessParamSize:
    def test_param_name_pattern(self):
        assert guess_param_size("Qwen2.5-Coder-7B-Instruct") == 7.0

    def test_param_name_pattern_decimal(self):
        assert guess_param_size("model-0.5b-chat") == 0.5

    def test_param_with_space_pattern(self):
        assert guess_param_size("Qwen 2.5 Coder 32B") == 32.0

    def test_no_param_match(self):
        assert guess_param_size("some-random-model") == 0.0

    def test_multiple_matches_looser_pattern(self):
        # "7B" at string start doesn't match _PARAM_NAME_RE (needs delimiter before),
        # so the looser pattern picks up "13B"
        assert guess_param_size("7B-13B-model") == 13.0


class TestDetectCategories:
    def test_code_keyword(self):
        for kw in CODE_KEYWORDS:
            result = detect_categories(f"model-{kw}-v2")
            assert "code" in result

    def test_tool_keyword(self):
        for kw in TOOL_KEYWORDS:
            result = detect_categories(f"model-{kw}-v2")
            assert "tools" in result

    def test_code_and_tools(self):
        result = detect_categories("coder-tool-v2")
        assert "code" in result
        assert "tools" in result

    def test_general_fallback(self):
        result = detect_categories("random-model-name")
        assert result == ["general"]

    def test_case_insensitive(self):
        result = detect_categories("MODEL-CODE-V2")
        assert "code" in result


class TestComputeVramBudget:
    def test_dedicated_vram_above_threshold(self):
        # If VRAM >= 2048 MB, use it directly
        budget = compute_vram_budget(8192, 16384, runtime="ollama")
        assert budget == 8192

    def test_small_vram_uses_ram_fraction_gguf(self):
        # If VRAM < 2048 MB, use 60% of RAM for GGUF
        budget = compute_vram_budget(1024, 16384, runtime="ollama")
        assert budget == int(16384 * 0.6)

    def test_small_vram_uses_ram_fraction_mlx(self):
        # If VRAM < 2048 MB, use 75% of RAM for MLX
        budget = compute_vram_budget(1024, 16384, runtime="mlx")
        assert budget == int(16384 * 0.75)

    def test_zero_vram_uses_ram(self):
        budget = compute_vram_budget(0, 8192, runtime="ollama")
        assert budget == int(8192 * 0.6)

    def test_zero_everything_fallback_gguf(self):
        budget = compute_vram_budget(0, 0, runtime="ollama")
        assert budget == DEFAULT_VRAM_FALLBACK_MB

    def test_zero_everything_fallback_mlx(self):
        budget = compute_vram_budget(0, 0, runtime="mlx")
        assert budget == DEFAULT_MLX_BUDGET_MB


class TestVramEstimates:
    def test_gguf_overhead(self):
        # 1 GB file = 1024 MB + 500 MB overhead
        size = 1 * 1024 * 1024 * 1024  # 1 GiB
        result = estimate_vram_gguf(size)
        assert result == 1024 + VRAM_OVERHEAD_GGUF_MB

    def test_mlx_overhead(self):
        size = 1 * 1024 * 1024 * 1024  # 1 GiB
        result = estimate_vram_mlx(size)
        assert result == 1024 + VRAM_OVERHEAD_MLX_MB

    def test_ollama_estimate(self):
        result = estimate_vram_ollama(7.0)
        assert result == int(7.0 * 600 + VRAM_OVERHEAD_OLLAMA_MB)

    def test_zero_bytes(self):
        assert estimate_vram_gguf(0) == VRAM_OVERHEAD_GGUF_MB
        assert estimate_vram_mlx(0) == VRAM_OVERHEAD_MLX_MB