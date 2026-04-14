"""Tests for ollmatuning.huggingface — GGUF model discovery."""
from ollmatuning.huggingface import GGUFFile, HFModel, QUANT_PREFERENCE
from ollmatuning.utils import estimate_vram_gguf


class TestGGUFFile:
    def test_size_gb(self):
        f = GGUFFile(filename="model.Q4_K_M.gguf", quant="Q4_K_M", size_bytes=4_000_000_000)
        assert abs(f.size_gb - 3.725) < 0.01

    def test_est_vram_mb(self):
        f = GGUFFile(filename="model.Q4_K_M.gguf", quant="Q4_K_M", size_bytes=4_000_000_000)
        expected = estimate_vram_gguf(4_000_000_000)
        assert f.est_vram_mb == expected


class TestHFModel:
    def test_ollama_base(self):
        m = HFModel(repo_id="bartowski/Qwen-7B-GGUF", downloads=100, likes=10, tags=[], gguf_files=[])
        assert m.ollama_base == "hf.co/bartowski/Qwen-7B-GGUF"

    def test_best_quant_for_vram_picks_preferred(self):
        files = [
            GGUFFile("m.Q2_K.gguf", "Q2_K", 2_000_000_000),
            GGUFFile("m.Q4_K_M.gguf", "Q4_K_M", 4_000_000_000),
            GGUFFile("m.Q8_0.gguf", "Q8_0", 7_000_000_000),
        ]
        m = HFModel(repo_id="test/model", downloads=0, likes=0, tags=[], gguf_files=files)
        # With 6 GB budget, should prefer Q4_K_M (index 0 in QUANT_PREFERENCE)
        best = m.best_quant_for_vram(6144)
        assert best.quant == "Q4_K_M"

    def test_best_quant_for_vram_none_fits(self):
        files = [
            GGUFFile("m.Q8_0.gguf", "Q8_0", 10_000_000_000),
        ]
        m = HFModel(repo_id="test/model", downloads=0, likes=0, tags=[], gguf_files=files)
        # With 6 GB budget, nothing fits
        assert m.best_quant_for_vram(6144) is None


class TestQuantPreference:
    def test_q4km_is_preferred(self):
        assert "Q4_K_M" in QUANT_PREFERENCE

    def test_ordering(self):
        # Q4_K_M should come before Q8_0
        assert QUANT_PREFERENCE.index("Q4_K_M") < QUANT_PREFERENCE.index("Q8_0")