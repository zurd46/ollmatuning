"""Tests for ollmatuning.recommend — model discovery and Candidate logic."""
from unittest.mock import patch, MagicMock
from ollmatuning.recommend import Candidate, shortlist, _tag_to_size_b
from ollmatuning.system import SystemInfo, GPU


class TestCandidate:
    def test_pretty(self):
        c = Candidate("llama3.2:3b", "llama3.2", 3.2, 2048, ["code", "tools"])
        assert "llama3.2:3b" in c.pretty
        assert "3.2" in c.pretty
        assert "2048" in c.pretty

    def test_to_dict(self):
        c = Candidate("m", "f", 7.0, 4000, ["tools"], "huggingface", "ollama")
        d = c.to_dict()
        assert d == {
            "model": "m", "family": "f", "size_b": 7.0,
            "est_vram_mb": 4000, "categories": ["tools"],
            "source": "huggingface", "runtime": "ollama",
        }

    def test_defaults(self):
        c = Candidate("m", "f", 1, 500, [])
        assert c.source == "ollama"
        assert c.runtime == "ollama"


class TestShortlist:
    def test_one_per_family(self):
        cands = [
            Candidate("llama:3b", "llama", 3, 2000, ["code", "tools"]),
            Candidate("llama:8b", "llama", 8, 5000, ["code", "tools"]),
            Candidate("qwen:7b", "qwen", 7, 4000, ["code"]),
        ]
        result = shortlist(cands)
        families = {c.family for c in result}
        assert families == {"llama", "qwen"}
        # Should pick the first (highest-ranked) per family
        assert result[0].model == "llama:3b"

    def test_limit(self):
        cands = [
            Candidate("m1", "f1", 1, 500, []),
            Candidate("m2", "f2", 2, 1000, []),
            Candidate("m3", "f3", 3, 1500, []),
        ]
        result = shortlist(cands, limit=2)
        assert len(result) == 2

    def test_no_limit_returns_all_families(self):
        cands = [
            Candidate("m1", "f1", 1, 500, []),
            Candidate("m2", "f2", 2, 1000, []),
        ]
        result = shortlist(cands)
        assert len(result) == 2


class TestTagToSizeB:
    def test_simple_tag(self):
        assert _tag_to_size_b("7b") == 7.0

    def test_decimal_tag(self):
        assert _tag_to_size_b("0.5b") == 0.5

    def test_large_tag(self):
        assert _tag_to_size_b("32b") == 32.0

    def test_no_size_returns_zero(self):
        assert _tag_to_size_b("latest") == 0.0

    def test_tag_with_extra(self):
        assert _tag_to_size_b("7b-q4_k_m") == 7.0


def _make_info(vram_mb=8192, ram_mb=16384):
    return SystemInfo(
        os="Linux", os_version="6.1", arch="x86_64",
        cpu="AMD", cpu_cores=16, ram_mb=ram_mb,
        gpus=[GPU(name="Test", vendor="nvidia", vram_mb=vram_mb, driver_version="550")],
    )
