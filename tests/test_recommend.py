"""Tests for ollmatuning.recommend — Ollama model discovery."""
from unittest.mock import patch

from ollmatuning.recommend import (
    Candidate,
    _parse_library_page,
    _tag_to_size_b,
    shortlist,
)
from ollmatuning.system import SystemInfo, GPU


class TestCandidate:
    def test_pretty(self):
        c = Candidate(model="codellama:7b", family="codellama", size_b=7.0,
                      est_vram_mb=5200, categories=["code", "tools"])
        assert "7B" in c.pretty
        assert "5200 MB" in c.p

    def test_default_runtime(self):
        c = Candidate(model="test", family="test", size_b=1, est_vram_mb=100, categories=[])
        assert c.runtime == "ollama"
        assert c.source == "ollama"


class TestParseLibraryPage:
    def test_extracts_slugs(self):
        html = '<a href="/library/llama">llama</a> <a href="/library/codellama">code</a>'
        slugs = _parse_library_page(html)
        assert "llama" in slugs
        assert "codellama" in slugs

    def test_deduplicates(self):
        html = '<a href="/library/llama">a</a><a href="/library/llama">b</a>'
        slugs = _parse_library_page(html)
        assert slugs == ["llama"]

    def test_empty_page(self):
        assert _parse_library_page("") == []


class TestTagToSize:
    def test_7b(self):
        assert _tag_to_size_b("7b") == 7.0

    def test_13b(self):
        assert _tag_to_size_b("13b") == 13.0

    def test_no_size(self):
        assert _tag_to_size_b("latest") == 0.0

    def test_decimal_size(self):
        assert _tag_to_size_b("0.5b") == 0.5


class TestShortlist:
    def test_picks_one_per_family(self):
        candidates = [
            Candidate(model="a:7b", family="a", size_b=7, est_vram_mb=5000, categories=["code"]),
            Candidate(model="a:13b", family="a", size_b=13, est_vram_mb=9000, categories=["code"]),
            Candidate(model="b:7b", family="b", size_b=7, est_vram_mb=5000, categories=["tools"]),
        ]
        result = shortlist(candidates, limit=10)
        families = [c.family for c in result]
        assert len(set(families)) == len(families)  # no duplicates

    def test_limit(self):
        candidates = [
            Candidate(model=f"m{i}", family=f"f{i}", size_b=7, est_vram_mb=5000, categories=[])
            for i in range(10)
        ]
        result = shortlist(candidates, limit=3)
        assert len(result) == 3

    def test_no_limit(self):
        candidates = [
            Candidate(model=f"m{i}", family=f"f{i}", size_b=7, est_vram_mb=5000, categories=[])
            for i in range(10)
        ]
        result = shortlist(candidates, limit=None)
        assert len(result) == 10