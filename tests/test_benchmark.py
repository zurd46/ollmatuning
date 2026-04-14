"""Tests for ollmatuning.benchmark — Ollama HTTP API interaction."""
from unittest.mock import patch, MagicMock
import json

from ollmatuning.benchmark import (
    BenchResult, _auth_headers, ollama_is_up, list_local_models,
    BENCH_PROMPTS,
)


class TestBenchResult:
    def test_summary_ok(self):
        r = BenchResult("test:7b", 42.5, 512, 12.0, 15.0, True, vram_mb=4096)
        assert "42.5" in r.summary()
        assert "4096" in r.summary()

    def test_summary_error(self):
        r = BenchResult("test:7b", 0, 0, 0, 0, False, error="timeout")
        assert "ERROR" in r.summary()
        assert "timeout" in r.summary()

    def test_summary_no_vram(self):
        r = BenchResult("test:7b", 42.5, 512, 12.0, 15.0, True)
        assert "VRAM" not in r.summary()

    def test_to_dict_ok(self):
        r = BenchResult("test:7b", 42.5, 512, 12.0, 15.0, True, vram_mb=4096, peak_vram_mb=4200)
        d = r.to_dict()
        assert d["model"] == "test:7b"
        assert d["tokens_per_sec"] == 42.5
        assert d["ok"] is True
        assert d["vram_mb"] == 4096
        assert d["peak_vram_mb"] == 4200

    def test_to_dict_error(self):
        r = BenchResult("test:7b", 0, 0, 0, 0, False, error="timeout")
        d = r.to_dict()
        assert d["ok"] is False
        assert d["error"] == "timeout"
        assert d["vram_mb"] is None

    def test_to_dict_none_error(self):
        r = BenchResult("test:7b", 42.5, 512, 12.0, 15.0, True)
        d = r.to_dict()
        assert d["error"] is None
        assert d["vram_mb"] is None


class TestAuthHeaders:
    def test_no_key(self):
        with patch.dict("os.environ", {"OLLAMA_API_KEY": ""}, clear=False):
            headers = {"User-Agent": "test"}
            from ollmatuning.benchmark import OLLAMA_API_KEY
            if not OLLAMA_API_KEY:
                assert "Authorization" not in headers or "Bearer" not in str(headers.get("Authorization", ""))

    def test_user_agent_always_present(self):
        h = _auth_headers()
        assert "User-Agent" in h


class TestBenchPrompts:
    def test_has_code_prompt(self):
        labels = [p[0] for p in BENCH_PROMPTS]
        assert "code" in labels

    def test_has_tool_use_prompt(self):
        labels = [p[0] for p in BENCH_PROMPTS]
        assert "tool-use" in labels

    def test_has_reasoning_prompt(self):
        labels = [p[0] for p in BENCH_PROMPTS]
        assert "reasoning" in labels

    def test_all_prompts_are_tuples(self):
        for p in BENCH_PROMPTS:
            assert len(p) == 2
            assert isinstance(p[0], str)
            assert len(p[1]) > 0


class TestOllamaIsUp:
    def test_ollama_up(self):
        with patch("ollmatuning.benchmark._http_get", return_value={"models": []}):
            assert ollama_is_up() is True

    def test_ollama_down(self):
        with patch("ollmatuning.benchmark._http_get", side_effect=RuntimeError("connection refused")):
            assert ollama_is_up() is False


class TestListLocalModels:
    def test_returns_names(self):
        resp = {"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]}
        with patch("ollmatuning.benchmark._http_get", return_value=resp):
            models = list_local_models()
        assert "llama3.2:3b" in models
        assert "qwen2.5:7b" in models

    def test_empty_list(self):
        with patch("ollmatuning.benchmark._http_get", return_value={"models": []}):
            models = list_local_models()
        assert models == []

    def test_error_returns_empty(self):
        with patch("ollmatuning.benchmark._http_get", side_effect=RuntimeError):
            models = list_local_models()
        assert models == []
