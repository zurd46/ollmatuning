"""Tests for ollmatuning.benchmark — Ollama HTTP benchmark."""
from ollmatuning.benchmark import BenchResult


class TestBenchResult:
    def test_summary_success(self):
        r = BenchResult(
            model="test-model",
            tokens_per_sec=42.5,
            eval_count=100,
            eval_seconds=2.35,
            total_seconds=3.0,
            ok=True,
            vram_mb=8192,
        )
        s = r.summary()
        assert "42.50 tok/s" in s
        assert "100 tok" in s
        assert "8192 MB VRAM" in s

    def test_summary_error(self):
        r = BenchResult(
            model="bad-model",
            tokens_per_sec=0.0,
            eval_count=0,
            eval_seconds=0.0,
            total_seconds=0.0,
            ok=False,
            error="connection refused",
        )
        s = r.summary()
        assert "ERROR" in s
        assert "connection refused" in s
        assert "FEHLER" not in s  # Was FEHLER before fix

    def test_summary_no_vram(self):
        r = BenchResult(
            model="test",
            tokens_per_sec=10.0,
            eval_count=50,
            eval_seconds=5.0,
            total_seconds=6.0,
            ok=True,
            vram_mb=0,
        )
        s = r.summary()
        assert "VRAM" not in s

    def test_bench_result_defaults(self):
        r = BenchResult(
            model="x", tokens_per_sec=1.0, eval_count=1,
            eval_seconds=1.0, total_seconds=1.0, ok=True,
        )
        assert r.vram_mb == 0
        assert r.peak_vram_mb == 0
        assert r.error == ""