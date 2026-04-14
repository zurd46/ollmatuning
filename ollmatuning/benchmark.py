"""Benchmark Ollama models on tokens/second using the Ollama HTTP API.

Auth: reads OLLAMA_HOST and OLLAMA_API_KEY from the environment. When a key
is set, every request is signed with `Authorization: Bearer <key>`. This lets
you point the tool at an Ollama instance behind a reverse proxy (Caddy / nginx)
that enforces Bearer authentication.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .utils import (
    USER_AGENT,
    HTTP_TIMEOUT_SHORT,
    HTTP_TIMEOUT_BENCHMARK_WARMUP,
    HTTP_TIMEOUT_BENCHMARK,
    HTTP_TIMEOUT_PULL,
    BENCH_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"User-Agent": USER_AGENT}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    if extra:
        headers.update(extra)
    return headers


def _make_ssl_context() -> ssl.SSLContext | None:
    """Return a verified SSL context for HTTPS, or None for HTTP."""
    if OLLAMA_HOST.startswith("https://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx
    return None


# Coding + tool-style + reasoning prompts — representative of real use.
BENCH_PROMPTS = [
    (
        "code",
        "Write a Python function `merge_intervals(intervals)` that merges overlapping "
        "intervals and returns the sorted result. Include a short docstring and 3 "
        "assert-based test cases. Output only code.",
    ),
    (
        "tool-use",
        "You can call the function `get_weather(city: str) -> dict`. The user asks: "
        "'What is the weather in Berlin and Tokyo right now?'. Respond with a JSON "
        "array of the exact function calls you would make, nothing else.",
    ),
    (
        "reasoning",
        "A farmer has a wolf, a goat, and a cabbage. He needs to cross a river but "
        "his boat can only carry him and one item at a time. If left alone, the wolf "
        "will eat the goat, and the goat will eat the cabbage. What is the minimum "
        "number of crossings needed? Explain step by step.",
    ),
]


@dataclass
class BenchResult:
    model: str
    tokens_per_sec: float
    eval_count: int
    eval_seconds: float
    total_seconds: float
    ok: bool
    error: str = ""
    vram_mb: int = 0       # actual VRAM/memory used by the model
    peak_vram_mb: int = 0  # peak VRAM/memory during inference

    def summary(self) -> str:
        if not self.ok:
            return f"{self.model}: ERROR ({self.error})"
        vram = f", {self.vram_mb} MB VRAM" if self.vram_mb else ""
        return (
            f"{self.model}: {self.tokens_per_sec:6.2f} tok/s "
            f"({self.eval_count} tok in {self.eval_seconds:.2f}s{vram})"
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "model": self.model,
            "tokens_per_sec": self.tokens_per_sec,
            "eval_count": self.eval_count,
            "eval_seconds": self.eval_seconds,
            "total_seconds": self.total_seconds,
            "ok": self.ok,
            "error": self.error if self.error else None,
            "vram_mb": self.vram_mb if self.vram_mb else None,
            "peak_vram_mb": self.peak_vram_mb if self.peak_vram_mb else None,
        }


def _http_post(path: str, payload: dict, timeout: int = HTTP_TIMEOUT_BENCHMARK_WARMUP) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=data, headers=_auth_headers(), method="POST"
    )
    ctx = _make_ssl_context()
    try:
        if ctx:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        else:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ssl.SSLError) as e:
        raise RuntimeError(f"HTTP error: {e}") from e


def _http_get(path: str, timeout: int = HTTP_TIMEOUT_SHORT) -> dict:
    req = urllib.request.Request(f"{OLLAMA_HOST}{path}", headers=_auth_headers())
    ctx = _make_ssl_context()
    try:
        if ctx:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        else:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
    except ssl.SSLError as e:
        raise RuntimeError(f"SSL verification failed: {e}") from e


def ollama_is_up() -> bool:
    try:
        _http_get("/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError):
        return False


def list_local_models() -> list[str]:
    try:
        data = _http_get("/api/tags")
    except Exception:
        logger.debug("Failed to list local models", exc_info=True)
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def pull_model(model: str, verbose: bool = True) -> bool:
    """Stream /api/pull; return True on success."""
    data = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/pull",
        data=data,
        headers=_auth_headers({"Content-Type": "application/json"}),
        method="POST",
    )
    ctx = _make_ssl_context()
    try:
        if ctx:
            r = urllib.request.urlopen(req, context=ctx, timeout=HTTP_TIMEOUT_PULL)
        else:
            r = urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_PULL)
        with r:
            last_status = ""
            for raw in r:
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line in pull stream for %s", model)
                    continue
                status = obj.get("status", "")
                if "error" in obj:
                    if verbose:
                        logger.info("Pull error for %s: %s", model, obj["error"])
                    return False
                if verbose and status and status != last_status:
                    logger.info("%s: %s", model, status)
                    last_status = status
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if verbose:
            logger.warning("Pull failed for %s: %s", model, e)
        return False


def _get_model_vram(model: str) -> int:
    """Query /api/ps to get actual VRAM usage for a loaded model (in MB)."""
    try:
        data = _http_get("/api/ps", timeout=5)
        for m in data.get("models", []):
            name = m.get("name", "")
            if name == model or name.startswith(model + ":"):
                size_bytes = m.get("size_vram", 0) or m.get("size", 0)
                return int(size_bytes / (1024 * 1024))
    except Exception:
        logger.debug("Failed to query VRAM for %s", model, exc_info=True)
    return 0


def benchmark_model(model: str, prompts: list[tuple[str, str]] | None = None) -> BenchResult:
    prompts = prompts or BENCH_PROMPTS
    total_tokens = 0
    total_eval_ns = 0
    t0 = time.perf_counter()
    try:
        # Warm-up: load weights into memory.
        _http_post("/api/generate", {
            "model": model,
            "prompt": "Hi.",
            "stream": False,
            "options": {"num_predict": 8},
        }, timeout=HTTP_TIMEOUT_BENCHMARK_WARMUP)

        # Measure actual VRAM after model is loaded.
        vram_mb = _get_model_vram(model)

        for _label, prompt in prompts:
            resp = _http_post("/api/generate", {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": BENCH_MAX_TOKENS, "temperature": 0.0},
            }, timeout=HTTP_TIMEOUT_BENCHMARK)
            total_tokens += int(resp.get("eval_count", 0))
            total_eval_ns += int(resp.get("eval_duration", 0))

        # Measure peak VRAM after inference.
        peak_vram_mb = _get_model_vram(model)
    except Exception as e:
        return BenchResult(
            model=model, tokens_per_sec=0.0, eval_count=0,
            eval_seconds=0.0, total_seconds=0.0, ok=False, error=str(e),
        )

    eval_s = total_eval_ns / 1e9 if total_eval_ns else 0.0
    tps = (total_tokens / eval_s) if eval_s > 0 else 0.0
    return BenchResult(
        model=model,
        tokens_per_sec=tps,
        eval_count=total_tokens,
        eval_seconds=eval_s,
        total_seconds=time.perf_counter() - t0,
        ok=True,
        vram_mb=vram_mb,
        peak_vram_mb=peak_vram_mb or vram_mb,
    )
