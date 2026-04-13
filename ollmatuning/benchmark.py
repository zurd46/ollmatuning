"""Benchmark Ollama models on tokens/second using the Ollama HTTP API.

Auth: reads OLLAMA_HOST and OLLAMA_API_KEY from the environment. When a key
is set, every request is signed with `Authorization: Bearer <key>`. This lets
you point the tool at an Ollama instance behind a reverse proxy (Caddy / nginx)
that enforces Bearer authentication.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import ssl  # Added SSL module for security
from dataclasses import dataclass

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"User-Agent": "ollmatuning/0.1"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    if extra:
        headers.update(extra)
    return headers

# Coding + tool-style prompts — representative of real use.
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
            return f"{self.model}: FEHLER ({self.error})"
        vram = f", {self.vram_mb} MB VRAM" if self.vram_mb else ""
        return (
            f"{self.model}: {self.tokens_per_sec:6.2f} tok/s "
            f"({self.eval_count} tok in {self.eval_seconds:.2f}s{vram})"
        )


def _http_post(path: str, payload: dict, timeout: int = 600) -> dict:
    # Handle SSL context based on URL scheme
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=data, headers=_auth_headers(), method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ssl.SSLError) as e:
        raise RuntimeError(f"HTTP error: {e}") from e


def _http_get(path: str, timeout: int = 10) -> dict:
    # Handle SSL context based on URL scheme
    if OLLAMA_HOST.startswith("https://"):
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        req = urllib.request.Request(f"{OLLAMA_HOST}{path}", headers=_auth_headers())
        try:
            with urllib.request.urlopen(req, context=context, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except ssl.SSLError as e:
            raise RuntimeError(f"SSL verification failed: {e}") from e
    else:
        req = urllib.request.Request(f"{OLLAMA_HOST}{path}", headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))


def ollama_is_up() -> bool:
    try:
        _http_get("/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def list_local_models() -> list[str]:
    try:
        data = _http_get("/api/tags")
    except Exception:
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
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            last_status = ""
            for raw in r:
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                status = obj.get("status", "")
                if "error" in obj:
                    if verbose:
                        print(f"    pull error: {obj['error']}")
                    return False
                if verbose and status and status != last_status:
                    print(f"    {status}")
                    last_status = status
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if verbose:
            print(f"    pull failed: {e}")
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
        pass
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
        }, timeout=600)

        # Measure actual VRAM after model is loaded.
        vram_mb = _get_model_vram(model)

        for _label, prompt in prompts:
            resp = _http_post("/api/generate", {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0.0},
            }, timeout=1200)
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