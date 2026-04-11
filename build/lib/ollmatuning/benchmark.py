"""Benchmark Ollama models on tokens/second using the local HTTP API."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_HOST = "http://127.0.0.1:11434"

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

    def summary(self) -> str:
        if not self.ok:
            return f"{self.model}: FEHLER ({self.error})"
        return (
            f"{self.model}: {self.tokens_per_sec:6.2f} tok/s "
            f"({self.eval_count} tok in {self.eval_seconds:.2f}s)"
        )


def _http_post(path: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_get(path: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(f"{OLLAMA_HOST}{path}")
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
        headers={"Content-Type": "application/json"},
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

        for _label, prompt in prompts:
            resp = _http_post("/api/generate", {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0.0},
            }, timeout=1200)
            total_tokens += int(resp.get("eval_count", 0))
            total_eval_ns += int(resp.get("eval_duration", 0))
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
    )
