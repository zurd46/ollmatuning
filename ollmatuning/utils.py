"""Shared utilities and constants for ollmatuning.

Centralises constants, HTTP helpers, and common heuristics that were
duplicated across huggingface.py, mlx_models.py, recommend.py, and
benchmark.py.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "ollmatuning/0.1 (+https://github.com/daniel-zurmuehle/ollmatuning)"

HF_API_BASE = "https://huggingface.co/api"
OLLAMA_LIBRARY_URL = "https://ollama.com"

# VRAM / memory budget defaults
VRAM_OVERHEAD_GGUF_MB = 500       # overhead for GGUF models
VRAM_OVERHEAD_MLX_MB = 300       # overhead for MLX models
VRAM_OVERHEAD_OLLAMA_MB = 1024   # overhead for Ollama library models
RAM_FRACTION_GGUF = 0.6          # fraction of RAM for GGUF when no dedicated VRAM
RAM_FRACTION_MLX = 0.75          # fraction of RAM for MLX (unified memory)
DEFAULT_VRAM_FALLBACK_MB = 8192  # fallback when VRAM and RAM are unknown
DEFAULT_MLX_BUDGET_MB = 16384   # fallback budget for MLX
VRAM_SMALL_THRESHOLD_MB = 2048  # below this, use RAM fraction instead

# HTTP timeouts (seconds)
HTTP_TIMEOUT_SHORT = 10
HTTP_TIMEOUT_MEDIUM = 15
HTTP_TIMEOUT_LONG = 20
HTTP_TIMEOUT_PULL = 3600
HTTP_TIMEOUT_BENCHMARK_WARMUP = 600
HTTP_TIMEOUT_BENCHMARK = 1200

# Thread pool sizes
HF_FETCH_WORKERS = 12
OLLAMA_VERIFY_WORKERS = 16

# Rate limiting: minimum seconds between HTTP requests to same host
HTTP_RATE_LIMIT_SECONDS = 0.2

# Bench defaults
BENCH_MAX_TOKENS = 512

# Category detection keywords
CODE_KEYWORDS = ("code", "coder", "coding")
TOOL_KEYWORDS = ("tool", "function", "func-call")
REASONING_KEYWORDS = ("reason", "think", "cot", "chain-of-thought")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple per-host rate limiter to avoid getting blocked by HF/ollama.com."""

    def __init__(self, min_interval: float = HTTP_RATE_LIMIT_SECONDS):
        self._min_interval = min_interval
        self._last_request: dict[str, float] = {}

    def wait(self, url: str) -> None:
        domain = _url_domain(url)
        last = self._last_request.get(domain, 0)
        elapsed = time.monotonic() - last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request[domain] = time.monotonic()


_rate_limiter = _RateLimiter()


def _url_domain(url: str) -> str:
    """Extract the domain from a URL."""
    # Simple parsing: after :// up to the next /
    start = url.find("://")
    if start < 0:
        return url
    rest = url[start + 3:]
    slash = rest.find("/")
    if slash < 0:
        return rest
    return rest[:slash]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger for the application."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# HTTP helpers with rate limiting
# ---------------------------------------------------------------------------


def fetch_json(url: str, timeout: int = HTTP_TIMEOUT_LONG) -> dict | list | None:
    """Fetch a URL and parse the response as JSON.

    Returns None on any network or parsing error.
    Applies rate limiting to avoid getting blocked.
    """
    _rate_limiter.wait(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        logger.warning("fetch_json(%s) failed: %s", url, exc)
        return None


def fetch_text(url: str, timeout: int = HTTP_TIMEOUT_MEDIUM) -> str:
    """Fetch a URL and return the response body as text.

    Returns empty string on any error.
    Applies rate limiting to avoid getting blocked.
    """
    _rate_limiter.wait(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("fetch_text(%s) failed: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Model name heuristics
# ---------------------------------------------------------------------------

# MoE pattern: e.g. "30B-A3B" means 30B total params, 3B active.
# We want the total count for VRAM estimation.
_TOTAL_PARAM_RE = re.compile(r"[-_.](\d+(?:\.\d+)?)B[-_.]")
_MOE_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)(?:B|billion)[-_]?A[0-9.]+b")
_GENERIC_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:B|billion)", re.IGNORECASE)


def guess_param_size(repo_id: str) -> float:
    """Try to extract parameter count (in billions) from repo/model name.

    Handles MoE models like 'Qwen3-235B-A22B' — returns total params (235),
    not active params. Prefers the -<N>B- pattern which is most common.
    Falls back to generic <N>B or <N>billion pattern.
    """
    # First try the explicit -NB- pattern (includes MoE total: 30B-A3B -> 30)
    m = _TOTAL_PARAM_RE.search(repo_id)
    if m:
        return float(m.group(1))

    # Try MoE-style: 30B-A3B, 235B-A22B etc.
    m = _MOE_PARAM_RE.search(repo_id)
    if m:
        return float(m.group(1))

    # Generic pattern
    m = _GENERIC_PARAM_RE.search(repo_id.replace("-", " "))
    if m:
        return float(m.group(1))

    return 0.0


def detect_categories(text: str) -> list[str]:
    """Detect model categories from tags, name, or other text.

    Returns a non-empty list (falls back to ["general"]).
    """
    lower = text.lower()
    cats: list[str] = []
    if any(k in lower for k in CODE_KEYWORDS):
        cats.append("code")
    if any(k in lower for k in TOOL_KEYWORDS):
        cats.append("tools")
    if any(k in lower for k in REASONING_KEYWORDS):
        cats.append("reasoning")
    if not cats:
        cats.append("general")
    return cats


# ---------------------------------------------------------------------------
# VRAM / memory budget helpers
# ---------------------------------------------------------------------------


def compute_vram_budget(
    vram_mb: int,
    ram_mb: int,
    runtime: str = "ollama",
) -> int:
    """Compute the usable VRAM/memory budget for model selection.

    - If dedicated VRAM < VRAM_SMALL_THRESHOLD_MB, use a fraction of RAM.
    - Falls back to DEFAULT_VRAM_FALLBACK_MB if both are 0.
    - For MLX, uses RAM_FRACTION_MLX; for GGUF/Ollama, RAM_FRACTION_GGUF.
    """
    if vram_mb >= VRAM_SMALL_THRESHOLD_MB:
        return vram_mb
    if ram_mb > 0:
        fraction = RAM_FRACTION_MLX if runtime == "mlx" else RAM_FRACTION_GGUF
        return int(ram_mb * fraction)
    return DEFAULT_VRAM_FALLBACK_MB if runtime != "mlx" else DEFAULT_MLX_BUDGET_MB


def estimate_vram_gguf(file_size_bytes: int) -> int:
    """Estimate VRAM needed for a GGUF model from its file size."""
    return int(file_size_bytes / (1024**2) + VRAM_OVERHEAD_GGUF_MB)


def estimate_vram_mlx(safetensor_bytes: int) -> int:
    """Estimate VRAM needed for an MLX model from safetensors file size."""
    return int(safetensor_bytes / (1024**2) + VRAM_OVERHEAD_MLX_MB)


def estimate_vram_ollama(size_b: float) -> int:
    """Estimate VRAM needed for an Ollama library model from param count."""
    return int(size_b * 600 + VRAM_OVERHEAD_OLLAMA_MB)
