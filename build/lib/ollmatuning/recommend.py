"""Find coding/tool-capable Ollama models that fit the detected hardware."""
from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from .system import SystemInfo

OLLAMA_LIBRARY = "https://ollama.com"
USER_AGENT = "ollmatuning/0.1 (+https://github.com/)"

# Curated fallback when the scrape fails — all known to be strong at code + tools.
FALLBACK_MODELS = [
    "qwen2.5-coder",
    "deepseek-coder-v2",
    "qwen2.5",
    "llama3.1",
    "llama3.2",
    "mistral-nemo",
    "codellama",
    "granite-code",
    "codegemma",
]


@dataclass
class Candidate:
    model: str             # e.g. "qwen2.5-coder:7b"
    family: str            # e.g. "qwen2.5-coder"
    size_b: float          # parameter count in billions (best-effort)
    est_vram_mb: int       # rough VRAM need at Q4
    categories: list[str]  # ["code", "tools", ...]

    @property
    def pretty(self) -> str:
        return f"{self.model} (~{self.size_b:g}B, ~{self.est_vram_mb} MB VRAM)"


def _fetch(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return ""


def _parse_library_page(page_html: str) -> list[str]:
    """Extract model family slugs from an ollama.com/search or /library page."""
    # Links look like: <a href="/library/qwen2.5-coder" ...>
    slugs = re.findall(r'href="/library/([a-z0-9][a-z0-9._\-]*)"', page_html)
    seen: list[str] = []
    for s in slugs:
        if s not in seen:
            seen.append(s)
    return seen


def _parse_model_tags(model_page_html: str) -> list[tuple[str, float]]:
    """Return [(tag, size_b)] for a model's detail page."""
    tags: list[tuple[str, float]] = []
    # Tags appear as /library/<family>:<tag>
    raw_tags = re.findall(r"/library/[a-z0-9._\-]+:([a-z0-9._\-]+)", model_page_html)
    seen = set()
    for t in raw_tags:
        if t in seen:
            continue
        seen.add(t)
        size = _tag_to_size_b(t)
        if size > 0:
            tags.append((t, size))
    return tags


def _tag_to_size_b(tag: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", tag.lower())
    if m:
        return float(m.group(1))
    return 0.0


def _estimate_vram_mb(size_b: float) -> int:
    # Q4_K_M rule of thumb: ~0.6 GB per B params + ~1 GB overhead.
    return int(size_b * 600 + 1024)


def discover_models(verbose: bool = False) -> list[str]:
    """Scrape ollama.com for models tagged with code and/or tools support."""
    found: list[str] = []
    for cat in ("code", "tools"):
        url = f"{OLLAMA_LIBRARY}/search?c={cat}"
        if verbose:
            print(f"  fetch {url}")
        page = _fetch(url)
        if page:
            found.extend(_parse_library_page(page))
    # Preserve order, dedupe.
    ordered: list[str] = []
    for f in found:
        if f not in ordered:
            ordered.append(f)
    if not ordered:
        if verbose:
            print("  (scrape leer — nutze Fallback-Liste)")
        ordered = list(FALLBACK_MODELS)
    return ordered


def expand_candidates(
    families: list[str],
    info: SystemInfo,
    verbose: bool = False,
) -> list[Candidate]:
    """For each family, fetch its tags and keep those that fit in available VRAM/RAM."""
    # Budget: prefer VRAM; fall back to 60% of system RAM when no discrete GPU VRAM is known.
    vram = info.usable_vram_mb
    if vram < 2048:
        vram = int(info.ram_mb * 0.6)
    if vram <= 0:
        vram = 8192  # conservative default

    candidates: list[Candidate] = []
    for fam in families:
        url = f"{OLLAMA_LIBRARY}/library/{fam}"
        if verbose:
            print(f"  fetch {url}")
        page = _fetch(url)
        tags = _parse_model_tags(page) if page else []
        if not tags:
            # Heuristic default tags when scrape fails.
            tags = [("7b", 7.0), ("8b", 8.0), ("13b", 13.0), ("14b", 14.0), ("32b", 32.0)]
        for tag, size_b in tags:
            est = _estimate_vram_mb(size_b)
            if est > vram:
                continue
            cats = _guess_categories(fam)
            candidates.append(
                Candidate(
                    model=f"{fam}:{tag}",
                    family=fam,
                    size_b=size_b,
                    est_vram_mb=est,
                    categories=cats,
                )
            )
    # Sort: larger = generally smarter; keep biggest-that-fits first.
    candidates.sort(key=lambda c: (-c.size_b, c.model))
    return candidates


def _guess_categories(family: str) -> list[str]:
    f = family.lower()
    cats: list[str] = []
    if "coder" in f or "code" in f:
        cats.append("code")
    if any(k in f for k in ("qwen2.5", "llama3", "mistral", "nemo", "granite", "command-r", "hermes")):
        cats.append("tools")
    if not cats:
        cats.append("general")
    return cats


def shortlist(candidates: list[Candidate], limit: int = 3) -> list[Candidate]:
    """Pick top-N distinct families so we benchmark a variety, not 5 sizes of one model."""
    seen_families: set[str] = set()
    picked: list[Candidate] = []
    for c in candidates:
        if c.family in seen_families:
            continue
        seen_families.add(c.family)
        picked.append(c)
        if len(picked) >= limit:
            break
    return picked
