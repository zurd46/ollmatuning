"""Search HuggingFace for quantized GGUF models compatible with Ollama.

Ollama >= 0.4 can pull HF GGUF models directly via:
    ollama run hf.co/{user}/{repo}           (default quant)
    ollama run hf.co/{user}/{repo}:{quant}   (specific quant, e.g. Q4_K_M)

This module uses the HuggingFace API to discover popular GGUF models,
extract available quantizations from file listings, estimate VRAM from
file sizes, and produce Candidate objects that slot into the existing
recommend → benchmark pipeline.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .system import SystemInfo
from .utils import (
    HF_API_BASE,
    USER_AGENT,
    HTTP_TIMEOUT_MEDIUM,
    HTTP_TIMEOUT_LONG,
    HF_FETCH_WORKERS,
    fetch_json,
    guess_param_size,
    detect_categories,
    compute_vram_budget,
    estimate_vram_gguf,
)

logger = logging.getLogger(__name__)

# Quantization levels ordered from smallest to largest (roughly).
QUANT_PREFERENCE = [
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
    "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
    "Q6_K",
    "Q8_0",
    "Q3_K_M", "Q3_K_S", "Q3_K_L",
    "Q2_K",
    "IQ4_XS", "IQ4_NL",
    "IQ3_XXS", "IQ3_XS",
    "IQ2_XXS", "IQ2_XS",
]

# Regex to extract quant tag from GGUF filenames.
_QUANT_RE = re.compile(
    r"[.\-_]((?:I?Q\d[_.]\w+|Q\d_\d|F16|F32|BF16))\s*\.gguf$",
    re.IGNORECASE,
)


@dataclass
class GGUFFile:
    filename: str
    quant: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def est_vram_mb(self) -> int:
        """GGUF file size ≈ model weight size. VRAM ≈ file + overhead."""
        return estimate_vram_gguf(self.size_bytes)


@dataclass
class HFModel:
    repo_id: str          # e.g. "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
    downloads: int
    likes: int
    tags: list[str]
    gguf_files: list[GGUFFile]

    @property
    def ollama_base(self) -> str:
        return f"hf.co/{self.repo_id}"

    def best_quant_for_vram(self, vram_mb: int) -> GGUFFile | None:
        """Pick the best quantization that fits in VRAM budget."""
        fitting = [f for f in self.gguf_files if f.est_vram_mb <= vram_mb]
        if not fitting:
            return None
        # Prefer Q4_K_M > Q5_K_M > Q6_K > Q8_0 > others
        pref_map = {q.upper(): i for i, q in enumerate(QUANT_PREFERENCE)}

        def rank(f: GGUFFile) -> tuple[int, int]:
            idx = pref_map.get(f.quant.upper(), 999)
            return (idx, -f.size_bytes)

        fitting.sort(key=rank)
        return fitting[0]


def search_gguf_models(
    query: str = "",
    limit: int = 40,
    verbose: bool = False,
) -> list[HFModel]:
    """Search HuggingFace for GGUF models.

    Default search (no query) returns the most downloaded GGUF repos.
    Useful queries: "coder", "instruct", "tool", "function-calling".
    """
    params = (
        f"filter=gguf&sort=downloads&direction=-1&limit={limit}"
    )
    if query:
        params += f"&search={urllib.request.quote(query)}"
    url = f"{HF_API_BASE}/models?{params}"
    if verbose:
        logger.debug("HF search: %s", url)

    data = fetch_json(url)
    if not data or not isinstance(data, list):
        return []

    models: list[HFModel] = []
    for item in data:
        repo_id = item.get("id", "")
        if not repo_id:
            continue
        tags = item.get("tags", [])
        # Skip non-GGUF repos that somehow appeared
        if "gguf" not in [t.lower() for t in tags]:
            continue
        models.append(HFModel(
            repo_id=repo_id,
            downloads=item.get("downloads", 0),
            likes=item.get("likes", 0),
            tags=tags,
            gguf_files=[],
        ))

    if verbose:
        logger.debug("HF found %d GGUF repos", len(models))
    return models


def fetch_gguf_files(model: HFModel, verbose: bool = False) -> None:
    """Populate model.gguf_files by reading the repo's file listing."""
    url = f"{HF_API_BASE}/models/{model.repo_id}?blobs=true"
    if verbose:
        logger.debug("HF files: %s", url)

    data = fetch_json(url, timeout=HTTP_TIMEOUT_MEDIUM)
    if not data or not isinstance(data, dict):
        return

    siblings = data.get("siblings", [])
    for sib in siblings:
        fname = sib.get("rfilename", "")
        if not fname.lower().endswith(".gguf"):
            continue
        size = sib.get("size", 0) or 0
        if size <= 0:
            continue
        m = _QUANT_RE.search(fname)
        quant = m.group(1).upper().replace(".", "_") if m else "unknown"
        model.gguf_files.append(GGUFFile(
            filename=fname,
            quant=quant,
            size_bytes=size,
        ))


def discover_hf_models(
    queries: list[str] | None = None,
    limit_per_query: int = 25,
    verbose: bool = False,
    progress_cb=None,
) -> list[HFModel]:
    """Search HF for GGUF models across multiple queries, deduplicate, fetch file info."""
    if queries is None:
        queries = ["coder GGUF", "instruct GGUF", "tool calling GGUF", "GGUF"]

    # Gather unique repos across queries.
    seen: set[str] = set()
    all_models: list[HFModel] = []
    for q in queries:
        for m in search_gguf_models(query=q, limit=limit_per_query, verbose=verbose):
            if m.repo_id not in seen:
                seen.add(m.repo_id)
                all_models.append(m)

    if verbose:
        logger.debug("HF total unique repos: %d", len(all_models))

    # Fetch file listings in parallel.
    with ThreadPoolExecutor(max_workers=HF_FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_gguf_files, m, verbose): m for m in all_models}
        done = 0
        for fut in as_completed(futures):
            done += 1
            m = futures[fut]
            if progress_cb:
                progress_cb(done, len(futures), m.repo_id)
            try:
                fut.result()
            except Exception:
                logger.debug("Failed to fetch files for %s", m.repo_id, exc_info=True)

    # Keep only repos that have actual GGUF files.
    with_files = [m for m in all_models if m.gguf_files]
    # Sort by downloads descending.
    with_files.sort(key=lambda m: m.downloads, reverse=True)

    if verbose:
        logger.debug("HF repos with GGUF files: %d", len(with_files))

    return with_files


def expand_hf_candidates(
    models: list[HFModel],
    info: SystemInfo,
    verbose: bool = False,
) -> list["Candidate"]:
    """Convert HF models into Candidate objects that fit VRAM budget."""
    from .recommend import Candidate

    vram = compute_vram_budget(info.usable_vram_mb, info.ram_mb, runtime="ollama")

    candidates: list[Candidate] = []
    for m in models:
        best = m.best_quant_for_vram(vram)
        if not best:
            if verbose:
                logger.debug("HF skip %s — no quant fits %d MB", m.repo_id, vram)
            continue

        size_b = guess_param_size(m.repo_id)
        # Determine categories from tags/name.
        lower_tags = " ".join(m.tags).lower() + " " + m.repo_id.lower()
        cats = detect_categories(lower_tags)

        ollama_model = f"hf.co/{m.repo_id}:{best.quant}"
        candidates.append(Candidate(
            model=ollama_model,
            family=m.repo_id.split("/")[-1],
            size_b=size_b if size_b > 0 else best.size_gb * 0.5,
            est_vram_mb=best.est_vram_mb,
            categories=cats,
            source="huggingface",
        ))

    # Sort: code first, then larger models first.
    candidates.sort(key=lambda c: (0 if "code" in c.categories else 1, -c.size_b, c.model))
    return candidates