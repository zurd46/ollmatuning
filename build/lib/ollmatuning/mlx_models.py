"""Search HuggingFace for MLX models optimized for Apple Silicon.

MLX models use Apple's MLX framework and run natively on Apple Silicon
with unified memory. They are typically pre-quantized (4-bit or 8-bit)
and stored as .safetensors files.

The mlx-community organization on HuggingFace hosts most MLX-converted
models. This module searches for them, estimates memory usage from
safetensors file sizes, and produces Candidate objects.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .system import SystemInfo

HF_API = "https://huggingface.co/api"
USER_AGENT = "ollmatuning/0.1 (+https://github.com/)"

_QUANT_RE = re.compile(r"[-_]((\d+)[\s-]?bit)", re.IGNORECASE)
_PARAM_RE = re.compile(r"[-_.](\d+(?:\.\d+)?)[Bb][-_.]")


@dataclass
class MLXModel:
    repo_id: str
    downloads: int
    likes: int
    tags: list[str]
    quant_bits: int          # 4, 8, or 0 if unknown
    safetensor_bytes: int    # total size of .safetensors files

    @property
    def est_vram_mb(self) -> int:
        """MLX memory-maps weights. VRAM ~ file size + ~300 MB overhead."""
        return int(self.safetensor_bytes / (1024**2) + 300)


def _fetch_json(url: str, timeout: int = 20) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _parse_quant_bits(repo_id: str) -> int:
    """Extract quantization bits from repo name (e.g. '-4bit' -> 4)."""
    m = _QUANT_RE.search(repo_id)
    if m:
        return int(m.group(2))
    lower = repo_id.lower()
    if "4bit" in lower or "4-bit" in lower or "w4a16" in lower:
        return 4
    if "8bit" in lower or "8-bit" in lower or "w8a16" in lower:
        return 8
    if "3bit" in lower or "3-bit" in lower:
        return 3
    return 0


def _guess_param_size(repo_id: str) -> float:
    """Try to extract parameter count (in billions) from repo name."""
    m = _PARAM_RE.search(repo_id)
    if m:
        return float(m.group(1))
    # Try looser pattern
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", repo_id.replace("-", " "))
    if m2:
        return float(m2.group(1))
    return 0.0


def search_mlx_models(
    query: str = "",
    limit: int = 40,
    verbose: bool = False,
) -> list[MLXModel]:
    """Search HuggingFace for MLX models."""
    params = f"filter=mlx&sort=downloads&direction=-1&limit={limit}"
    if query:
        params += f"&search={urllib.request.quote(query)}"
    url = f"{HF_API}/models?{params}"
    if verbose:
        print(f"  MLX search: {url}")

    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return []

    models: list[MLXModel] = []
    for item in data:
        repo_id = item.get("id", "")
        if not repo_id:
            continue
        tags = item.get("tags", [])
        if "mlx" not in [t.lower() for t in tags]:
            continue
        quant = _parse_quant_bits(repo_id)
        models.append(MLXModel(
            repo_id=repo_id,
            downloads=item.get("downloads", 0),
            likes=item.get("likes", 0),
            tags=tags,
            quant_bits=quant,
            safetensor_bytes=0,
        ))

    if verbose:
        print(f"  MLX found {len(models)} repos")
    return models


def fetch_mlx_model_size(model: MLXModel, verbose: bool = False) -> None:
    """Populate safetensor_bytes by reading the repo's file listing."""
    url = f"{HF_API}/models/{model.repo_id}?blobs=true"
    if verbose:
        print(f"  MLX files: {url}")

    data = _fetch_json(url, timeout=15)
    if not data or not isinstance(data, dict):
        return

    total = 0
    for sib in data.get("siblings", []):
        fname = sib.get("rfilename", "")
        size = sib.get("size", 0) or 0
        if fname.endswith(".safetensors") and size > 0:
            total += size
    model.safetensor_bytes = total


def discover_mlx_models(
    queries: list[str] | None = None,
    limit_per_query: int = 25,
    verbose: bool = False,
    progress_cb=None,
) -> list[MLXModel]:
    """Search HF for MLX models across multiple queries, deduplicate, fetch sizes."""
    if queries is None:
        queries = ["mlx coder", "mlx instruct", "mlx tool", "mlx"]

    seen: set[str] = set()
    all_models: list[MLXModel] = []
    for q in queries:
        for m in search_mlx_models(query=q, limit=limit_per_query, verbose=verbose):
            if m.repo_id not in seen:
                seen.add(m.repo_id)
                all_models.append(m)

    if verbose:
        print(f"  MLX total unique repos: {len(all_models)}")

    # Fetch file sizes in parallel.
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_mlx_model_size, m, verbose): m for m in all_models}
        done = 0
        for fut in as_completed(futures):
            done += 1
            m = futures[fut]
            if progress_cb:
                progress_cb(done, len(futures), m.repo_id)
            try:
                fut.result()
            except Exception:
                pass

    # Keep only repos that have safetensors files.
    with_files = [m for m in all_models if m.safetensor_bytes > 0]
    with_files.sort(key=lambda m: m.downloads, reverse=True)

    if verbose:
        print(f"  MLX repos with safetensors: {len(with_files)}")

    return with_files


def expand_mlx_candidates(
    models: list[MLXModel],
    info: SystemInfo,
    verbose: bool = False,
) -> list["Candidate"]:
    """Convert MLX models into Candidate objects that fit memory budget."""
    from .recommend import Candidate

    # Apple Silicon uses unified memory — entire RAM is available.
    budget_mb = int(info.ram_mb * 0.75) if info.ram_mb > 0 else 16384

    candidates: list[Candidate] = []
    for m in models:
        est_vram = m.est_vram_mb
        if est_vram > budget_mb:
            if verbose:
                print(f"  MLX skip {m.repo_id} — {est_vram} MB > {budget_mb} MB budget")
            continue

        size_b = _guess_param_size(m.repo_id)
        cats: list[str] = []
        lower_id = m.repo_id.lower()
        if any(k in lower_id for k in ("code", "coder", "coding")):
            cats.append("code")
        if any(k in lower_id for k in ("tool", "function", "func-call")):
            cats.append("tools")
        if not cats:
            cats.append("general")

        quant_label = f"{m.quant_bits}bit" if m.quant_bits > 0 else "fp16"
        candidates.append(Candidate(
            model=m.repo_id,
            family=m.repo_id.split("/")[-1],
            size_b=size_b if size_b > 0 else m.safetensor_bytes / (1024**3) * 0.5,
            est_vram_mb=est_vram,
            categories=cats,
            source="huggingface",
            runtime="mlx",
        ))

    candidates.sort(key=lambda c: (0 if "code" in c.categories else 1, -c.size_b, c.model))
    return candidates
