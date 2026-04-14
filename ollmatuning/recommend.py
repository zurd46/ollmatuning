"""Find tool-capable Ollama models that fit the detected hardware.

Hard rule: every candidate MUST support tool/function calling.
Source of truth: ollama.com's `?c=tools` category listing.
"""
from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .system import SystemInfo
from .utils import (
    OLLAMA_LIBRARY_URL,
    USER_AGENT,
    HTTP_TIMEOUT_SHORT,
    HTTP_TIMEOUT_MEDIUM,
    OLLAMA_VERIFY_WORKERS,
    fetch_text,
    compute_vram_budget,
    estimate_vram_ollama,
)

logger = logging.getLogger(__name__)

# Hard rule: NO hardcoded model names. Everything comes from live research
# against ollama.com's search and library pages.


@dataclass
class Candidate:
    model: str
    family: str
    size_b: float
    est_vram_mb: int
    categories: list[str]  # always contains "tools"; may also contain "code"
    source: str = "ollama"   # "ollama" or "huggingface"
    runtime: str = "ollama"  # "ollama" or "mlx"

    @property
    def pretty(self) -> str:
        return f"{self.model} (~{self.size_b:g}B, ~{self.est_vram_mb} MB VRAM)"


def _head_ok(url: str, timeout: int = HTTP_TIMEOUT_SHORT) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        if e.code in (403, 405):
            return bool(fetch_text(url, timeout=timeout))
        return False
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def verify_model_exists(model: str) -> bool:
    """Check that <family>:<tag> resolves on ollama.com."""
    if ":" not in model:
        return False
    family, tag = model.split(":", 1)
    return _head_ok(f"{OLLAMA_LIBRARY_URL}/library/{family}:{tag}")


def _parse_library_page(page_html: str) -> list[str]:
    slugs = re.findall(r'href="/library/([a-z0-9][a-z0-9._\-]*)"', page_html)
    seen: list[str] = []
    for s in slugs:
        if s not in seen:
            seen.append(s)
    return seen


def _fetch_category(cat: str, verbose: bool = False) -> list[str]:
    """Fetch a category page plus its paginated continuations."""
    families: list[str] = []
    # ollama.com uses ?c=<cat>&p=<page> — walk pages until empty.
    for page_num in range(1, 11):  # 10 pages cap is plenty
        url = (
            f"{OLLAMA_LIBRARY_URL}/search?c={cat}"
            if page_num == 1
            else f"{OLLAMA_LIBRARY_URL}/search?c={cat}&p={page_num}"
        )
        if verbose:
            logger.debug("fetch %s", url)
        page = fetch_text(url)
        if not page:
            break
        found = _parse_library_page(page)
        new = [f for f in found if f not in families]
        if not new:
            break
        families.extend(new)
    return families


def _fetch_full_library(verbose: bool = False) -> list[str]:
    """Scrape the main /library listing for additional families."""
    families: list[str] = []
    for page_num in range(1, 11):
        url = (
            f"{OLLAMA_LIBRARY_URL}/library"
            if page_num == 1
            else f"{OLLAMA_LIBRARY_URL}/library?p={page_num}"
        )
        if verbose:
            logger.debug("fetch %s", url)
        page = fetch_text(url)
        if not page:
            break
        found = _parse_library_page(page)
        new = [f for f in found if f not in families]
        if not new:
            break
        families.extend(new)
    return families


_CAPABILITY_RE = re.compile(
    r">(tools|embedding|vision|thinking)(?:<|[\"'])", re.IGNORECASE
)


def model_capabilities(family: str) -> set[str]:
    """Return the capability badges shown on https://ollama.com/library/<family>."""
    page = fetch_text(f"{OLLAMA_LIBRARY_URL}/library/{family}")
    if not page:
        return set()
    return {m.lower() for m in _CAPABILITY_RE.findall(page)}


def model_has_tools(family: str) -> bool:
    """True iff ollama.com's capability badges for this family include 'tools'."""
    return "tools" in model_capabilities(family)


def discover_models(
    verbose: bool = False,
    progress_cb=None,
) -> tuple[list[str], set[str]]:
    """Return (tool_families, code_family_set) — 100 % live from ollama.com.

    Zero hardcoded model names. Strategy:
      1. Fetch the full `/library` — ~200+ families, the authoritative list.
      2. For every family, fetch its detail page in parallel and read the
         capability badges. Keep only those that advertise 'tools'.
      3. Also fetch `/search?c=code` to mark which tool-capable models are
         also code-capable (used for ranking, not gating).

    Hard rule: NOTHING without a verified 'tools' badge is returned.
    """
    library = _fetch_full_library(verbose=verbose)
    if verbose:
        logger.debug("full library: %d families", len(library))
    if not library:
        return [], set()

    code = _fetch_category("code", verbose=verbose)
    code_set = set(code)
    if verbose:
        logger.debug("code category: %d families", len(code))

    # Parallel capability probe — the slow step, so we fan out.
    tool_families: list[str] = []
    with ThreadPoolExecutor(max_workers=OLLAMA_VERIFY_WORKERS) as pool:
        futures = {pool.submit(model_capabilities, fam): fam for fam in library}
        done = 0
        for fut in as_completed(futures):
            fam = futures[fut]
            done += 1
            if progress_cb:
                progress_cb(done, len(futures), fam)
            try:
                caps = fut.result()
            except Exception:
                logger.debug("Failed to get capabilities for %s", fam, exc_info=True)
                caps = set()
            if "tools" in caps:
                tool_families.append(fam)
                if verbose:
                    logger.debug("tools ok: %s  %s", fam, sorted(caps))

    # Preserve library order (popularity-ranked by ollama.com).
    order_index = {fam: i for i, fam in enumerate(library)}
    tool_families.sort(key=lambda f: order_index.get(f, 1 << 30))

    # Rank: code AND tools first, then tools only.
    intersection = [f for f in tool_families if f in code_set]
    rest = [f for f in tool_families if f not in code_set]
    ordered = intersection + rest
    return ordered, code_set


def _parse_model_tags(model_page_html: str) -> list[tuple[str, float]]:
    tags: list[tuple[str, float]] = []
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


def _fetch_authoritative_tags(family: str, verbose: bool = False) -> list[tuple[str, float]]:
    url = f"{OLLAMA_LIBRARY_URL}/library/{family}/tags"
    if verbose:
        logger.debug("fetch %s", url)
    page = fetch_text(url)
    if not page:
        return []
    tags = _parse_model_tags(page)
    if not tags:
        main = fetch_text(f"{OLLAMA_LIBRARY_URL}/library/{family}")
        if main:
            tags = _parse_model_tags(main)
    return tags


def expand_candidates(
    families: list[str],
    info: SystemInfo,
    code_set: set[str] | None = None,
    verbose: bool = False,
) -> list[Candidate]:
    """Expand each tool-capable family into size-filtered Candidate objects.

    Safety: we NEVER invent tags — families with no parseable published tags are
    dropped. Families are assumed tool-capable (the caller filters this), so every
    returned Candidate carries the 'tools' category.
    """
    code_set = code_set or set()

    vram = compute_vram_budget(info.usable_vram_mb, info.ram_mb, runtime="ollama")

    candidates: list[Candidate] = []
    for fam in families:
        tags = _fetch_authoritative_tags(fam, verbose=verbose)
        if not tags:
            if verbose:
                logger.debug("skip %s — no published tags", fam)
            continue
        cats = ["tools"]
        if fam in code_set:
            cats.insert(0, "code")
        for tag, size_b in tags:
            est = estimate_vram_ollama(size_b)
            if est > vram:
                continue
            candidates.append(
                Candidate(
                    model=f"{fam}:{tag}",
                    family=fam,
                    size_b=size_b,
                    est_vram_mb=est,
                    categories=list(cats),
                )
            )
    # Sort: code+tools first, then bigger-first within that.
    candidates.sort(key=lambda c: (0 if "code" in c.categories else 1, -c.size_b, c.model))
    return candidates


def verify_candidates(
    candidates: list[Candidate],
    verbose: bool = False,
) -> list[Candidate]:
    """HEAD-check every candidate model:tag in parallel. Tool capability is
    already enforced upstream in `discover_models`, so this only gates tag existence.
    """
    verified: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=OLLAMA_VERIFY_WORKERS) as pool:
        futures = {pool.submit(verify_model_exists, c.model): c for c in candidates}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                verified.append(c)
                if verbose:
                    logger.debug("ok %s", c.model)
            elif verbose:
                logger.debug("x %s (not on ollama.com)", c.model)
    # Preserve the original ranking order.
    order = {c.model: i for i, c in enumerate(candidates)}
    verified.sort(key=lambda c: order[c.model])
    return verified


def shortlist(candidates: list[Candidate], limit: int | None = None) -> list[Candidate]:
    """Pick one distinct family per entry. If `limit` is None, return all families.

    This keeps the benchmark set diverse (max one size per family) while still
    covering every eligible family when the caller wants the full sweep.
    """
    seen_families: set[str] = set()
    picked: list[Candidate] = []
    for c in candidates:
        if c.family in seen_families:
            continue
        seen_families.add(c.family)
        picked.append(c)
        if limit is not None and len(picked) >= limit:
            break
    return picked