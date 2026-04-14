"""Benchmark MLX models on Apple Silicon using mlx-lm.

This module provides a benchmark pipeline that does NOT require Ollama.
Models are downloaded from HuggingFace and run natively via the mlx-lm
library using Apple's MLX framework on unified memory.

Downloads are resumable — interrupted downloads pick up where they left off
thanks to huggingface_hub's partial file (.incomplete) mechanism.

Measures actual Metal GPU memory usage via mlx.core.metal APIs.
"""
from __future__ import annotations

import time

from .benchmark import BenchResult, BENCH_PROMPTS


def mlx_lm_available() -> bool:
    """Check if mlx-lm is installed and importable."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def _metal_memory_mb() -> tuple[int, int]:
    """Return (active_mb, peak_mb) from Metal GPU memory stats."""
    try:
        import mlx.core as mx
        active = mx.metal.get_active_memory() // (1024 * 1024)
        peak = mx.metal.get_peak_memory() // (1024 * 1024)
        return active, peak
    except Exception:
        return 0, 0


def _reset_peak_memory() -> None:
    """Reset the Metal peak memory counter."""
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def is_model_cached(repo_id: str) -> bool:
    """Check if a model is fully cached locally (no network needed).

    Verifies that the repo exists in the HF cache AND has at least one
    .safetensors file downloaded (not just metadata).
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                # Check that actual model files exist, not just metadata.
                for revision in repo.revisions:
                    has_weights = any(
                        f.file_name.endswith(".safetensors")
                        for f in revision.files
                    )
                    if has_weights:
                        return True
        return False
    except Exception:
        return False


def _try_load_cached(repo_id: str) -> str | None:
    """Try to get the local cache path without any network calls.

    Returns the local snapshot path if fully cached, None otherwise.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                for revision in repo.revisions:
                    has_weights = any(
                        f.file_name.endswith(".safetensors")
                        for f in revision.files
                    )
                    if has_weights:
                        return str(revision.snapshot_path)
        return None
    except Exception:
        return None


def download_mlx_model(repo_id: str, verbose: bool = True) -> str | None:
    """Download/resume an MLX model from HuggingFace. Returns local cache path.

    - If fully cached: returns path instantly (no network)
    - If partially cached: resumes download
    - If new: downloads from scratch

    Uses huggingface_hub.snapshot_download() which:
    - Resumes interrupted downloads automatically (.incomplete files)
    - Shows progress bars per file
    - Caches in ~/.cache/huggingface/hub/
    """
    # Fast path: already fully cached → no network needed.
    cached_path = _try_load_cached(repo_id)
    if cached_path:
        if verbose:
            print(f"    {repo_id}: using cached version (no download)")
        return cached_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    try:
        local_dir = snapshot_download(
            repo_id,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.*",
                "*.model",        # sentencepiece
                "*.tiktoken",
            ],
            resume_download=True,
        )
        return local_dir
    except KeyboardInterrupt:
        if verbose:
            print(f"\n    Download paused for {repo_id}. "
                  "Run again to resume where you left off.")
        raise
    except Exception as e:
        if verbose:
            print(f"    Download failed: {e}")
        return None


def benchmark_mlx_model(
    repo_id: str,
    prompts: list[tuple[str, str]] | None = None,
    max_tokens: int = 256,
) -> BenchResult:
    """Benchmark a single MLX model from HuggingFace.

    1. Checks cache first (no network if already downloaded)
    2. Downloads/resumes if needed
    3. Loads via mlx-lm
    4. Measures tokens/second + actual Metal GPU memory
    """
    prompts = prompts or BENCH_PROMPTS

    try:
        from mlx_lm import load, generate
    except ImportError:
        return BenchResult(
            model=repo_id,
            tokens_per_sec=0.0,
            eval_count=0,
            eval_seconds=0.0,
            total_seconds=0.0,
            ok=False,
            error="mlx-lm not installed. Run: pip install 'ollmatuning[mlx]'",
        )

    t0 = time.perf_counter()

    # Step 1: Download/cache check.
    local_path = download_mlx_model(repo_id)
    if local_path is None:
        return BenchResult(
            model=repo_id,
            tokens_per_sec=0.0,
            eval_count=0,
            eval_seconds=0.0,
            total_seconds=time.perf_counter() - t0,
            ok=False,
            error="Download failed (run again to resume)",
        )

    # Step 2: Load into MLX.
    try:
        _reset_peak_memory()
        model, tokenizer = load(local_path)
        vram_mb, _ = _metal_memory_mb()
        _reset_peak_memory()
    except Exception as e:
        return BenchResult(
            model=repo_id,
            tokens_per_sec=0.0,
            eval_count=0,
            eval_seconds=0.0,
            total_seconds=time.perf_counter() - t0,
            ok=False,
            error=f"Load failed: {e}",
        )

    # Step 3: Benchmark.
    total_tokens = 0
    total_eval_s = 0.0

    try:
        # Warm-up: short generation to populate caches.
        generate(model, tokenizer, prompt="Hi.", max_tokens=8)

        for _label, prompt in prompts:
            gen_start = time.perf_counter()
            output = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.0,
            )
            gen_end = time.perf_counter()

            out_tokens = len(tokenizer.encode(output))
            total_tokens += out_tokens
            total_eval_s += gen_end - gen_start

    except Exception as e:
        return BenchResult(
            model=repo_id,
            tokens_per_sec=0.0,
            eval_count=total_tokens,
            eval_seconds=total_eval_s,
            total_seconds=time.perf_counter() - t0,
            ok=False,
            error=f"Generate failed: {e}",
        )

    # Step 4: Measure peak memory.
    _, peak_mb = _metal_memory_mb()

    tps = (total_tokens / total_eval_s) if total_eval_s > 0 else 0.0
    return BenchResult(
        model=repo_id,
        tokens_per_sec=tps,
        eval_count=total_tokens,
        eval_seconds=total_eval_s,
        total_seconds=time.perf_counter() - t0,
        ok=True,
        vram_mb=vram_mb,
        peak_vram_mb=peak_mb if peak_mb > vram_mb else vram_mb,
    )
