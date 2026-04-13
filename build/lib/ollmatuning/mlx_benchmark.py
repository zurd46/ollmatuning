"""Benchmark MLX models on Apple Silicon using mlx-lm.

This module provides a benchmark pipeline that does NOT require Ollama.
Models are downloaded from HuggingFace and run natively via the mlx-lm
library using Apple's MLX framework on unified memory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .benchmark import BenchResult, BENCH_PROMPTS


def mlx_lm_available() -> bool:
    """Check if mlx-lm is installed and importable."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def benchmark_mlx_model(
    repo_id: str,
    prompts: list[tuple[str, str]] | None = None,
    max_tokens: int = 256,
) -> BenchResult:
    """Benchmark a single MLX model from HuggingFace.

    Downloads the model (cached by huggingface_hub), loads it via mlx-lm,
    generates responses, and measures tokens/second.
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
            error="mlx-lm not installed. Run: pip install ollmatuning[mlx]",
        )

    t0 = time.perf_counter()

    try:
        # Load model + tokenizer (auto-downloads from HF hub).
        model, tokenizer = load(repo_id)
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

    total_tokens = 0
    total_eval_s = 0.0

    try:
        # Warm-up: short generation to load weights into memory.
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

            # Count output tokens via tokenizer.
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

    tps = (total_tokens / total_eval_s) if total_eval_s > 0 else 0.0
    return BenchResult(
        model=repo_id,
        tokens_per_sec=tps,
        eval_count=total_tokens,
        eval_seconds=total_eval_s,
        total_seconds=time.perf_counter() - t0,
        ok=True,
    )
