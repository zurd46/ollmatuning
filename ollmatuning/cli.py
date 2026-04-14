"""ollmatuning — CLI entry point with OS-aware model selection.

Default behavior based on platform:
  - macOS Apple Silicon -> MLX models from HuggingFace (native, fastest)
  - Everything else     -> GGUF models from HuggingFace (via Ollama)

Override with --gguf (force GGUF) or --mlx (force MLX).
Legacy Ollama library search available with --ollama.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from . import __version__
from . import ui
from .system import detect_system, is_apple_silicon, SystemInfo
from .recommend import (
    discover_models, expand_candidates, shortlist,
    verify_candidates, verify_model_exists, Candidate,
)
from .benchmark import (
    ollama_is_up, list_local_models, pull_model, benchmark_model, BenchResult,
)
from .huggingface import discover_hf_models, expand_hf_candidates
from .mlx_models import discover_mlx_models, expand_mlx_candidates
from .mlx_benchmark import mlx_lm_available, benchmark_mlx_model
from .utils import setup_logging

CONFIG_PATH = Path.home() / ".ollmatuning" / "config.json"


# ---------------------------------------------------------------------------
# Helpers: model search
# ---------------------------------------------------------------------------

def _resolve_runtime(args: argparse.Namespace) -> str:
    """Determine runtime: 'mlx' or 'ollama' based on OS and flags."""
    if getattr(args, "mlx", False):
        return "mlx"
    if getattr(args, "gguf", False):
        return "ollama"
    if is_apple_silicon():
        return "mlx"
    return "ollama"


def _search_mlx(info: SystemInfo, verbose: bool = False) -> list[Candidate]:
    """Search HuggingFace for MLX models optimized for Apple Silicon."""
    ui.step("Searching MLX models on HuggingFace (Apple Silicon optimized) ...")
    mlx_models = discover_mlx_models(verbose=verbose)
    if not mlx_models:
        ui.warn("No MLX models found on HuggingFace.")
        return []
    ui.info(f"Found {len(mlx_models)} MLX repos on HuggingFace")
    cands = expand_mlx_candidates(mlx_models, info, verbose=verbose)
    if cands:
        ui.info(f"{len(cands)} MLX models fit your memory budget")
    return cands


def _search_gguf(info: SystemInfo, verbose: bool = False) -> list[Candidate]:
    """Search HuggingFace for quantized GGUF models."""
    ui.step("Searching quantized GGUF models on HuggingFace ...")
    hf_models = discover_hf_models(verbose=verbose)
    if not hf_models:
        ui.warn("No GGUF models found on HuggingFace.")
        return []
    ui.info(f"Found {len(hf_models)} GGUF repos on HuggingFace")
    cands = expand_hf_candidates(hf_models, info, verbose=verbose)
    if cands:
        ui.info(f"{len(cands)} quantized models fit your VRAM budget")
    return cands


def _search_ollama(info: SystemInfo, verbose: bool = False) -> list[Candidate]:
    """Legacy: search ollama.com library for tool-capable models."""
    ui.step("Searching models on ollama.com ...")
    families, code_set = discover_models(verbose=verbose)
    ui.show_families(families, source="ollama.com")
    cands = expand_candidates(families, info, code_set=code_set, verbose=verbose)
    return cands


def _search_models(
    runtime: str,
    info: SystemInfo,
    verbose: bool = False,
    include_ollama: bool = False,
) -> list[Candidate]:
    """Search for models based on the resolved runtime."""
    if runtime == "mlx":
        cands = _search_mlx(info, verbose=verbose)
    else:
        cands = _search_gguf(info, verbose=verbose)

    if include_ollama:
        ollama_cands = _search_ollama(info, verbose=verbose)
        if ollama_cands:
            verified = _pick_verified_ollama(ollama_cands, limit=None, verbose=verbose)
            cands = _sort_candidates(cands + verified)

    return cands


def _pick_verified_ollama(
    cands: list[Candidate],
    limit: int | None,
    verbose: bool = False,
) -> list[Candidate]:
    """Shortlist and HEAD-verify Ollama candidates."""
    pool = shortlist(cands, limit=limit)
    ui.info(f"Verifying {len(pool)} Ollama candidates ...")
    verified = verify_candidates(pool, verbose=verbose)
    return verified


def _sort_candidates(cands: list[Candidate]) -> list[Candidate]:
    """Sort: code+tools first, then code, then tools, then general."""
    def sort_key(c: Candidate) -> tuple:
        cats = c.categories
        has_code = "code" in cats
        has_tools = "tools" in cats
        tier = 0 if (has_code and has_tools) else 1 if has_code else 2 if has_tools else 3
        return (tier, -c.size_b, c.model)
    return sorted(cands, key=sort_key)


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------

def _run_benchmark_pipeline(
    candidates: list[Candidate],
    allow_download: bool = False,
) -> list[BenchResult]:
    """Benchmark a list of candidates, dispatching MLX vs Ollama per model."""
    from .mlx_benchmark import is_model_cached

    local = set(list_local_models()) if ollama_is_up() else set()
    results: list[BenchResult] = []
    skipped_count = 0
    download_skipped: list[str] = []

    with ui.make_progress() as progress:
        task = progress.add_task("Benchmark pipeline", total=len(candidates))
        for c in candidates:
            model = c.model
            try:
                if c.runtime == "mlx":
                    cached = is_model_cached(model)
                    if cached:
                        progress.update(task, description=f"[bright_cyan]{model}[/bright_cyan] -> cached, benchmarking")
                    elif allow_download:
                        progress.update(task, description=f"[yellow]{model}[/yellow] -> downloading (Ctrl+C to skip)")
                    else:
                        progress.update(task, description=f"[dim]{model}[/dim] -> not cached, skipping")
                        download_skipped.append(model)
                        results.append(BenchResult(model, 0, 0, 0, 0, False, "not downloaded (use --download)"))
                        progress.advance(task)
                        continue
                    r = benchmark_mlx_model(model)
                else:
                    progress.update(task, description=f"[bright_cyan]{model}[/bright_cyan] -> check")
                    already_local = model in local or any(
                        x == model or x.startswith(model + ":") for x in local
                    )
                    if not already_local:
                        progress.update(task, description=f"[yellow]{model}[/yellow] -> pull (Ctrl+C to skip)")
                        if not pull_model(model, verbose=False):
                            ui.error(f"Pull failed: {model}")
                            results.append(BenchResult(model, 0, 0, 0, 0, False, "pull failed"))
                            progress.advance(task)
                            continue
                    progress.update(task, description=f"[bright_magenta]{model}[/bright_magenta] -> benchmark")
                    r = benchmark_model(model)

                if r.ok:
                    vram_info = f" | {r.vram_mb / 1024:.1f} GB VRAM" if r.vram_mb else ""
                    ui.success(f"{model}: [bold bright_green]{r.tokens_per_sec:.2f} tok/s[/bold bright_green]{vram_info}")
                else:
                    ui.error(f"{model}: {r.error[:80]}")
                results.append(r)

            except KeyboardInterrupt:
                skipped_count += 1
                ui.warn(f"Skipped {model} (Ctrl+C). Download progress saved — rerun to resume.")
                results.append(BenchResult(model, 0, 0, 0, 0, False, "skipped by user"))

            progress.advance(task)

    if skipped_count:
        ui.info(f"{skipped_count} model(s) skipped. Run again to resume downloads.")
    if download_skipped:
        ui.warn(f"{len(download_skipped)} model(s) not downloaded — skipped:")
        for m in download_skipped:
            ui.info(f"  - {m}")
        ui.info("Re-run with [bold bright_cyan]--download[/bold bright_cyan] to fetch and benchmark them.")

    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace) -> int:
    info = detect_system()
    if getattr(args, "json", False):
        print(json.dumps(info.to_dict(), indent=2))
        return 0
    ui.show_banner()
    ui.show_system(info)
    runtime = _resolve_runtime(args)
    ui.info(f"Recommended runtime: [bold bright_cyan]{runtime.upper()}[/bold bright_cyan]"
            f" ({'Apple Silicon detected' if runtime == 'mlx' else 'GGUF via Ollama'})")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    ui.show_banner()
    info = detect_system()
    ui.show_system(info)

    runtime = _resolve_runtime(args)
    ui.info(f"Runtime: [bold bright_cyan]{runtime.upper()}[/bold bright_cyan]")

    cands = _search_models(
        runtime, info,
        verbose=args.verbose,
        include_ollama=getattr(args, "ollama", False),
    )

    if not cands:
        ui.error("No models found that fit your hardware.")
        return 1

    if args.limit is not None:
        cands = cands[:args.limit]

    ui.show_candidates(cands, title=f"Top {len(cands)} models for your hardware")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    use_json = getattr(args, "json", False)
    if not use_json:
        ui.show_banner()
        info = detect_system()
        ui.show_system(info)
    else:
        info = detect_system()

    runtime = _resolve_runtime(args)
    if not use_json:
        ui.info(f"Runtime: [bold bright_cyan]{runtime.upper()}[/bold bright_cyan]")

    # Pre-flight checks
    if runtime == "mlx" and not mlx_lm_available():
        if use_json:
            print(json.dumps({"error": "mlx-lm not installed", "runtime": "mlx"}))
        else:
            ui.error("mlx-lm is not installed.")
            ui.info("Install it with:  [bold bright_cyan]pip install 'ollmatuning[mlx]'[/bold bright_cyan]")
        return 2

    if runtime == "ollama" and not ollama_is_up():
        if use_json:
            print(json.dumps({"error": "Ollama server not running", "runtime": "ollama"}))
        else:
            ui.error("Ollama server is not running on 127.0.0.1:11434.")
            ui.info("Start it with:  [bold bright_cyan]ollama serve[/bold bright_cyan]")
        return 2

    if args.models:
        if not use_json:
            ui.info(f"Benchmarking explicit models: [bold]{', '.join(args.models)}[/bold]")
        candidates = [
            Candidate(
                model=m, family=m.split("/")[-1], size_b=0,
                est_vram_mb=0, categories=[], source="manual", runtime=runtime,
            )
            for m in args.models
        ]
    else:
        cands = _search_models(
            runtime, info,
            verbose=args.verbose,
            include_ollama=getattr(args, "ollama", False),
        )
        if not cands:
            if use_json:
                print(json.dumps({"error": "No models found", "runtime": runtime}))
            else:
                ui.error("No models found.")
            return 1

        if args.limit is not None:
            cands = cands[:args.limit]

        if not use_json:
            ui.show_candidates(cands, title=f"{len(cands)} candidates to benchmark")
        candidates = cands

    results = _run_benchmark_pipeline(candidates, allow_download=getattr(args, "download", False))
    if not use_json:
        ui.show_results(results)

    ok_results = sorted(
        [r for r in results if r.ok],
        key=lambda r: r.tokens_per_sec,
        reverse=True,
    )
    if not ok_results:
        if use_json:
            print(json.dumps({"error": "No model benchmarked successfully", "results": [r.to_dict() for r in results]}))
        else:
            ui.error("No model benchmarked successfully.")
        return 1

    best = ok_results[0]
    if not use_json:
        ui.show_winner(best)

    if args.set_best:
        _save_config({
            "best_model": best.model,
            "tokens_per_sec": best.tokens_per_sec,
            "runtime": runtime,
            "vram_mb": best.vram_mb,
            "peak_vram_mb": best.peak_vram_mb,
        })
        if not use_json:
            ui.success(f"Saved to [bold]{CONFIG_PATH}[/bold]")
            _print_env_hint(best.model, runtime)

    if use_json:
        output = {
            "runtime": runtime,
            "best": best.to_dict(),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))

    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """All-in-one: detect -> search -> benchmark -> set winner."""
    use_json = getattr(args, "json", False)
    if not use_json:
        ui.show_banner()

    ui.step("Step 1/4: detecting hardware and drivers")
    info = detect_system()
    ui.show_system(info)

    runtime = _resolve_runtime(args)
    if not use_json:
        ui.info(f"Auto-detected runtime: [bold bright_cyan]{runtime.upper()}[/bold bright_cyan]")

    # Pre-flight checks
    if runtime == "mlx":
        if not mlx_lm_available():
            ui.error("mlx-lm is not installed (required for MLX on Apple Silicon).")
            ui.info("Install with:  [bold bright_cyan]pip install 'ollmatuning[mlx]'[/bold bright_cyan]")
            return 2
    else:
        if not ollama_is_up():
            ui.error("Ollama server is not running on 127.0.0.1:11434.")
            ui.info("Start it with:  [bold bright_cyan]ollama serve[/bold bright_cyan]")
            return 2

    fmt_name = "MLX" if runtime == "mlx" else "GGUF"
    ui.step(f"Step 2/4: searching {fmt_name} models on HuggingFace")
    cands = _search_models(
        runtime, info,
        verbose=args.verbose,
        include_ollama=getattr(args, "ollama", False),
    )
    if not cands:
        ui.error("No models found for your hardware.")
        return 1

    if args.limit is not None:
        cands = cands[:args.limit]

    ui.step("Step 3/4: selected candidates")
    ui.show_candidates(cands, title=f"Top {len(cands)} {fmt_name} models for your hardware")

    ui.step("Step 4/4: benchmarking tok/s")
    results = _run_benchmark_pipeline(cands, allow_download=True)
    ui.show_results(results)

    ok = sorted([r for r in results if r.ok], key=lambda r: r.tokens_per_sec, reverse=True)
    if not ok:
        ui.error("No model benchmarked successfully.")
        return 1

    best = ok[0]
    ui.show_winner(best)

    if not args.no_save:
        _save_config({
            "best_model": best.model,
            "tokens_per_sec": best.tokens_per_sec,
            "runtime": runtime,
            "vram_mb": best.vram_mb,
            "peak_vram_mb": best.peak_vram_mb,
        })
        if not use_json:
            ui.success(f"Saved to [bold]{CONFIG_PATH}[/bold]")
            _print_env_hint(best.model, runtime)

    if use_json:
        output = {
            "runtime": runtime,
            "best": {"model": best.model, "tokens_per_sec": best.tokens_per_sec, "vram_mb": best.vram_mb},
            "results": [{"model": r.model, "tokens_per_sec": r.tokens_per_sec, "ok": r.ok, "error": r.error or ""} for r in results],
        }
        print(json.dumps(output, indent=2))

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    ui.show_banner()
    if not CONFIG_PATH.exists():
        ui.warn("No configuration yet. Run [bold]ollmatuning benchmark --set-best[/bold] first.")
        return 1
    data = json.loads(CONFIG_PATH.read_text())
    from rich.panel import Panel
    from rich.box import ROUNDED
    body = "\n".join(f"[bold yellow]{k}[/bold yellow]: [white]{v}[/white]" for k, v in data.items())
    ui.console.print(Panel(body, title="[bold]Saved configuration[/bold]",
                           border_style="bright_green", box=ROUNDED))
    return 0


# ---------------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------------

def _save_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(data)
    CONFIG_PATH.write_text(json.dumps(existing, indent=2))


def _print_env_hint(model: str, runtime: str) -> None:
    if runtime == "mlx":
        ui.info(f'MLX model:   [bright_cyan]{model}[/bright_cyan]')
        ui.info(f'Run with:    [bright_cyan]mlx_lm.generate --model {model}[/bright_cyan]')
    elif os.name == "nt":
        ui.info(f'PowerShell:  [bright_cyan]$env:OLLAMA_MODEL = "{model}"[/bright_cyan]')
        ui.info(f'cmd.exe:     [bright_cyan]set OLLAMA_MODEL={model}[/bright_cyan]')
    else:
        ui.info(f'bash/zsh:    [bright_cyan]export OLLAMA_MODEL="{model}"[/bright_cyan]')


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _add_common_flags(sp: argparse.ArgumentParser) -> None:
    """Add flags shared by auto/recommend/benchmark."""
    sp.add_argument("--limit", type=int, default=6,
                    help="Cap number of candidates (default: 6, use 0 for all)")
    sp.add_argument("--mlx", action="store_true",
                    help="Force MLX runtime (Apple Silicon only)")
    sp.add_argument("--gguf", action="store_true",
                    help="Force GGUF runtime via Ollama (any platform)")
    sp.add_argument("--ollama", action="store_true",
                    help="Also search ollama.com library")
    sp.add_argument("-v", "--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ollmatuning",
        description=(
            "Find the best quantized model for your hardware. "
            "Auto-detects Apple Silicon (MLX) vs GGUF (Ollama)."
        ),
    )
    p.add_argument("-V", "--version", action="version", version=f"ollmatuning {__version__}")
    sub = p.add_subparsers(dest="command", required=False)

    sp = sub.add_parser("auto", help="All-in-one (detect -> search -> benchmark -> set)")
    _add_common_flags(sp)
    sp.add_argument("--download", action="store_true",
                    help="Download missing models (default: only benchmark cached)")
    sp.add_argument("--no-save", action="store_true", help="Do not save the winner")
    sp.add_argument("--json", action="store_true", help="Output results as JSON")
    sp.set_defaults(func=cmd_auto)

    sp = sub.add_parser("detect", help="Show hardware, drivers, and recommended runtime")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--mlx", action="store_true")
    sp.add_argument("--gguf", action="store_true")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("recommend", help="Find matching models (no download)")
    _add_common_flags(sp)
    sp.set_defaults(func=cmd_recommend)

    sp = sub.add_parser("benchmark", help="Pull/download candidates and measure tok/s")
    _add_common_flags(sp)
    sp.add_argument("--models", nargs="+", help="Explicit models (skips auto selection)")
    sp.add_argument("--download", action="store_true",
                    help="Download missing models (default: only benchmark cached)")
    sp.add_argument("--set-best", action="store_true", help="Save winner to ~/.ollmatuning/config.json")
    sp.add_argument("--json", action="store_true", help="Output results as JSON")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("show", help="Show saved configuration")
    sp.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))
    if not getattr(args, "command", None):
        args = p.parse_args(["auto", *(argv or [])])
    if getattr(args, "limit", None) == 0:
        args.limit = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        ui.error("Aborted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
