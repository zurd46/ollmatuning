"""ollmatuning — CLI entry point with a cool rich-based interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import __version__
from . import ui
from .system import detect_system
from .recommend import (
    discover_models, expand_candidates, shortlist,
    verify_candidates, verify_model_exists, Candidate,
)
from .benchmark import (
    ollama_is_up, list_local_models, pull_model, benchmark_model, BenchResult,
)

CONFIG_PATH = Path.home() / ".ollmatuning" / "config.json"


def cmd_detect(args: argparse.Namespace) -> int:
    info = detect_system()
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
        return 0
    ui.show_banner()
    ui.show_system(info)
    return 0


def _pick_verified(
    cands: list[Candidate],
    limit: int | None,
    verbose: bool = False,
) -> list[Candidate]:
    """Shortlist (optionally capped), HEAD-verify every candidate, return those that exist.

    When `limit` is None we verify the ENTIRE eligible set — no capping, no shortcuts.
    """
    pool = shortlist(cands, limit=limit)
    ui.info(f"Verifying {len(pool)} candidates against ollama.com ...")
    verified = verify_candidates(pool, verbose=verbose)
    return verified


def cmd_recommend(args: argparse.Namespace) -> int:
    ui.show_banner()
    info = detect_system()
    ui.show_system(info)

    ui.step("Searching code and tool models on ollama.com ...")
    families, code_set = discover_models(verbose=args.verbose)
    ui.show_families(families)

    ui.step("Filtering by VRAM/RAM budget (tool-capable only) ...")
    cands = expand_candidates(families, info, code_set=code_set, verbose=args.verbose)
    if not cands:
        ui.error("No models with published tags found.")
        return 1

    ui.step("Safety check: every candidate must exist on ollama.com ...")
    picked = _pick_verified(cands, limit=args.limit, verbose=args.verbose)

    if not picked:
        ui.error("No verified models found. Not enough VRAM/RAM or network issues?")
        return 1

    ui.show_candidates(picked, title=f"Top {len(picked)} verified for your hardware")
    return 0


def _run_benchmark_pipeline(models: list[str]) -> list[BenchResult]:
    local = set(list_local_models())
    results: list[BenchResult] = []

    with ui.make_progress() as progress:
        task = progress.add_task("Benchmark pipeline", total=len(models))
        for m in models:
            progress.update(task, description=f"[bright_cyan]{m}[/bright_cyan] → check")
            already_local = m in local or any(
                x == m or x.startswith(m + ":") for x in local
            )
            if not already_local:
                progress.update(task, description=f"[yellow]{m}[/yellow] → pull")
                if not pull_model(m, verbose=False):
                    ui.error(f"Pull failed: {m}")
                    results.append(BenchResult(m, 0, 0, 0, 0, False, "pull failed"))
                    progress.advance(task)
                    continue
            progress.update(task, description=f"[bright_magenta]{m}[/bright_magenta] → benchmark")
            r = benchmark_model(m)
            if r.ok:
                ui.success(f"{m}: [bold bright_green]{r.tokens_per_sec:.2f} tok/s[/bold bright_green]")
            else:
                ui.error(f"{m}: {r.error[:80]}")
            results.append(r)
            progress.advance(task)
    return results


def cmd_benchmark(args: argparse.Namespace) -> int:
    ui.show_banner()

    if not ollama_is_up():
        ui.error("Ollama server is not running on 127.0.0.1:11434.")
        ui.info("Start it with:  [bold bright_cyan]ollama serve[/bold bright_cyan]")
        return 2

    info = detect_system()
    ui.show_system(info)

    if args.models:
        ui.info(f"Verifying explicit models: [bold]{', '.join(args.models)}[/bold]")
        models: list[str] = []
        for m in args.models:
            if verify_model_exists(m):
                ui.success(m)
                models.append(m)
            else:
                ui.error(f"{m} not on ollama.com — skipped")
        if not models:
            ui.error("No valid models provided.")
            return 1
    else:
        ui.step("Searching matching candidates ...")
        families, code_set = discover_models(verbose=args.verbose)
        ui.show_families(families)
        cands = expand_candidates(families, info, code_set=code_set, verbose=args.verbose)
        if not cands:
            ui.error("No models with published tags found.")
            return 1
        ui.step("Safety check against ollama.com ...")
        picked = _pick_verified(cands, limit=args.limit, verbose=args.verbose)
        if not picked:
            ui.error("No verified models found.")
            return 1
        ui.show_candidates(picked, title=f"{len(picked)} verified candidates")
        models = [c.model for c in picked]

    results = _run_benchmark_pipeline(models)
    ui.show_results(results)

    ok_results = sorted(
        [r for r in results if r.ok],
        key=lambda r: r.tokens_per_sec,
        reverse=True,
    )
    if not ok_results:
        ui.error("No model benchmarked successfully.")
        return 1

    best = ok_results[0]
    ui.show_winner(best)

    if args.set_best:
        _save_config({"best_model": best.model, "tokens_per_sec": best.tokens_per_sec})
        ui.success(f"Saved to [bold]{CONFIG_PATH}[/bold]")
        _print_env_hint(best.model)

    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """All-in-one: banner → detect → recommend → verify → benchmark → set winner."""
    ui.show_banner()

    ui.step("Step 1/4: detecting hardware and drivers")
    info = detect_system()
    ui.show_system(info)

    if not ollama_is_up():
        ui.error("Ollama server is not running on 127.0.0.1:11434.")
        ui.info("Start it with:  [bold bright_cyan]ollama serve[/bold bright_cyan]")
        return 2

    ui.step("Step 2/4: searching tool-capable models on ollama.com")
    families, code_set = discover_models(verbose=args.verbose)
    ui.show_families(families)

    ui.step("Step 3/4: VRAM/RAM filter + tools-capability safety check")
    cands = expand_candidates(families, info, code_set=code_set, verbose=args.verbose)
    if not cands:
        ui.error("No models with published tags found.")
        return 1
    picked = _pick_verified(cands, limit=args.limit, verbose=args.verbose)
    if not picked:
        ui.error("No verified models found.")
        return 1
    ui.show_candidates(picked, title=f"Top {len(picked)} verified for your hardware")
    models = [c.model for c in picked]

    ui.step("Step 4/4: pulling and measuring tok/s")
    results = _run_benchmark_pipeline(models)
    ui.show_results(results)

    ok = sorted([r for r in results if r.ok], key=lambda r: r.tokens_per_sec, reverse=True)
    if not ok:
        ui.error("No model benchmarked successfully.")
        return 1

    best = ok[0]
    ui.show_winner(best)

    if not args.no_save:
        _save_config({"best_model": best.model, "tokens_per_sec": best.tokens_per_sec})
        ui.success(f"Saved to [bold]{CONFIG_PATH}[/bold]")
        _print_env_hint(best.model)

    return 0


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


def _print_env_hint(model: str) -> None:
    if os.name == "nt":
        ui.info(f'PowerShell:  [bright_cyan]$env:OLLAMA_MODEL = "{model}"[/bright_cyan]')
        ui.info(f'cmd.exe:     [bright_cyan]set OLLAMA_MODEL={model}[/bright_cyan]')
    else:
        ui.info(f'bash/zsh:    [bright_cyan]export OLLAMA_MODEL="{model}"[/bright_cyan]')


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ollmatuning",
        description="Detect GPU/system, find best Ollama coding+tool LLM, benchmark tokens/sec, set winner.",
    )
    p.add_argument("-V", "--version", action="version", version=f"ollmatuning {__version__}")
    sub = p.add_subparsers(dest="command", required=False)

    sp = sub.add_parser("auto", help="All-in-one (detect -> recommend -> benchmark -> set)")
    sp.add_argument("--limit", type=int, default=6,
                    help="Cap number of candidates to benchmark (default: 6, use 0 for all)")
    sp.add_argument("--no-save", action="store_true", help="Do not save the winner")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_auto)

    sp = sub.add_parser("detect", help="Show hardware and drivers")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("recommend", help="Find matching models (no download)")
    sp.add_argument("--limit", type=int, default=6,
                    help="Cap number of candidates to benchmark (default: 6, use 0 for all)")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_recommend)

    sp = sub.add_parser("benchmark", help="Pull candidates and measure tok/s")
    sp.add_argument("--limit", type=int, default=6,
                    help="Cap number of candidates to benchmark (default: 6, use 0 for all)")
    sp.add_argument("--models", nargs="+", help="Explicit models (skips auto selection)")
    sp.add_argument("--set-best", action="store_true", help="Save winner to ~/.ollmatuning/config.json")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("show", help="Show saved configuration")
    sp.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    if not getattr(args, "command", None):
        args = p.parse_args(["auto", *(argv or [])])
    # Treat --limit 0 as "no cap".
    if getattr(args, "limit", None) == 0:
        args.limit = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        ui.error("Aborted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
