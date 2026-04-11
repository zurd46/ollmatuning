"""ollmatuning — CLI entry point with a cool rich-based interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from . import ui
from .system import detect_system
from .recommend import discover_models, expand_candidates, shortlist
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


def cmd_recommend(args: argparse.Namespace) -> int:
    ui.show_banner()
    info = detect_system()
    ui.show_system(info)

    ui.step("Suche Code- und Tool-Modelle auf ollama.com ...")
    families = discover_models(verbose=args.verbose)
    ui.show_families(families)

    ui.step("Filtere nach passender VRAM/RAM-Größe ...")
    cands = expand_candidates(families, info, verbose=args.verbose)
    picked = shortlist(cands, limit=args.limit)

    if not picked:
        ui.error("Keine passenden Modelle gefunden. Zu wenig VRAM/RAM?")
        return 1

    ui.show_candidates(picked, title=f"Top {len(picked)} für deine Hardware")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    ui.show_banner()

    if not ollama_is_up():
        ui.error("Ollama-Server läuft nicht auf 127.0.0.1:11434.")
        ui.info("Starte ihn mit:  [bold bright_cyan]ollama serve[/bold bright_cyan]")
        return 2

    info = detect_system()
    ui.show_system(info)

    if args.models:
        models = args.models
        ui.info(f"Benchmarke explizite Modelle: [bold]{', '.join(models)}[/bold]")
    else:
        ui.step("Suche passende Kandidaten ...")
        families = discover_models(verbose=args.verbose)
        ui.show_families(families)
        cands = expand_candidates(families, info, verbose=args.verbose)
        picked = shortlist(cands, limit=args.limit)
        if not picked:
            ui.error("Keine passenden Modelle gefunden.")
            return 1
        ui.show_candidates(picked, title=f"{len(picked)} Kandidaten")
        models = [c.model for c in picked]

    local = set(list_local_models())
    results: list[BenchResult] = []

    with ui.make_progress() as progress:
        task = progress.add_task("Benchmark-Pipeline", total=len(models))
        for m in models:
            progress.update(task, description=f"[bright_cyan]{m}[/bright_cyan] → check")
            already_local = m in local or any(
                x == m or x.startswith(m + ":") for x in local
            )
            if not already_local:
                progress.update(task, description=f"[yellow]{m}[/yellow] → pull")
                if not pull_model(m, verbose=False):
                    ui.error(f"Pull fehlgeschlagen: {m}")
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

    ui.show_results(results)

    ok_results = sorted(
        [r for r in results if r.ok],
        key=lambda r: r.tokens_per_sec,
        reverse=True,
    )
    if not ok_results:
        ui.error("Kein Modell erfolgreich gebencht.")
        return 1

    best = ok_results[0]
    ui.show_winner(best)

    if args.set_best:
        _save_config({"best_model": best.model, "tokens_per_sec": best.tokens_per_sec})
        ui.success(f"Gespeichert in [bold]{CONFIG_PATH}[/bold]")
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


def cmd_auto(args: argparse.Namespace) -> int:
    """All-in-one: banner → detect → recommend → benchmark → set winner."""
    ui.show_banner()

    ui.step("Schritt 1/4: Hardware & Treiber erkennen")
    info = detect_system()
    ui.show_system(info)

    if not ollama_is_up():
        ui.error("Ollama-Server läuft nicht auf 127.0.0.1:11434.")
        ui.info("Starte ihn mit:  [bold bright_cyan]ollama serve[/bold bright_cyan]")
        return 2

    ui.step("Schritt 2/4: Code- und Tool-Modelle auf ollama.com suchen")
    families = discover_models(verbose=args.verbose)
    ui.show_families(families)

    ui.step("Schritt 3/4: Nach VRAM/RAM filtern und Top-Kandidaten wählen")
    cands = expand_candidates(families, info, verbose=args.verbose)
    picked = shortlist(cands, limit=args.limit)
    if not picked:
        ui.error("Keine passenden Modelle gefunden.")
        return 1
    ui.show_candidates(picked, title=f"Top {len(picked)} für deine Hardware")
    models = [c.model for c in picked]

    ui.step("Schritt 4/4: Pullen und tok/s benchmarken")
    local = set(list_local_models())
    results: list[BenchResult] = []
    with ui.make_progress() as progress:
        task = progress.add_task("Benchmark-Pipeline", total=len(models))
        for m in models:
            already_local = m in local or any(
                x == m or x.startswith(m + ":") for x in local
            )
            if not already_local:
                progress.update(task, description=f"[yellow]{m}[/yellow] → pull")
                if not pull_model(m, verbose=False):
                    ui.error(f"Pull fehlgeschlagen: {m}")
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

    ui.show_results(results)
    ok = sorted([r for r in results if r.ok], key=lambda r: r.tokens_per_sec, reverse=True)
    if not ok:
        ui.error("Kein Modell erfolgreich gebencht.")
        return 1

    best = ok[0]
    ui.show_winner(best)

    if not args.no_save:
        _save_config({"best_model": best.model, "tokens_per_sec": best.tokens_per_sec})
        ui.success(f"Gespeichert in [bold]{CONFIG_PATH}[/bold]")
        _print_env_hint(best.model)

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    ui.show_banner()
    if not CONFIG_PATH.exists():
        ui.warn("Noch keine Konfiguration. Führe zuerst [bold]ollmatuning benchmark --set-best[/bold] aus.")
        return 1
    data = json.loads(CONFIG_PATH.read_text())
    from rich.panel import Panel
    from rich.box import ROUNDED
    body = "\n".join(f"[bold yellow]{k}[/bold yellow]: [white]{v}[/white]" for k, v in data.items())
    ui.console.print(Panel(body, title="[bold]Gespeicherte Konfiguration[/bold]",
                           border_style="bright_green", box=ROUNDED))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ollmatuning",
        description="Detect GPU/system, find best Ollama coding+tool LLM, benchmark tokens/sec, set winner.",
    )
    p.add_argument("-V", "--version", action="version", version=f"ollmatuning {__version__}")
    sub = p.add_subparsers(dest="command", required=False)

    sp = sub.add_parser("auto", help="ALLES in einem Befehl (detect → recommend → benchmark → set)")
    sp.add_argument("--limit", type=int, default=3)
    sp.add_argument("--no-save", action="store_true", help="Sieger nicht speichern")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_auto)

    sp = sub.add_parser("detect", help="Hardware und Treiber anzeigen")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("recommend", help="Passende Modelle finden (ohne Download)")
    sp.add_argument("--limit", type=int, default=3)
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_recommend)

    sp = sub.add_parser("benchmark", help="Kandidaten pullen und tok/s messen")
    sp.add_argument("--limit", type=int, default=3)
    sp.add_argument("--models", nargs="+", help="Explizite Modelle (überspringt Auto-Auswahl)")
    sp.add_argument("--set-best", action="store_true", help="Sieger in ~/.ollmatuning/config.json speichern")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("show", help="Gespeicherte Konfiguration anzeigen")
    sp.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    # Default command: `auto` when nothing is given.
    if not getattr(args, "command", None):
        args = p.parse_args(["auto", *(argv or [])])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        ui.error("Abgebrochen.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
