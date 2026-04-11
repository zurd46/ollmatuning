"""Rich-based UI helpers — banner, tables, panels, progress."""
from __future__ import annotations

import sys

# Force UTF-8 on legacy Windows consoles so box-drawing and emoji render.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.box import ROUNDED, HEAVY
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)

from .system import SystemInfo
from .recommend import Candidate
from .benchmark import BenchResult

console = Console()

BANNER = r"""
  ___  _     _     __  __    _  _____ _   _ _   _ ___ _   _  ____
 / _ \| |   | |   |  \/  |  / \|_   _| | | | \ | |_ _| \ | |/ ___|
| | | | |   | |   | |\/| | / _ \ | | | | | |  \| || ||  \| | |  _
| |_| | |___| |___| |  | |/ ___ \| | | |_| | |\  || || |\  | |_| |
 \___/|_____|_____|_|  |_/_/   \_\_|  \___/|_| \_|___|_| \_|\____|
"""

TAGLINE = "GPU → Treiber → bestes LLM → tokens/sec — finde & setze den Sieger"


def show_banner() -> None:
    text = Text(BANNER, style="bold cyan")
    sub = Text(TAGLINE, style="bold magenta")
    panel = Panel(
        Align.center(Text.assemble(text, "\n", sub)),
        box=HEAVY,
        border_style="bright_blue",
        padding=(0, 2),
    )
    console.print(panel)


def show_system(info: SystemInfo) -> None:
    table = Table(
        title="[bold]System & Hardware[/bold]",
        box=ROUNDED,
        border_style="cyan",
        header_style="bold bright_white on blue",
        title_style="bold bright_cyan",
        expand=True,
    )
    table.add_column("Komponente", style="bold yellow", no_wrap=True)
    table.add_column("Wert", style="white")

    table.add_row("OS", f"{info.os} {info.arch}  [dim]({info.os_version[:40]})[/dim]")
    table.add_row("CPU", f"{info.cpu}  [dim]({info.cpu_cores} cores)[/dim]")
    ram_gb = info.ram_mb / 1024
    table.add_row("RAM", f"{ram_gb:.1f} GB")

    if not info.gpus:
        table.add_row("GPU", "[red]keine erkannt[/red]")
    for i, g in enumerate(info.gpus):
        vram_gb = g.vram_mb / 1024
        vendor_color = {
            "nvidia": "bright_green",
            "amd": "bright_red",
            "intel": "bright_blue",
            "apple": "bright_magenta",
        }.get(g.vendor, "white")
        status_icon = "[green]●[/green]" if g.driver_ok else "[yellow]●[/yellow]"
        table.add_row(
            f"GPU{i}",
            f"{status_icon} [{vendor_color}]{g.name}[/{vendor_color}]  "
            f"[bold]{vram_gb:.1f} GB VRAM[/bold]\n"
            f"   [dim]{g.driver_note}[/dim]",
        )

    console.print(table)
    console.print()


def show_candidates(candidates: list[Candidate], title: str = "Kandidaten") -> None:
    table = Table(
        title=f"[bold]{title}[/bold]",
        box=ROUNDED,
        border_style="magenta",
        header_style="bold bright_white on magenta",
        title_style="bold bright_magenta",
        expand=True,
    )
    table.add_column("#", style="bold yellow", justify="right", width=3)
    table.add_column("Modell", style="bold bright_cyan")
    table.add_column("Params", style="white", justify="right")
    table.add_column("~VRAM", style="green", justify="right")
    table.add_column("Kategorien", style="bright_magenta")

    for i, c in enumerate(candidates, 1):
        cats = " ".join(f"[black on bright_blue] {x} [/black on bright_blue]" for x in c.categories)
        table.add_row(
            str(i),
            c.model,
            f"{c.size_b:g}B",
            f"{c.est_vram_mb / 1024:.1f} GB",
            cats,
        )
    console.print(table)
    console.print()


def show_families(families: list[str], n_shown: int = 12) -> None:
    head = ", ".join(families[:n_shown])
    more = f" [dim]+{len(families) - n_shown} weitere[/dim]" if len(families) > n_shown else ""
    console.print(
        Panel(
            f"[bright_white]{head}[/bright_white]{more}",
            title=f"[bold]{len(families)} Modell-Familien von ollama.com[/bold]",
            border_style="cyan",
            box=ROUNDED,
        )
    )
    console.print()


def show_results(results: list[BenchResult]) -> None:
    ok = [r for r in results if r.ok]
    ok.sort(key=lambda r: r.tokens_per_sec, reverse=True)
    ranked = ok + [r for r in results if not r.ok]

    best_tps = ok[0].tokens_per_sec if ok else 0.0

    table = Table(
        title="[bold]Benchmark-Ergebnisse[/bold]",
        box=HEAVY,
        border_style="bright_green",
        header_style="bold black on bright_green",
        title_style="bold bright_green",
        expand=True,
    )
    table.add_column("Rang", style="bold yellow", justify="right", width=5)
    table.add_column("Modell", style="bold bright_cyan")
    table.add_column("tok/s", style="bold bright_white", justify="right")
    table.add_column("Tokens", style="white", justify="right")
    table.add_column("Eval-Zeit", style="white", justify="right")
    table.add_column("Balken", ratio=1)

    for i, r in enumerate(ranked, 1):
        if not r.ok:
            table.add_row(
                f"{i}",
                r.model,
                "[red]-[/red]",
                "-",
                "-",
                f"[red]✗ {r.error[:50]}[/red]",
            )
            continue
        bar_len = int((r.tokens_per_sec / best_tps) * 30) if best_tps > 0 else 0
        bar = "█" * bar_len
        style = "bright_green" if i == 1 else "green"
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f" {i}")
        table.add_row(
            f"{medal}",
            r.model,
            f"[{style}]{r.tokens_per_sec:.2f}[/{style}]",
            f"{r.eval_count}",
            f"{r.eval_seconds:.2f}s",
            f"[{style}]{bar}[/{style}]",
        )

    console.print(table)
    console.print()


def show_winner(result: BenchResult) -> None:
    body = Text.assemble(
        ("🏆  ", "bold yellow"),
        (result.model, "bold bright_cyan"),
        ("\n\n", ""),
        ("Leistung:  ", "bold white"),
        (f"{result.tokens_per_sec:.2f} tok/s", "bold bright_green"),
        ("\n", ""),
        ("Tokens:    ", "bold white"),
        (f"{result.eval_count}", "white"),
        ("\n", ""),
        ("Eval-Zeit: ", "bold white"),
        (f"{result.eval_seconds:.2f}s", "white"),
    )
    panel = Panel(
        Align.center(body),
        title="[bold bright_yellow]★ SIEGER ★[/bold bright_yellow]",
        border_style="bright_yellow",
        box=HEAVY,
        padding=(1, 4),
    )
    console.print(panel)


def info(msg: str) -> None:
    console.print(f"[bright_blue]ℹ[/bright_blue]  {msg}")


def success(msg: str) -> None:
    console.print(f"[bright_green]✓[/bright_green]  {msg}")


def warn(msg: str) -> None:
    console.print(f"[bright_yellow]⚠[/bright_yellow]  {msg}")


def error(msg: str) -> None:
    console.print(f"[bright_red]✗[/bright_red]  {msg}")


def step(msg: str) -> None:
    console.print(f"[bold magenta]»[/bold magenta] [bold white]{msg}[/bold white]")


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="bright_cyan"),
        TextColumn("[bold bright_white]{task.description}"),
        BarColumn(bar_width=30, complete_style="bright_green", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
