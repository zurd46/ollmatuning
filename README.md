# ollmatuning

**Finde automatisch das beste quantisierte LLM für deine Hardware — mit echten Tokens/Sekunde- und VRAM-Messungen.**

`ollmatuning` erkennt dein System, sucht auf [HuggingFace](https://huggingface.co) nach optimalen quantisierten Modellen, benchmarkt sie und setzt den Sieger. Auf Apple Silicon läuft alles nativ über **MLX** — kein Ollama nötig. Auf Linux/Windows werden **GGUF**-Modelle via Ollama genutzt.

## Highlights

- **OS-aware Runtime** — Apple Silicon → MLX (nativ, schnellste), alles andere → GGUF via Ollama
- **HuggingFace-Suche** — findet quantisierte Modelle (4-bit, 8-bit) direkt auf HuggingFace
- **Smart Dedup** — wählt pro Modellfamilie die beste Quantisierung (bevorzugt 4-bit)
- **Echte VRAM-Messung** — misst tatsächlichen Speicherverbrauch (Metal API / Ollama `/api/ps`)
- **Resumable Downloads** — unterbrochene Downloads setzen automatisch fort
- **Hardware-Erkennung** — NVIDIA, AMD, Intel, Apple Silicon (Windows / Linux / macOS)
- **Treiber-Check** — warnt bei veralteten NVIDIA-Treibern (<525 für CUDA 12)
- **Cooles Rich-UI** — Banner, farbige Tabellen, Progress-Bar, Medaillen-Ranking

## Wie funktioniert die Runtime-Erkennung?

| OS | Hardware | Runtime | Format | Ollama nötig? |
|---|---|---|---|---|
| macOS | Apple Silicon (M1–M5) | `mlx-lm` | MLX (safetensors) | Nein |
| macOS | Intel | Ollama | GGUF | Ja |
| Linux | NVIDIA / AMD | Ollama | GGUF | Ja |
| Windows | NVIDIA | Ollama | GGUF | Ja |

## Voraussetzungen

- Python **3.10+**
- **Apple Silicon (MLX):** `pip install 'ollmatuning[mlx]'`
- **Alle anderen:** [Ollama](https://ollama.com/download) installiert und gestartet (`ollama serve`)

## Installation

```bash
git clone <repo> ollmatuning
cd ollmatuning

# Standard (GGUF via Ollama)
pip install .

# Apple Silicon (MLX — empfohlen auf M1/M2/M3/M4/M5)
pip install '.[mlx]'
```

## Schnellstart

```bash
ollmatuning
```

Das war's. `ollmatuning` erkennt automatisch dein System und wählt die richtige Runtime:

1. **Detect** — Hardware, Treiber, Runtime erkennen
2. **Search** — Quantisierte Modelle auf HuggingFace suchen
3. **Benchmark** — Download + `tokens/sec` + VRAM messen
4. **Set** — Sieger in `~/.ollmatuning/config.json` speichern

### Auf Apple Silicon (MLX)

```bash
pip install '.[mlx]'
ollmatuning          # sucht MLX-Modelle, benchmarkt nativ
```

Kein `ollama serve` nötig — Modelle laufen direkt über Apple's MLX Framework.

### Auf Linux / Windows (GGUF + Ollama)

```bash
pip install .
ollama serve         # Terminal 1
ollmatuning          # Terminal 2
```

## Befehle

| Befehl | Zweck |
|---|---|
| `ollmatuning` / `ollmatuning auto` | Alles in einem Rutsch (Default) |
| `ollmatuning detect` | Hardware, Treiber, empfohlene Runtime |
| `ollmatuning detect --json` | Maschinenlesbare Ausgabe |
| `ollmatuning recommend` | Kandidaten anzeigen, nichts downloaden |
| `ollmatuning benchmark` | Kandidaten downloaden + messen |
| `ollmatuning benchmark --set-best` | Zusätzlich Sieger speichern |
| `ollmatuning benchmark --models <model1> <model2>` | Explizite Modelle testen |
| `ollmatuning show` | Gespeicherten Sieger anzeigen |

### Flags

| Flag | Beschreibung |
|---|---|
| `--limit N` | Anzahl Kandidaten begrenzen (Default: 6, `0` = alle) |
| `--mlx` | MLX erzwingen (nur Apple Silicon) |
| `--gguf` | GGUF via Ollama erzwingen (alle Plattformen) |
| `--ollama` | Zusätzlich ollama.com Library durchsuchen |
| `--no-save` | Sieger nicht speichern (`auto`) |
| `--set-best` | Sieger speichern (`benchmark`) |
| `-v, --verbose` | URLs und Details zeigen |
| `-V, --version` | Version anzeigen |

## Wie die Modell-Suche funktioniert

### HuggingFace (Standard)

1. **API-Suche** — `huggingface.co/api/models?filter=mlx` (Apple Silicon) oder `?filter=gguf` (alle anderen)
2. **Mehrere Queries** — sucht nach "coder", "instruct", "tool calling" Modellen
3. **File-Analyse** — liest `.safetensors` / `.gguf` Dateigrößen für VRAM-Schätzung
4. **Smart Dedup** — pro Basis-Modell nur die beste Quantisierung (4-bit bevorzugt)
5. **VRAM-Filter** — Apple Silicon: 75% Unified Memory Budget, GPU: erkannter VRAM

### Ollama Library (optional mit `--ollama`)

1. Scrapt `ollama.com/library` für alle Modellfamilien
2. Prüft Capability-Badges (`tools`, `code`)
3. Filtert Tags nach VRAM-Budget
4. HEAD-verifiziert jeden Kandidaten

## Wie der Benchmark funktioniert

Für jedes Modell:

1. **Download/Pull** — MLX: `huggingface_hub` (resumable), Ollama: `/api/pull` (resumable)
2. **VRAM messen** — MLX: `mlx.core.metal.get_active_memory()`, Ollama: `/api/ps`
3. **Warm-up** — kurze Generation um Gewichte in den Speicher zu laden
4. **Code-Prompt** — "schreibe `merge_intervals` + 3 asserts"
5. **Tool-Use-Prompt** — "erzeuge JSON-Calls für `get_weather(berlin)` und `get_weather(tokyo)`"
6. **Peak-VRAM** — misst Speicher-Peak nach Inferenz
7. **Metrik** — `tokens/sec` + tatsächlicher VRAM-Verbrauch

### Beispiel-Output

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    Benchmark Results                        ┃
┣━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┫
┃Rank ┃ Model                  ┃ tok/s ┃  VRAM  ┃   Peak    ┃
┣━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━╋━━━━━━━━━━━┫
┃ 🥇  ┃ Qwen2.5-Coder-14B-4bit┃ 58.42 ┃ 8.0 GB ┃  8.4 GB   ┃
┃ 🥈  ┃ Qwen3-Coder-30B-4bit  ┃ 42.50 ┃ 16.3 GB┃ 16.8 GB   ┃
┃ 🥉  ┃ Qwen2.5-Coder-32B-4bit┃ 38.10 ┃ 17.5 GB┃ 18.1 GB   ┃
┗━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━┻━━━━━━━━━━━┛
```

## Unterbrochene Downloads

Downloads sind **immer resumable**:

- **Ctrl+C während Download** — überspringt das aktuelle Modell, macht mit dem nächsten weiter
- **Erneut starten** — setzt unterbrochene Downloads automatisch dort fort wo aufgehört wurde
- **HuggingFace** — `.incomplete`-Dateien in `~/.cache/huggingface/`
- **Ollama** — native Resume-Unterstützung in `/api/pull`

## Dateistruktur

```
ollmatuning/
├── pyproject.toml
├── README.md
└── ollmatuning/
    ├── __init__.py
    ├── cli.py              # argparse, Subcommands, OS-aware Dispatch
    ├── system.py           # Hardware/Treiber-Erkennung, Apple Silicon Detection
    ├── recommend.py        # ollama.com Scrape + VRAM-Filter (optional)
    ├── huggingface.py      # HuggingFace API: GGUF-Modell-Suche
    ├── mlx_models.py       # HuggingFace API: MLX-Modell-Suche + Dedup
    ├── benchmark.py        # Ollama HTTP API Benchmark + VRAM-Messung
    ├── mlx_benchmark.py    # MLX Benchmark + Metal Memory + Resume
    └── ui.py               # Rich Banner/Tables/Progress
```

## Konfiguration

`ollmatuning auto` und `benchmark --set-best` speichern in:

```
~/.ollmatuning/config.json
```

Beispiel (Apple Silicon / MLX):

```json
{
  "best_model": "lmstudio-community/Qwen2.5-Coder-14B-Instruct-MLX-4bit",
  "tokens_per_sec": 58.42,
  "runtime": "mlx",
  "vram_mb": 8192,
  "peak_vram_mb": 8600
}
```

Beispiel (GGUF / Ollama):

```json
{
  "best_model": "hf.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M",
  "tokens_per_sec": 45.20,
  "runtime": "ollama",
  "vram_mb": 9500,
  "peak_vram_mb": 9800
}
```

## Troubleshooting

| Problem | Lösung |
|---|---|
| `mlx-lm is not installed` | `pip install 'ollmatuning[mlx]'` |
| `Ollama server is not running` | `ollama serve` in einem anderen Terminal starten |
| `nvidia-smi` nicht gefunden | NVIDIA-Treiber installieren; Fallback auf WMI/lspci |
| OOM beim Benchmark | `--limit 3` oder explizit `--models <kleines-modell>` |
| Download zu langsam | Ctrl+C zum Skippen, erneut starten zum Fortsetzen |
| Kaputte Box-Zeichen (Windows) | Windows Terminal + UTF-8 (`chcp 65001`) |

## Lizenz

MIT
