# ollmatuning

**Finde automatisch das beste Ollama-LLM für Coding und Tool-Use auf deiner Hardware — mit echten Tokens/Sekunde-Messungen.**

`ollmatuning` erkennt deine GPU und Treiber, recherchiert live auf [ollama.com](https://ollama.com), filtert Modelle nach verfügbarem VRAM, pullt die besten Kandidaten, benchmarkt sie mit repräsentativen Code- und Tool-Use-Prompts, und speichert den Sieger.

## Features

- **Hardware-Erkennung** — NVIDIA (`nvidia-smi`), AMD, Intel, Apple Silicon (Windows / Linux / macOS)
- **Treiber-Check** — warnt bei veralteten NVIDIA-Treibern (<525 für CUDA 12)
- **Live-Recherche** — scraped `ollama.com/search?c=code` und `?c=tools` für aktuelle Modell-Familien
- **VRAM-Fit** — schätzt Q4-Speicherbedarf (~0.6 GB/B Params + 1 GB Overhead) und filtert Modelle, die nicht passen
- **Echte Benchmarks** — misst `tokens/sec` aus der Ollama HTTP-API (`eval_count / eval_duration`)
- **Code + Tool-Use Prompts** — repräsentativ: Merge-Intervals-Funktion + JSON-Tool-Call-Prompt
- **Ein-Befehl-Workflow** — `ollmatuning auto` macht alles, oder jeder Schritt einzeln
- **Cooles Rich-UI** — Banner, farbige Tabellen, Progress-Bar, Medaillen-Ranking

## Voraussetzungen

- Python **3.10+**
- [Ollama](https://ollama.com/download) installiert und als Server erreichbar (`ollama serve` auf `127.0.0.1:11434`)
- `rich` (wird mit `pip install .` installiert)
- Optional: NVIDIA-Treiber + CUDA für GPU-Beschleunigung

## Installation

```bash
git clone <repo> ollmatuning
cd ollmatuning
pip install .
```

Oder direkt aus dem Quellverzeichnis ohne Installation:

```bash
python -m ollmatuning.cli <befehl>
```

## Schnellstart — ein Befehl macht alles

```bash
ollama serve           # Terminal 1
ollmatuning            # Terminal 2  (= ollmatuning auto)
```

Das war's. `ollmatuning` läuft standardmäßig im `auto`-Modus und führt alle vier Schritte durch:

1. **Detect** — Hardware und Treiber
2. **Recommend** — Live-Recherche auf ollama.com und VRAM-Filter
3. **Benchmark** — Pull + `tokens/sec` messen
4. **Set** — Sieger in `~/.ollmatuning/config.json` speichern und `OLLAMA_MODEL`-Hinweis ausgeben

## Einzelne Befehle

Jeder Schritt ist auch einzeln aufrufbar:

| Befehl | Zweck |
|---|---|
| `ollmatuning auto` | Alles in einem Rutsch (Default) |
| `ollmatuning detect` | Nur Hardware + Treiber anzeigen |
| `ollmatuning detect --json` | Maschinenlesbare Ausgabe |
| `ollmatuning recommend` | Kandidaten anzeigen, nichts pullen |
| `ollmatuning benchmark` | Kandidaten pullen + messen |
| `ollmatuning benchmark --set-best` | Zusätzlich Sieger speichern |
| `ollmatuning benchmark --models qwen2.5-coder:14b llama3.1:8b` | Explizite Modelle testen |
| `ollmatuning show` | Gespeicherten Sieger anzeigen |

### Nützliche Flags

- `--limit N` — wie viele Kandidaten getestet werden (Default: 3)
- `--no-save` — (`auto`) Sieger nicht persistieren
- `-v, --verbose` — URLs und Details während der Recherche zeigen
- `-V, --version` — Version anzeigen

## Wie die Recherche funktioniert

1. **Kategorie-Seiten scrapen:** `https://ollama.com/search?c=code` und `?c=tools`
2. **Familien extrahieren:** Alle `/library/<slug>` Links dedupen
3. **Tag-Seiten laden:** `https://ollama.com/library/<slug>` — Parameter-Größen `7b`, `14b`, `32b` etc. extrahieren
4. **VRAM filtern:** `est_vram_mb = 0.6 * size_b * 1024 + 1024` → muss ≤ GPU-VRAM sein (Fallback: 60 % System-RAM bei reinen CPU-Systemen)
5. **Diversifizieren:** Pro Familie nur ein Tag (größter, der passt), damit nicht 5 Größen desselben Modells gebencht werden

Wenn das Scraping fehlschlägt (kein Internet, Struktur geändert), wird eine kuratierte Fallback-Liste verwendet: `qwen2.5-coder`, `deepseek-coder-v2`, `qwen2.5`, `llama3.1`, `llama3.2`, `mistral-nemo`, `codellama`, `granite-code`, `codegemma`.

## Wie der Benchmark funktioniert

Für jedes Modell:

1. **Warm-up** — `num_predict=8`, um Gewichte in den RAM/VRAM zu laden
2. **Code-Prompt** — "schreibe `merge_intervals` + 3 asserts"
3. **Tool-Use-Prompt** — "erzeuge JSON-Calls für `get_weather(berlin)` und `get_weather(tokyo)`"
4. **Metrik** — Summe `eval_count / eval_duration` aus der Ollama HTTP-API → **tokens/sec**

Das Ranking ist deterministisch: höchste tok/s gewinnt. Der Sieger wird mit einem 🥇-Medaille-Panel hervorgehoben.

## Dateistruktur

```
ollmatuning/
├── pyproject.toml
├── README.md
└── ollmatuning/
    ├── __init__.py
    ├── cli.py          # argparse + Subcommands
    ├── system.py       # Hardware/Treiber-Erkennung
    ├── recommend.py    # ollama.com Scrape + VRAM-Filter
    ├── benchmark.py    # Ollama HTTP API + tok/s
    └── ui.py           # Rich Banner/Tables/Progress
```

## Konfiguration

`ollmatuning auto` und `benchmark --set-best` speichern in:

```
~/.ollmatuning/config.json
```

Beispiel:

```json
{
  "best_model": "qwen2.5-coder:14b",
  "tokens_per_sec": 58.42
}
```

Zum Setzen für andere Tools:

```bash
# bash/zsh
export OLLAMA_MODEL="qwen2.5-coder:14b"
```

```powershell
# PowerShell
$env:OLLAMA_MODEL = "qwen2.5-coder:14b"
```

```cmd
:: cmd.exe
set OLLAMA_MODEL=qwen2.5-coder:14b
```

## Troubleshooting

- **`FEHLER: Ollama-Server läuft nicht`** — starte `ollama serve` in einem anderen Terminal
- **`nvidia-smi` nicht gefunden** — installiere den aktuellen NVIDIA-Treiber; ohne ihn fällt `ollmatuning` auf WMI/CIM (Windows), `lspci` (Linux) zurück
- **OOM beim Benchmark** — mit `--limit 5` werden auch kleinere Modelle getestet; oder explizit per `--models qwen2.5-coder:7b`
- **Scrape liefert nichts** — Fallback-Liste springt automatisch ein; prüfe Internet/Proxy
- **Windows: kaputte Box-Zeichen** — Terminal ggf. auf Windows Terminal / UTF-8 umstellen (`chcp 65001`)

## Lizenz

MIT — siehe Header in den Quelldateien.
