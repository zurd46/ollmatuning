# ollmatuning

Find the best quantized LLM for your hardware. Auto-detects GPU, searches models, benchmarks tokens/second, and sets the winner.

## Features

- **Hardware detection** — NVIDIA, AMD, Intel, Apple Silicon with driver health checks
- **Smart model discovery** — Live search on HuggingFace (GGUF + MLX) and ollama.com
- **Dual runtime support** — MLX (Apple Silicon native) or GGUF (Ollama, any platform)
- **Memory-aware filtering** — Only suggests models that fit your VRAM/RAM budget
- **Real benchmarking** — Measures actual tokens/second with coding, tool-use, and reasoning prompts
- **Rich CLI output** — Colorful tables, progress bars, and results

## Installation

```bash
pip install .           # Core (Ollama/GGUF)
pip install '.[mlx]'    # + MLX support for Apple Silicon
pip install '.[dev]'    # + test dependencies
```

## Quick Start

```bash
# All-in-one: detect hardware, find models, benchmark, and set the best
ollmatuning auto

# Just detect hardware
ollmatuning detect

# Find matching models (no download)
ollmatuning recommend

# Benchmark specific models
ollmatuning benchmark --models llama3.2:3b qwen2.5-coder:7b

# JSON output for scripting
ollmatuning benchmark --json
ollmatuning detect --json
```

## Flags

| Flag | Description |
|------|-------------|
| `--mlx` | Force MLX runtime (Apple Silicon) |
| `--gguf` | Force GGUF runtime via Ollama |
| `--ollama` | Also search ollama.com library |
| `--download` | Download missing models during benchmark |
| `--limit N` | Cap number of candidates (default: 6) |
| `--set-best` | Save winner to ~/.ollmatuning/config.json |
| `--json` | Output results as JSON (scripting) |
| `-v` | Verbose logging |
| `--no-save` | Don't save the winner (auto mode) |

## Architecture

```
cli.py          → argparse, command dispatch
system.py       → GPU/CPU/RAM detection (macOS, Linux, Windows)
utils.py        → HTTP helpers, rate limiter, VRAM budget, heuristics
recommend.py    → ollama.com model discovery + Candidate dataclass
huggingface.py  → GGUF model discovery on HuggingFace
mlx_models.py   → MLX model discovery on HuggingFace
benchmark.py    → Ollama benchmark pipeline (HTTP API)
mlx_benchmark.py → MLX benchmark pipeline (mlx-lm, Metal GPU memory)
ui.py           → Rich terminal UI
```

## Config

After running with `--set-best` or `auto`, config is saved to `~/.ollmatuning/config.json`:

```json
{
  "best_model": "hf.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M",
  "tokens_per_sec": 45.2,
  "runtime": "ollama",
  "vram_mb": 4500,
  "peak_vram_mb": 4800
}
```

## Development

```bash
pip install -e '.[dev]'
python -m pytest tests/ -v
python -m ollmatuning detect
```

## License

MIT — see LICENSE file.
