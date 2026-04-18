# RemCoder Project Memo

## Overview
ollmatuning is a tool that automatically detects your hardware (GPU, CPU, RAM), finds the best quantized LLM for your system, benchmarks models for performance, and sets the optimal model. It supports both GGUF/Ollama and MLX runtimes.

## Tech Stack
- Python 3.10+
- rich - for terminal UI
- mlx-lm (optional) - for Apple Silicon MLX support
- pytest (dev) - for testing

## Structure
- ollmatuning/ - main package with modules:
  - cli.py - command-line interface and argument parsing
  - system.py - hardware detection logic
  - utils.py - utility functions like HTTP helpers, rate limiting, VRAM budgeting
  - recommend.py - model discovery from ollama.com
  - huggingface.py - GGUF model discovery on HuggingFace
  - mlx_models.py - MLX model discovery on HuggingFace
  - benchmark.py - Ollama benchmark pipeline using HTTP API
  - mlx_benchmark.py - MLX benchmark pipeline with mlx-lm and Metal GPU memory handling
  - ui.py - Rich terminal UI components

## Build / Test / Lint
- `pip install .` - Install core package
- `pip install '.[mlx]'` - Add MLX support for Apple Silicon  
- `pip install '.[dev]'` - Install dev dependencies including pytest
- `python -m pytest tests/ -v` - Run tests
- `python -m ollmatuning detect` - Test hardware detection

## Known TODOs
- None found

## Conventions
- Uses rich for terminal UI components
- Modular architecture with separate modules for different functions (hardware, models, benchmarking)
- Command-line interface with argparse in cli.py