"""ollmatuning — find the best quantized LLM for your hardware."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ollmatuning")
except PackageNotFoundError:
    # Fallback for development installs where package metadata isn't available.
    __version__ = "0.1.0"