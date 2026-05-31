"""hakim-vision: synthetic playing-card dataset generation + detector tooling."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hakim-vision")
except PackageNotFoundError:  # pragma: no cover - editable installs without metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
