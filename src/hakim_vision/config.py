"""Centralized configuration for the synthetic-data pipeline.

All magic numbers and paths from the legacy notebook live here, typed and
overridable from env vars (HAKIM_VISION_*) or .env files via pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationConfig(BaseSettings):
    """Settings for synthetic Baloot/playing-card scene generation."""

    model_config = SettingsConfigDict(
        env_prefix="HAKIM_VISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Root data directory.")
    cards_dir: Path = Field(
        default=Path("data/shards/cards"),
        description="Directory of cards-*.tar shards.",
    )
    backgrounds_dir: Path = Field(
        default=Path("data/shards/backgrounds"),
        description="Directory of backgrounds-*.tar shards.",
    )
    output_dir: Path = Field(default=Path("data/scenes"), description="Generated scenes.")

    # Focus / quality filtering (was hardcoded `min_focus=120` in the notebook).
    min_focus: float = Field(
        default=120.0,
        ge=0.0,
        description="Minimum Laplacian variance for a frame to be considered in-focus.",
    )

    # Augmentation aspect-ratio guard (was hardcoded `keep_ratio=5`).
    max_aspect_ratio: float = Field(
        default=5.0,
        gt=1.0,
        description="Reject augmented boxes whose aspect ratio exceeds this value.",
    )

    # Scene composition. The renderer only supports 2- or 3-card scenes today;
    # widen these bounds at the same time as `render_random_scene`.
    cards_per_scene_min: int = Field(default=2, ge=2, le=3)
    cards_per_scene_max: int = Field(default=3, ge=2, le=3)

    # Reproducibility.
    seed: int = Field(default=42, description="Global RNG seed.")


__all__ = ["GenerationConfig"]
