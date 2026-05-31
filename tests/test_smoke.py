"""Smoke tests: package imports, CLI runs, basic geometry round-trips."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from hakim_vision import __version__
from hakim_vision.cli import app
from hakim_vision.config import GenerationConfig
from hakim_vision.geometry import YoloBox, voc_to_yolo

runner = CliRunner()


def test_package_version_is_set() -> None:
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_config_loads_defaults() -> None:
    cfg = GenerationConfig()
    assert cfg.min_focus == pytest.approx(120.0)
    assert cfg.max_aspect_ratio == pytest.approx(5.0)
    assert cfg.cards_per_scene_min <= cfg.cards_per_scene_max


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "hakim-vision" in result.stdout


def test_cli_generate_requires_shard_dirs() -> None:
    # `generate` now needs --backgrounds and --cards; invoking without them
    # must fail with a non-zero exit code rather than silently no-op.
    result = runner.invoke(app, ["generate", "--count", "3"])
    assert result.exit_code != 0


def test_voc_to_yolo_centered_square() -> None:
    box = voc_to_yolo((100, 100), (25.0, 75.0, 25.0, 75.0))
    assert box == YoloBox(cx=0.5, cy=0.5, w=0.5, h=0.5)


def test_voc_to_yolo_rejects_degenerate_box() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        voc_to_yolo((100, 100), (50.0, 50.0, 10.0, 20.0))


def test_voc_to_yolo_rejects_bad_image_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        voc_to_yolo((0, 100), (1.0, 2.0, 3.0, 4.0))
