"""Top-level CLI for hakim-vision."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress

from hakim_vision import __version__
from hakim_vision.config import GenerationConfig
from hakim_vision.synthetic.assets import Backgrounds, Cards
from hakim_vision.synthetic.pack import pack_backgrounds, pack_cards
from hakim_vision.synthetic.scene import render_random_scene, write_yolo_label

app = typer.Typer(
    name="hakim-vision",
    help="Synthetic playing-card dataset generation and detector tooling.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(f"hakim-vision {__version__}")


@app.command("config-show")
def config_show() -> None:
    """Print the resolved generation config (env + defaults)."""
    cfg = GenerationConfig()
    console.print_json(data=cfg.model_dump(mode="json"))


@app.command()
def generate(
    backgrounds_dir: Path = typer.Option(
        ..., "--backgrounds", help="Directory of backgrounds-*.tar shards."
    ),
    cards_dir: Path = typer.Option(..., "--cards", help="Directory of cards-*.tar shards."),
    output_dir: Path = typer.Option(
        Path("data/scenes"), "--output", "-o", help="Where to write scenes + labels."
    ),
    count: int = typer.Option(10, "--count", "-n", min=1, help="Number of scenes."),
    n_cards: int = typer.Option(2, "--n-cards", min=2, max=3),
    canvas_size: int = typer.Option(720, "--canvas-size", min=64),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Render `count` synthetic scenes and write YOLO `.txt` labels alongside."""
    cfg = GenerationConfig()
    effective_seed = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(effective_seed)

    bg_shards = sorted(backgrounds_dir.glob("*.tar"))
    card_shards = sorted(cards_dir.glob("*.tar"))
    if not bg_shards:
        raise typer.BadParameter(f"no shards found in {backgrounds_dir}")
    if not card_shards:
        raise typer.BadParameter(f"no shards found in {cards_dir}")

    backgrounds = Backgrounds(bg_shards, rng=np.random.default_rng(effective_seed + 1))
    cards = Cards(card_shards, rng=np.random.default_rng(effective_seed + 2))
    class_to_id = {name: i for i, name in enumerate(cards.class_names)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text("\n".join(cards.class_names) + "\n", encoding="utf-8")

    with Progress() as progress:
        task = progress.add_task("rendering", total=count)
        for i in range(count):
            scene = render_random_scene(
                cards,
                backgrounds,
                rng=rng,
                n_cards=n_cards,
                canvas_size=canvas_size,
            )
            stem = f"scene_{i:06d}"
            cv2.imwrite(str(output_dir / f"{stem}.png"), scene.image)
            (output_dir / f"{stem}.txt").write_text(
                write_yolo_label(scene, class_to_id), encoding="utf-8"
            )
            progress.advance(task)

    console.print(f"Wrote {count} scenes to {output_dir}")


@app.command("pack-backgrounds")
def cmd_pack_backgrounds(
    images_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Argument(...),
    shard_size: int = typer.Option(1024, "--shard-size", min=1),
    prefix: str = typer.Option("backgrounds", "--prefix"),
) -> None:
    """Pack background textures into tar shards loadable by `Backgrounds`."""
    shards = pack_backgrounds(images_dir, output_dir, shard_size=shard_size, shard_prefix=prefix)
    console.print(f"Wrote {len(shards)} shard(s) to {output_dir}")


@app.command("pack-cards")
def cmd_pack_cards(
    cards_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Argument(...),
    shard_size: int = typer.Option(512, "--shard-size", min=1),
    prefix: str = typer.Option("cards", "--prefix"),
    no_hulls: bool = typer.Option(
        False,
        "--no-hulls",
        help="Skip live corner-hull extraction (use reference rectangles instead).",
    ),
) -> None:
    """Pack extracted card images into tar shards loadable by `Cards`."""
    shards = pack_cards(
        cards_root,
        output_dir,
        shard_size=shard_size,
        shard_prefix=prefix,
        extract_hulls=not no_hulls,
    )
    console.print(f"Wrote {len(shards)} shard(s) to {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
