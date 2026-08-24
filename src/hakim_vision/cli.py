"""Top-level CLI for hakim-vision."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from hakim_vision import __version__
from hakim_vision.config import GenerationConfig
from hakim_vision.datasets import (
    discover_layout,
    inspect_dataset,
    preview_dataset,
    prune_leaked_images,
    remap_dataset,
    scan_dataset_duplicates,
    suggest_mapping,
)
from hakim_vision.models.train import SUPPORTED_MODELS, TrainingConfig, train_detector
from hakim_vision.models.yolo_export import export_yolo_to_onnx
from hakim_vision.server import create_server, default_web_dir
from hakim_vision.synthetic.assets import Backgrounds, Cards
from hakim_vision.synthetic.pack import pack_backgrounds, pack_cards
from hakim_vision.synthetic.scene import render_random_scene, write_yolo_label

# The CLI prints Arabic and emoji. A legacy Windows console defaults to cp1252
# and raises UnicodeEncodeError mid-render, so switch the streams to UTF-8 when
# they are not already; a console that refuses simply keeps its encoding.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower().replace("-", "") != "utf8":
        with contextlib.suppress(Exception):
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

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


dataset_app = typer.Typer(
    name="dataset",
    help="Inspect and prepare third-party card datasets for Baloot training.",
    no_args_is_help=True,
)
app.add_typer(dataset_app)


@dataset_app.command("inspect")
def cmd_dataset_inspect(
    source: Path = typer.Option(..., "--source", "-s", help="Dataset root directory."),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    """Count images, boxes, and classes in a YOLO dataset, and flag problems."""
    layout = discover_layout(source)
    report = inspect_dataset(layout.root, layout=layout)

    if as_json:
        console.print_json(data=report.to_dict())
        return

    console.print(f"[bold]{report.root}[/]")
    console.print(f"config: {report.config_path or '— none found, classes inferred from labels'}")

    split_table = Table("split", "images", "labelled", "boxes")
    for split in report.splits:
        split_table.add_row(
            split.name, str(split.images), str(split.labelled_images), str(split.instances)
        )
    console.print(split_table)

    counts = report.class_counts
    console.print(
        f"{len(report.class_names)} source classes, {len(counts)} present in labels, "
        f"{report.total_instances} boxes total"
    )

    suggestions = suggest_mapping(report.class_names)
    keep = [item for item in suggestions if item.target]
    console.print(
        f"[green]{len(keep)}[/] classes map onto the 32-card Baloot deck; "
        f"[yellow]{len(suggestions) - len(keep)}[/] would be dropped."
    )

    problems = [issue for issue in report.issues if issue.count]
    if problems:
        issue_table = Table("issue", "split", "count", "examples")
        for issue in problems:
            issue_table.add_row(
                issue.code, issue.split, str(issue.count), ", ".join(issue.examples[:3])
            )
        console.print(issue_table)
    else:
        console.print("[green]No label problems found.[/]")


@dataset_app.command("preview")
def cmd_dataset_preview(
    source: Path = typer.Option(..., "--source", "-s", help="Dataset root directory."),
    output: Path = typer.Option(
        Path("data/preview"), "--output", "-o", help="Directory to save preview images."
    ),
    count: int = typer.Option(
        16, "--count", "-n", min=1, help="Number of sample images to preview."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sample selection."),
    split: str | None = typer.Option(
        None, "--split", help="Specific split (e.g. train, valid, test)."
    ),
) -> None:
    """Render sample dataset images with bounding boxes and class labels for visual review."""
    rendered = preview_dataset(source, output, count=count, seed=seed, split_name=split)
    console.print(f"[green]Saved {len(rendered)} preview images to:[/] {output}")


@dataset_app.command("dedupe")
def cmd_dataset_dedupe(
    source: Path = typer.Option(..., "--source", "-s", help="Dataset root directory."),
    threshold: int = typer.Option(
        4,
        "--threshold",
        "-t",
        min=0,
        max=64,
        help="Hamming distance threshold for near-duplicates.",
    ),
    drop_leaks: bool = typer.Option(
        False, "--drop-leaks", help="Prune leaked eval images (in valid/test matching train)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Detect and optionally prune near-duplicate images across dataset splits using dHash."""
    layout = discover_layout(source)
    report = scan_dataset_duplicates(layout.root, threshold=threshold, layout=layout)

    if as_json:
        console.print_json(data=report.to_dict())
        return

    console.print(f"[bold]{report.root}[/]")
    console.print(
        f"Scanned {report.total_scanned} images with dHash threshold <= {report.threshold}"
    )
    console.print(f"Intra-split near-duplicates: {len(report.intra_split_matches)}")
    console.print(f"Cross-split near-duplicates: {len(report.cross_split_leaks)}")
    console.print(
        f"Leaked eval images (in valid/test matching train): {len(report.leaked_eval_images)}"
    )

    if drop_leaks:
        pruned = prune_leaked_images(report, layout=layout, dry_run=False)
        console.print(f"[green]Pruned {len(pruned)} leaked images from eval splits.[/]")
    elif report.leaked_eval_images:
        console.print("[yellow]Pass --drop-leaks to remove leaked images from valid/test.[/]")


@dataset_app.command("remap")
def cmd_dataset_remap(
    source: Path = typer.Option(..., "--source", "-s", help="Dataset root directory."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the Baloot dataset."),
    mapping: Path | None = typer.Option(
        None,
        "--mapping",
        help="JSON of {source class: Baloot class or null}. Omit to use the suggested mapping.",
    ),
    link_mode: str = typer.Option(
        "auto", "--link-mode", help="auto | link | copy — how images enter the new dataset."
    ),
    unmapped: str = typer.Option(
        "other", "--unmapped", help="other | drop — how non-Baloot cards (2-6) are handled."
    ),
    keep_empty: bool = typer.Option(
        False, "--keep-empty", help="Keep images whose boxes were all dropped."
    ),
) -> None:
    """Write a 33-class (or 32-class) Baloot dataset from a 52-card source dataset."""
    layout = discover_layout(source)
    report = inspect_dataset(layout.root, layout=layout)

    if mapping is not None:
        table: dict[str, str | None] = json.loads(mapping.read_text(encoding="utf-8"))
    else:
        table = {item.source_name: item.target for item in suggest_mapping(report.class_names)}

    kept = sum(1 for value in table.values() if value)
    console.print(f"Remapping {kept} Baloot classes (unmapped mode: '{unmapped}') -> {output}")

    result = remap_dataset(
        layout.root,
        output,
        table,
        layout=layout,
        link_mode=link_mode,
        drop_empty=not keep_empty,
        unmapped=unmapped,
    )
    console.print_json(data=result.to_dict())
    console.print(f"[green]Train with:[/] hakim-vision train --data {result.data_yaml}")


@app.command("train")
def cmd_train(
    data: Path = typer.Option(..., "--data", help="Dataset data.yaml (from `dataset remap`)."),
    model: str = typer.Option(
        "yolo11n.pt", "--model", help=f"One of: {', '.join(SUPPORTED_MODELS)}"
    ),
    epochs: int = typer.Option(120, "--epochs", min=1),
    imgsz: int = typer.Option(704, "--imgsz", min=64),
    batch: int = typer.Option(32, "--batch", min=1),
    workers: int = typer.Option(
        8, "--workers", min=1, help="Dataloader workers to prevent CPU bottlenecks."
    ),
    patience: int = typer.Option(20, "--patience", min=1),
    device: str = typer.Option(
        "", "--device", help="'0' for the first GPU, 'cpu', or blank to auto-detect."
    ),
    project: Path = typer.Option(Path("runs/hakim"), "--project"),
    name: str = typer.Option("baloot", "--name"),
    resume: bool = typer.Option(False, "--resume"),
    fliplr: float = typer.Option(
        0.0, "--fliplr", help="Horizontal flip probability (cards should be 0.0)."
    ),
    flipud: float = typer.Option(
        0.0, "--flipud", help="Vertical flip probability (cards should be 0.0)."
    ),
    degrees: float = typer.Option(180.0, "--degrees", help="Rotation augmentation in degrees."),
    cos_lr: bool = typer.Option(
        True, "--cos-lr/--no-cos-lr", help="Use cosine learning rate schedule."
    ),
    close_mosaic: int = typer.Option(
        10, "--close-mosaic", help="Disable mosaic augmentation for the last N epochs."
    ),
    scale: float = typer.Option(0.5, "--scale", help="Scale jitter factor."),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use automatic mixed precision."),
    half: bool = typer.Option(True, "--half/--no-half", help="Enable FP16 half precision."),
) -> None:
    """Fine-tune a card detector on a Baloot-class dataset (needs the `train` extra)."""
    config = TrainingConfig(
        data=data,
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        patience=patience,
        device=device,
        project=project,
        name=name,
        resume=resume,
        fliplr=fliplr,
        flipud=flipud,
        degrees=degrees,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        scale=scale,
        amp=amp,
        half=half,
    )
    outcome = train_detector(config)
    console.print_json(data=outcome.to_dict())
    if outcome.best_weights:
        console.print(
            f"[green]Export for the browser with:[/] "
            f"hakim-vision export --weights {outcome.best_weights} --imgsz {imgsz}"
        )


@app.command("export")
def cmd_export(
    weights: Path = typer.Option(..., "--weights", help="Trained .pt checkpoint."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Destination .onnx path."),
    imgsz: int = typer.Option(416, "--imgsz", min=64),
    half: bool = typer.Option(False, "--half", help="Export FP16 weights."),
) -> None:
    """Export trained weights to ONNX for the in-browser detector."""
    destination = export_yolo_to_onnx(weights, output, imgsz=imgsz, half=half)
    console.print(f"[green]Wrote[/] {destination}")


@app.command("quantize")
def cmd_quantize(
    model: Path = typer.Option(..., "--model", "-m", help="Float32 ONNX model to quantize."),
    output: Path = typer.Option(..., "--output", "-o", help="Destination INT8 ONNX path."),
    calib_dir: Path = typer.Option(
        ..., "--calib-dir", "-c", help="Directory containing calibration images."
    ),
    imgsz: int = typer.Option(416, "--imgsz", min=64),
    max_samples: int = typer.Option(200, "--max-samples", min=1),
) -> None:
    """Quantize a float32 ONNX model to static INT8 using calibration images."""
    from hakim_vision.models.quantize import quantize_onnx_static

    calib_images = sorted(
        [
            p
            for p in calib_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )[:max_samples]
    if not calib_images:
        raise typer.BadParameter(f"No valid calibration images found in {calib_dir}")

    dest = quantize_onnx_static(model, output, calib_images, imgsz=imgsz)
    console.print(f"[green]Quantized INT8 model written to:[/] {dest}")


@app.command("evaluate-parity")
def cmd_evaluate_parity(
    model: Path = typer.Option(..., "--model", "-m", help="ONNX model path."),
    test_images: Path = typer.Option(..., "--images", help="Test images directory."),
    test_labels: Path = typer.Option(..., "--labels", help="Test labels directory."),
    imgsz: int = typer.Option(416, "--imgsz", min=64),
    conf: float = typer.Option(0.25, "--conf", min=0.0, max=1.0),
    as_json: bool = typer.Option(False, "--json", help="Print result as JSON."),
) -> None:
    """Evaluate an ONNX model against holdout test images and labels."""
    from hakim_vision.models.evaluate_parity import evaluate_onnx_model

    result = evaluate_onnx_model(model, test_images, test_labels, imgsz=imgsz, conf_thresh=conf)
    if as_json:
        console.print_json(data=result.to_dict())
    else:
        table = Table("Metric", "Value")
        table.add_row("Model", result.model_name)
        table.add_row("Precision", f"{result.precision:.4f}")
        table.add_row("Recall", f"{result.recall:.4f}")
        table.add_row("mAP@50", f"{result.map50:.4f}")
        table.add_row("mAP@50-95", f"{result.map50_95:.4f}")
        table.add_row("Avg Latency", f"{result.avg_latency_ms:.2f} ms")
        table.add_row("Size", f"{result.model_size_mb:.2f} MB")
        console.print(table)


@app.command("studio")
def cmd_studio(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve the studio on."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically."),
    api: bool = typer.Option(
        True,
        "--api/--no-api",
        help="Expose the localhost job API so the studio can run dataset and training jobs.",
    ),
) -> None:
    """Serve the hand-value app and the dataset studio, with a local job API."""
    import webbrowser

    try:
        web_dir = default_web_dir()
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    project_root = Path.cwd()
    server, context = create_server(web_dir, port, project_root)
    base = f"http://127.0.0.1:{port}"
    studio_url = f"{base}/studio.html" + (f"#token={context.token}" if api else "")

    console.print(f"[bold green]✨ حاسبة يد البلوت | Hakim hand-value app:[/] {base}/index.html")
    console.print(f"[green]🔬 مختبر البيانات | Dataset studio:[/] {studio_url}")
    if api:
        console.print(
            "[dim]Job API on loopback only; the token in the studio URL authorises it. "
            "Do not share that URL.[/]"
        )
    else:
        console.print("[yellow]Job API disabled — the studio can inspect but not train.[/]")

    if open_browser:
        webbrowser.open(studio_url if api else f"{base}/index.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]تم إيقاف الاستوديو / Studio stopped.[/]")
    finally:
        context.jobs.stop_all()
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    app()
