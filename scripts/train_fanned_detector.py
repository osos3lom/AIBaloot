#!/usr/bin/env python
"""End-to-end GPU training run for a fanned-hand Baloot card detector.

The shipped ``baloot-v1`` model was trained on 2-3 card composites with at most
10% corner occlusion, and its own model card names the consequence: recall drops
on real photos of a held hand. This script closes that gap. It builds a card
atlas from the HuggingFace ``JackFurby/playing-cards`` dataset, renders dense
fanned hands with draw-order occlusion, fingers, shadows and camera tilt, mixes
in whatever real or existing data you point it at, trains YOLO11, and exports
the ONNX pair the browser runtime loads.

Stages run in order and each one is skippable, so a re-run after a crash costs
only what it has to:

    python scripts/train_fanned_detector.py all --device 0

    python scripts/train_fanned_detector.py atlas      # download + card templates
    python scripts/train_fanned_detector.py preview    # eyeball labels first
    python scripts/train_fanned_detector.py dataset    # render the fan dataset
    python scripts/train_fanned_detector.py train --device 0
    python scripts/train_fanned_detector.py export

Install the training extra first:  uv sync --extra train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):  # running the file directly, without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hakim_vision.datasets.hf_cards import HF_REPO, build_card_atlas
from hakim_vision.models.train import TrainingConfig, train_detector
from hakim_vision.models.yolo_export import (
    export_yolo_to_onnx,
    generate_dataset_yaml,
    get_baloot_detection_classes,
)
from hakim_vision.synthetic.fan import CardAtlas, FanConfig, load_backgrounds, render_fan_scene
from hakim_vision.synthetic.scene import Scene

logger = logging.getLogger("train_fanned_detector")

#: Detector input edge. Must match ``imgsz`` in web/models/model.json, because the
#: browser stretch-resizes to exactly this and a mismatch silently costs accuracy.
DEFAULT_IMGSZ = 704

#: Fraction of rendered scenes held out for validation.
VAL_FRACTION = 0.12

CLASS_NAMES = get_baloot_detection_classes()
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Paths:
    """Every directory the run reads or writes, derived from one work root."""

    root: Path

    @property
    def cache(self) -> Path:
        return self.root / "hf-cache"

    @property
    def atlas(self) -> Path:
        return self.root / "card-atlas"

    @property
    def dataset(self) -> Path:
        return self.root / "fan-dataset"

    @property
    def data_yaml(self) -> Path:
        return self.dataset / "data.yaml"

    @property
    def preview(self) -> Path:
        return self.root / "preview"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def exports(self) -> Path:
        return self.root / "exports"


# --------------------------------------------------------------------------- #
# Stage 1 - card atlas
# --------------------------------------------------------------------------- #


def stage_atlas(paths: Paths, args: argparse.Namespace) -> None:
    """Download a slice of the HF dataset and extract canonical card templates."""
    if paths.atlas.is_dir() and any(paths.atlas.glob("*.png")) and not args.force:
        logger.info("atlas already present at %s (use --force to rebuild)", paths.atlas)
        return
    atlas = build_card_atlas(
        paths.atlas,
        cache_dir=paths.cache,
        per_class=args.per_class,
        repo=args.repo,
    )
    missing = 52 - len(atlas)
    if missing > 0:
        logger.warning("atlas is missing %d of 52 deck classes", missing)
    logger.info("wrote %d templates to %s", sum(len(v) for v in atlas.values()), paths.atlas)


# --------------------------------------------------------------------------- #
# Stage 2 - render fanned hands
# --------------------------------------------------------------------------- #


def _fan_config(args: argparse.Namespace) -> FanConfig:
    return FanConfig(
        canvas=args.imgsz,
        cards_min=args.cards_min,
        cards_max=args.cards_max,
        min_visible=args.min_visible,
        hand_prob=args.hand_prob,
    )


def _render_labelled(
    atlas: CardAtlas,
    rng: np.random.Generator,
    config: FanConfig,
    backgrounds: Sequence[np.ndarray],
    *,
    min_labels: int = 2,
    attempts: int = 4,
) -> Scene:
    """Render a fan, resampling a few times if occlusion buried nearly every index.

    A hand where the pivot and fingers happen to cover all but one card is a real
    photo, just a nearly contentless training sample. Retrying is cheaper than
    spending an epoch's worth of GPU time on empty canvases.
    """
    scene = render_fan_scene(atlas, rng=rng, config=config, backgrounds=backgrounds)
    for _ in range(attempts - 1):
        if len(scene.labels) >= min_labels:
            break
        scene = render_fan_scene(atlas, rng=rng, config=config, backgrounds=backgrounds)
    return scene


def _write_scene(scene: Scene, images_dir: Path, labels_dir: Path, stem: str) -> bool:
    """Write one scene as a JPEG plus its YOLO label file. False if unlabelled."""
    lines = [
        f"{CLASS_INDEX[label.class_name]} "
        f"{label.yolo.cx:.6f} {label.yolo.cy:.6f} {label.yolo.w:.6f} {label.yolo.h:.6f}"
        for label in scene.labels
        if label.class_name in CLASS_INDEX
    ]
    if not lines:
        return False
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), scene.image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


@dataclass(frozen=True)
class RenderChunk:
    """One worker's slice of the render: contiguous indices and its own seed."""

    start: int
    count: int
    seed: int
    atlas_dir: Path
    dataset: Path
    backgrounds: Path | None
    config: FanConfig


def render_chunk(chunk: RenderChunk) -> tuple[int, int, int]:
    """Render and write one slice. Returns ``(train, val, dropped)`` counts.

    Each worker reloads the atlas and seeds its own generator from the chunk
    index, so the split across processes changes what is rendered but a given
    ``--seed`` plus ``--jobs`` still reproduces exactly.
    """
    atlas = CardAtlas.from_directory(chunk.atlas_dir)
    backgrounds = load_backgrounds(chunk.backgrounds)
    rng = np.random.default_rng(chunk.seed)
    train = val = dropped = 0

    for offset in range(chunk.count):
        scene = _render_labelled(atlas, rng, chunk.config, backgrounds)
        split = "val" if rng.random() < VAL_FRACTION else "train"
        written = _write_scene(
            scene,
            chunk.dataset / "images" / split,
            chunk.dataset / "labels" / split,
            f"fan_{chunk.start + offset:06d}",
        )
        if not written:
            dropped += 1
        elif split == "train":
            train += 1
        else:
            val += 1
    return train, val, dropped


def stage_dataset(paths: Paths, args: argparse.Namespace) -> None:
    """Render the fan dataset, then merge in any extra YOLO datasets."""
    if not paths.atlas.is_dir():
        raise FileNotFoundError(f"no card atlas at {paths.atlas} - run the `atlas` stage first")

    for split in ("train", "val"):
        (paths.dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (paths.dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, args.count))
    logger.info("rendering %d scenes across %d worker(s)", args.count, jobs)

    per_job, remainder = divmod(args.count, jobs)
    chunks: list[RenderChunk] = []
    start = 0
    for index in range(jobs):
        count = per_job + (1 if index < remainder else 0)
        chunks.append(
            RenderChunk(
                start=start,
                count=count,
                seed=args.seed + index,
                atlas_dir=paths.atlas,
                dataset=paths.dataset,
                backgrounds=Path(args.backgrounds) if args.backgrounds else None,
                config=_fan_config(args),
            )
        )
        start += count

    written = {"train": 0, "val": 0}
    empty = 0
    if jobs == 1:
        results = [render_chunk(chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(render_chunk, chunks))
    for train, val, dropped in results:
        written["train"] += train
        written["val"] += val
        empty += dropped

    if empty:
        logger.warning("%d scenes produced no visible label and were dropped", empty)

    for extra in args.merge or []:
        merged = _merge_dataset(Path(extra), paths.dataset)
        logger.info("merged %d images from %s", merged, extra)

    generate_dataset_yaml(paths.data_yaml, paths.dataset, include_other=True)
    logger.info(
        "dataset ready: train=%d val=%d -> %s", written["train"], written["val"], paths.data_yaml
    )


def _merge_dataset(source: Path, destination: Path) -> int:
    """Copy an existing YOLO dataset's splits into the fan dataset.

    Label files are copied verbatim, so the source must already use the same
    33-class indexing - which is what ``hakim-vision dataset remap`` produces.
    """
    copied = 0
    for split_dir in sorted((source / "images").iterdir()) if (source / "images").is_dir() else []:
        split = "val" if split_dir.name in {"val", "valid", "validation"} else split_dir.name
        if split not in {"train", "val"}:
            continue
        labels_dir = source / "labels" / split_dir.name
        for image_path in sorted(split_dir.iterdir()):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            prefix = f"{source.name}_{image_path.stem}"
            shutil.copy2(
                image_path, destination / "images" / split / f"{prefix}{image_path.suffix}"
            )
            shutil.copy2(label_path, destination / "labels" / split / f"{prefix}.txt")
            copied += 1
    return copied


# --------------------------------------------------------------------------- #
# Stage 3 - preview
# --------------------------------------------------------------------------- #


def stage_preview(paths: Paths, args: argparse.Namespace) -> None:
    """Render a handful of annotated scenes so labels can be checked by eye.

    Worth doing before every long run: a geometry bug here is invisible in the
    loss curve and obvious in a single picture.
    """
    if not paths.atlas.is_dir():
        raise FileNotFoundError(f"no card atlas at {paths.atlas} - run the `atlas` stage first")
    paths.preview.mkdir(parents=True, exist_ok=True)

    atlas = CardAtlas.from_directory(paths.atlas)
    backgrounds = load_backgrounds(Path(args.backgrounds) if args.backgrounds else None)
    config = _fan_config(args)
    rng = np.random.default_rng(args.seed)

    for index in range(args.preview_count):
        scene = _render_labelled(atlas, rng, config, backgrounds)
        canvas = scene.image.copy()
        for label in scene.labels:
            xmin, ymin, xmax, ymax = label.voc
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                label.class_name,
                (xmin, max(12, ymin - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(paths.preview / f"preview_{index:03d}.jpg"), canvas)
    logger.info("wrote %d previews to %s", args.preview_count, paths.preview)


# --------------------------------------------------------------------------- #
# Stage 4 - train
# --------------------------------------------------------------------------- #


def stage_train(paths: Paths, args: argparse.Namespace) -> Path | None:
    """Fine-tune YOLO11 on the fan dataset.

    The augmentation choices are the ones this target actually needs:

    * ``fliplr`` / ``flipud`` stay at 0 - a mirrored rank glyph is a different,
      nonexistent card, and teaching the model to accept one costs precision.
    * ``degrees=180`` because a hand can be held at any angle.
    * ``mosaic`` stays on for most of the run: four scenes per tile shrinks the
      indices, which is the regime that fails today. ``close_mosaic`` then hands
      the last epochs clean full-size scenes for sharp localisation.
    * ``scale`` is widened over the shipped run so the model sees hands held both
      close to the lens and across the table.
    """
    if not paths.data_yaml.is_file():
        raise FileNotFoundError(f"no dataset at {paths.data_yaml} - run the `dataset` stage first")

    config = TrainingConfig(
        data=paths.data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=paths.runs,
        name=args.name,
        seed=args.seed,
        resume=args.resume,
        fliplr=0.0,
        flipud=0.0,
        degrees=180.0,
        scale=0.6,
        close_mosaic=args.close_mosaic,
    )
    outcome = train_detector(config)
    logger.info("run directory: %s", outcome.run_dir)
    logger.info("metrics: %s", json.dumps(outcome.metrics, indent=2, sort_keys=True))
    if outcome.best_weights is None:
        logger.error("training produced no best.pt")
        return None
    logger.info("best weights: %s", outcome.best_weights)
    return outcome.best_weights


# --------------------------------------------------------------------------- #
# Stage 5 - export
# --------------------------------------------------------------------------- #


def _calibration_images(dataset: Path, limit: int = 200) -> list[Path]:
    images = sorted((dataset / "images" / "train").glob("*.jpg"))
    if not images:
        return []
    step = max(1, len(images) // limit)
    return images[::step][:limit]


def stage_export(paths: Paths, args: argparse.Namespace, weights: Path | None = None) -> None:
    """Export FP16 and INT8 ONNX, matching what ``web/model-runner.js`` loads."""
    checkpoint = Path(args.weights) if args.weights else weights
    if checkpoint is None:
        candidate = paths.runs / args.name / "weights" / "best.pt"
        checkpoint = candidate if candidate.is_file() else None
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError("no weights to export - pass --weights or run the `train` stage")

    paths.exports.mkdir(parents=True, exist_ok=True)
    fp16 = paths.exports / f"{args.name}.fp16.onnx"
    export_yolo_to_onnx(checkpoint, fp16, imgsz=args.imgsz, half=True)
    logger.info("FP16 ONNX: %s", fp16)

    if args.skip_int8:
        return
    calibration = _calibration_images(paths.dataset)
    if not calibration:
        logger.warning("no calibration images found; skipping INT8")
        return

    from hakim_vision.models.quantize import quantize_onnx_static

    fp32 = paths.exports / f"{args.name}.fp32.onnx"
    export_yolo_to_onnx(checkpoint, fp32, imgsz=args.imgsz, half=False)
    int8 = paths.exports / f"{args.name}.int8.onnx"
    quantize_onnx_static(fp32, int8, calibration, imgsz=args.imgsz)
    logger.info("INT8 ONNX: %s", int8)

    metadata = {
        "version": args.name,
        "name": Path(args.model).stem,
        "imgsz": args.imgsz,
        "preprocess": "stretch",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "other_index": CLASS_INDEX["other"],
    }
    (paths.exports / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("copy %s/*.onnx and model.json into web/models/ to deploy", paths.exports)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=("atlas", "preview", "dataset", "train", "export", "all"),
        help="Which stage to run. `all` runs atlas -> dataset -> train -> export.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/fan-training"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild stages that would be skipped."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--repo", default=HF_REPO, help="HuggingFace dataset to source cards from.")
    data.add_argument("--per-class", type=int, default=4, help="Card templates per deck class.")
    data.add_argument("--count", type=int, default=12000, help="Fanned scenes to render.")
    data.add_argument("--cards-min", type=int, default=5)
    data.add_argument("--cards-max", type=int, default=13)
    data.add_argument(
        "--min-visible",
        type=float,
        default=0.45,
        help="Fraction of a corner index that must stay visible for it to be labelled.",
    )
    data.add_argument("--hand-prob", type=float, default=0.75, help="Chance of drawing a hand.")
    data.add_argument("--backgrounds", type=Path, help="Directory of background photos.")
    data.add_argument(
        "--merge",
        action="append",
        help="Existing 33-class YOLO dataset root to fold in. Repeatable.",
    )
    data.add_argument("--preview-count", type=int, default=24)
    data.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Render worker processes. 0 uses every core; rendering is CPU-bound.",
    )

    train = parser.add_argument_group("training")
    train.add_argument("--model", default="yolo11n.pt", help="Starting checkpoint.")
    train.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    train.add_argument("--epochs", type=int, default=180)
    train.add_argument("--batch", type=int, default=32, help="Use -1 for Ultralytics auto-batch.")
    train.add_argument("--device", default="0", help='GPU index, "cpu", or "0,1" for multi-GPU.')
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--close-mosaic", type=int, default=20)
    train.add_argument("--name", default="baloot-fan-v2")
    train.add_argument("--resume", action="store_true")

    export = parser.add_argument_group("export")
    export.add_argument("--weights", type=Path, help="Checkpoint to export instead of the run's.")
    export.add_argument("--skip-int8", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = Paths(root=Path(args.work_dir))
    paths.root.mkdir(parents=True, exist_ok=True)

    if args.stage == "atlas":
        stage_atlas(paths, args)
    elif args.stage == "preview":
        stage_preview(paths, args)
    elif args.stage == "dataset":
        stage_dataset(paths, args)
    elif args.stage == "train":
        stage_train(paths, args)
    elif args.stage == "export":
        stage_export(paths, args)
    else:
        stage_atlas(paths, args)
        stage_preview(paths, args)
        stage_dataset(paths, args)
        best = stage_train(paths, args)
        stage_export(paths, args, weights=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
