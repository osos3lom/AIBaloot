"""YOLO Model Training, Evaluation, and WebGPU ONNX Export Pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BALOOT_RANKS: tuple[str, ...] = ("A", "K", "Q", "J", "10", "9", "8", "7")
BALOOT_SUITS: tuple[str, ...] = ("h", "d", "c", "s")
BALOOT_OTHER_CLASS = "other"


def get_baloot_classes(include_other: bool = False) -> list[str]:
    """Return the canonical Saudi Baloot playing card class names.

    If include_other is True, appends the 33rd 'other' class (used for non-Baloot cards 2-6).
    """
    classes: list[str] = []
    for suit in BALOOT_SUITS:
        for rank in BALOOT_RANKS:
            classes.append(f"{rank}{suit}")
    if include_other:
        classes.append(BALOOT_OTHER_CLASS)
    return classes


def get_baloot_detection_classes() -> list[str]:
    """Return the 33 class names used in the Baloot detector (32 cards + other)."""
    return get_baloot_classes(include_other=True)


def generate_dataset_yaml(
    output_path: Path,
    dataset_root: Path,
    train_split: str = "train",
    val_split: str = "val",
    include_other: bool = True,
) -> Path:
    """Generate standard Ultralytics YOLO dataset.yaml configuration file."""
    classes = get_baloot_classes(include_other=include_other)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        f"train: images/{train_split}",
        f"val: images/{val_split}",
        "names:",
    ]
    for idx, name in enumerate(classes):
        yaml_lines.append(f"  {idx}: {name}")

    content = "\n".join(yaml_lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    logger.info("Generated Baloot YOLO dataset configuration at: %s", output_path)
    return output_path


def export_yolo_to_onnx(
    model_path: Path | str,
    output_path: Path | str | None = None,
    imgsz: int = 416,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int = 17,
) -> Path:
    """Export a trained YOLO model to ONNX format optimized for WebGPU inference.

    Uses opset=17, simplify=True, and no embedded NMS (decoding/NMS happens in JS).
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install with `uv sync --extra train`."
        ) from error

    model = YOLO(str(model_path))
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
    )
    dest = Path(output_path) if output_path else Path(exported_file)
    if output_path and Path(exported_file).resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil_dest = dest
        import shutil

        shutil.copy2(exported_file, shutil_dest)
    logger.info("Successfully exported model to ONNX: %s", dest)
    return dest
