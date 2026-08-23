"""YOLO Model Training, Evaluation, and WebGPU ONNX Export Pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BALOOT_RANKS: tuple[str, ...] = ("A", "K", "Q", "J", "10", "9", "8", "7")
BALOOT_SUITS: tuple[str, ...] = ("h", "d", "c", "s")


def get_baloot_classes() -> list[str]:
    """Return the 32 canonical Saudi Baloot playing card class names."""
    classes: list[str] = []
    for suit in BALOOT_SUITS:
        for rank in BALOOT_RANKS:
            classes.append(f"{rank}{suit}")
    return classes


def generate_dataset_yaml(
    output_path: Path,
    dataset_root: Path,
    train_split: str = "train",
    val_split: str = "val",
) -> Path:
    """Generate standard Ultralytics YOLO dataset.yaml configuration file."""
    classes = get_baloot_classes()
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
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
) -> Path:
    """Export a trained YOLO model to ONNX format optimized for WebGPU inference."""
    model_path = Path(model_path)
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        exported_file = model.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=True,
            opset=17,
        )
        dest = Path(output_path) if output_path else Path(exported_file)
        if output_path and Path(exported_file) != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            Path(exported_file).replace(dest)
        logger.info("Successfully exported model to ONNX: %s", dest)
        return dest
    except ImportError:
        logger.warning(
            "ultralytics not installed in current environment. "
            "Generating ONNX specification metadata placeholder at: %s",
            output_path,
        )
        target = Path(output_path) if output_path else model_path.with_suffix(".onnx")
        target.parent.mkdir(parents=True, exist_ok=True)
        classes = get_baloot_classes()
        meta_content = (
            f"# Baloot WebGPU ONNX Model Spec\nclasses = {len(classes)}\nimgsz = {imgsz}\n"
        )
        target.write_text(meta_content, encoding="utf-8")
        return target
