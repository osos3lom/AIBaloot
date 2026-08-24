"""Render sample dataset images with bounding boxes and class labels."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np

from hakim_vision.datasets.yolo_layout import DatasetLayout, discover_layout

logger = logging.getLogger(__name__)


def render_preview_image(
    image_path: Path,
    label_path: Path | None,
    class_names: list[str],
) -> np.ndarray:
    """Read an image and draw its YOLO bounding boxes and labels."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]

    if label_path is not None and label_path.is_file():
        for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
            except ValueError:
                continue

            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(img_w - 1, int(cx + w / 2))
            y2 = min(img_h - 1, int(cy + h / 2))

            label_name = (
                class_names[class_id] if 0 <= class_id < len(class_names) else f"c{class_id}"
            )
            is_other = label_name.lower() == "other"
            color = (180, 180, 180) if is_other else (50, 205, 50)  # Gray for other, Lime for cards

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            text_size, _ = cv2.getTextSize(label_name, font, font_scale, thickness)
            tag_y = max(y1, text_size[1] + 4)
            cv2.rectangle(
                image,
                (x1, tag_y - text_size[1] - 4),
                (x1 + text_size[0] + 4, tag_y),
                color,
                -1,
            )
            cv2.putText(
                image,
                label_name,
                (x1 + 2, tag_y - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

    return image


def preview_dataset(
    source: Path,
    output_dir: Path,
    count: int = 16,
    seed: int = 42,
    split_name: str | None = None,
    layout: DatasetLayout | None = None,
) -> list[Path]:
    """Render `count` sample images from `source` dataset into `output_dir`.

    Args:
        source: Root dataset directory.
        output_dir: Where to save rendered preview images.
        count: Number of sample images to render.
        seed: Random seed for deterministic sample selection.
        split_name: Specific split name (e.g. 'train', 'valid', 'test') or None for all.
        layout: Optional pre-discovered dataset layout.

    Returns:
        List of generated image paths.
    """
    resolved = layout or discover_layout(source)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s for s in resolved.splits if s.name == split_name] if split_name else resolved.splits
    if not splits:
        raise ValueError(f"No matching splits found for split: '{split_name}'")

    all_pairs: list[tuple[Path, Path | None, str]] = []
    for split in splits:
        for img_path in sorted(split.images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                continue
            lbl_path = split.labels_dir / f"{img_path.stem}.txt" if split.labels_dir else None
            all_pairs.append((img_path, lbl_path, split.name))

    if not all_pairs:
        raise ValueError(f"No images found in dataset at {source}")

    rng = random.Random(seed)  # noqa: S311
    selected = rng.sample(all_pairs, min(count, len(all_pairs)))
    rendered_paths: list[Path] = []

    for i, (img_path, lbl_path, s_name) in enumerate(selected):
        rendered = render_preview_image(img_path, lbl_path, resolved.class_names)
        out_file = output_dir / f"preview_{i:03d}_{s_name}_{img_path.name}"
        cv2.imwrite(str(out_file), rendered)
        rendered_paths.append(out_file)

    logger.info("Rendered %d preview images to %s", len(rendered_paths), output_dir)
    return rendered_paths


__all__ = [
    "preview_dataset",
    "render_preview_image",
]
