"""Geometry helpers extracted from the legacy notebook.

Initial slice: bounding-box conversion to YOLO format. The full corner-detection
and keypoint pipeline (`findHull`, `hull_to_kps`, `kps_to_BB`) will be ported in
follow-up PRs as the notebook is decomposed.
"""

from __future__ import annotations

from typing import NamedTuple


class YoloBox(NamedTuple):
    """A normalized YOLO bounding box (cx, cy, w, h), all in [0, 1]."""

    cx: float
    cy: float
    w: float
    h: float


def voc_to_yolo(
    image_size: tuple[int, int],
    box: tuple[float, float, float, float],
) -> YoloBox:
    """Convert a VOC-style (xmin, xmax, ymin, ymax) box to normalized YOLO format.

    Args:
        image_size: (width, height) in pixels.
        box: (xmin, xmax, ymin, ymax) in pixels.

    Returns:
        A `YoloBox` with values normalized to the image dimensions.

    Raises:
        ValueError: if `image_size` has non-positive dimensions or the box is empty.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    xmin, xmax, ymin, ymax = box
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"degenerate box: {box}")

    cx = ((xmin + xmax) / 2.0) / width
    cy = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return YoloBox(cx=cx, cy=cy, w=w, h=h)


__all__ = ["YoloBox", "voc_to_yolo"]
