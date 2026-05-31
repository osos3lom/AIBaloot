"""Scene compositor: place 2 or 3 cards on a background, emit YOLO labels.

Design notes:

* Augmentation is a single OpenCV affine via ``random_affine_card``; richer
  photometric/elastic effects can be layered on top later.
* No global RNG. The caller injects ``numpy.random.Generator``.
* Occlusion handling is explicit: a placed card's corner-symbol hull is kept
  only if the **next** card (which renders on top) does not occlude more than
  ``intersect_ratio`` of it. The bounding box for a kept corner is the
  axis-aligned bbox around the transformed hull points, clipped to the canvas.
* Returns a typed result; the caller saves files (PNG + YOLO txt) or feeds the
  data straight into a training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from hakim_vision.geometry import YoloBox, voc_to_yolo
from hakim_vision.synthetic.assets import Backgrounds, CardSample, Cards
from hakim_vision.synthetic.constants import SCENE_SIZE
from hakim_vision.synthetic.hull import points_to_bbox, points_to_polygon
from hakim_vision.synthetic.transforms import (
    AugmentRange,
    PlacedCard,
    random_affine_card,
)

# Maximum fraction of a corner-hull that may be occluded by a later card
# before we drop that corner's bounding box from the labels.
_DEFAULT_INTERSECT_RATIO: float = 0.10


@dataclass(frozen=True)
class SceneLabel:
    """One labelled bounding box in a generated scene."""

    class_name: str
    yolo: YoloBox
    voc: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)


@dataclass(frozen=True)
class Scene:
    """A composited scene with labels."""

    image: NDArray[np.uint8]
    """BGR scene image of shape ``(canvas_size, canvas_size, 3)``."""

    labels: list[SceneLabel] = field(default_factory=list)
    """One or more bounding-box labels."""

    canvas_size: int = SCENE_SIZE


def _alpha_composite(base: NDArray[np.uint8], overlay_bgra: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Composite ``overlay_bgra`` onto BGR ``base`` using overlay's alpha channel."""
    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_bgra[:, :, :3].astype(np.float32)
    base_f = base.astype(np.float32)
    out = overlay_rgb * alpha + base_f * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _safe_polygon(points: NDArray[np.float32]) -> Polygon | None:
    if points.shape[0] < 3:
        return None
    poly = points_to_polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (poly.is_valid and poly.area > 0) else None


def _kept_after_occlusion(
    hull_points: NDArray[np.float32],
    occluder_card_corners: NDArray[np.float32] | None,
    *,
    intersect_ratio: float,
) -> bool:
    """Return True if the hull is mostly NOT covered by the occluder card."""
    if occluder_card_corners is None:
        return True
    hull_poly = _safe_polygon(hull_points)
    occluder_poly = _safe_polygon(occluder_card_corners)
    if hull_poly is None or occluder_poly is None:
        return False
    inter = hull_poly.intersection(occluder_poly).area
    visible_frac = (hull_poly.area - inter) / hull_poly.area
    return visible_frac >= (1.0 - intersect_ratio)


def _labels_for_card(
    placed: PlacedCard,
    *,
    class_name: str,
    canvas_size: int,
    occluder: PlacedCard | None,
    intersect_ratio: float,
) -> list[SceneLabel]:
    """Compute up to two corner labels for one placed card."""
    occluder_corners = occluder.card_corners if occluder is not None else None
    labels: list[SceneLabel] = []
    for hull_pts in (placed.hull_hl_points, placed.hull_lr_points):
        if not _kept_after_occlusion(
            hull_pts, occluder_corners, intersect_ratio=intersect_ratio
        ):
            continue
        bbox = points_to_bbox(hull_pts, image_size=(canvas_size, canvas_size))
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        yolo = voc_to_yolo((canvas_size, canvas_size), (xmin, xmax, ymin, ymax))
        labels.append(SceneLabel(class_name=class_name, yolo=yolo, voc=bbox))
    return labels


def compose_scene(
    background_bgr: NDArray[np.uint8],
    placed_cards: Sequence[tuple[PlacedCard, str]],
    *,
    canvas_size: int = SCENE_SIZE,
    intersect_ratio: float = _DEFAULT_INTERSECT_RATIO,
) -> Scene:
    """Composite N placed cards onto a background and emit labels.

    Cards are drawn back-to-front in the given order. A card's corner-symbol
    bounding box is kept only if no *later* card occludes more than
    ``intersect_ratio`` of that corner's hull.

    Args:
        background_bgr: ``(H, W, 3)`` BGR background; resized if needed.
        placed_cards: Sequence of ``(PlacedCard, class_name)`` pairs in
            back-to-front order.
        canvas_size: Target scene size (square).
        intersect_ratio: Maximum allowable corner occlusion ratio.

    Returns:
        A ``Scene`` with the composited image and per-corner labels.
    """
    if background_bgr.shape[:2] != (canvas_size, canvas_size):
        background_bgr = cv2.resize(
            background_bgr, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA
        )

    image = background_bgr.copy()
    for placed, _cls in placed_cards:
        image = _alpha_composite(image, placed.image)

    labels: list[SceneLabel] = []
    for i, (placed, cls) in enumerate(placed_cards):
        # Any card drawn after this one is a potential occluder. We use the
        # closest later card's outline as a conservative occlusion approximation
        # (mirrors the legacy 2-card / 3-card behavior).
        occluder = placed_cards[i + 1][0] if i + 1 < len(placed_cards) else None
        labels.extend(
            _labels_for_card(
                placed,
                class_name=cls,
                canvas_size=canvas_size,
                occluder=occluder,
                intersect_ratio=intersect_ratio,
            )
        )
    return Scene(image=image, labels=labels, canvas_size=canvas_size)


def render_random_scene(
    cards: Cards,
    backgrounds: Backgrounds,
    *,
    rng: np.random.Generator,
    n_cards: int = 2,
    canvas_size: int = SCENE_SIZE,
    aug: AugmentRange = AugmentRange(),
    intersect_ratio: float = _DEFAULT_INTERSECT_RATIO,
) -> Scene:
    """Sample N cards + 1 background and produce a fully composited scene."""
    if n_cards not in (2, 3):
        raise ValueError(f"n_cards must be 2 or 3, got {n_cards}")

    bg = backgrounds.sample(size=canvas_size)
    samples: list[CardSample] = [cards.sample() for _ in range(n_cards)]
    placed: list[tuple[PlacedCard, str]] = []
    for sample in samples:
        rgb_card = sample.image
        if rgb_card.shape[2] == 3:
            rgb_card = cv2.cvtColor(rgb_card, cv2.COLOR_BGR2BGRA)
        placed_card = random_affine_card(
            rgb_card,
            sample.hull_hl,
            sample.hull_lr,
            rng=rng,
            canvas_size=canvas_size,
            aug=aug,
        )
        placed.append((placed_card, sample.name))

    return compose_scene(
        bg, placed, canvas_size=canvas_size, intersect_ratio=intersect_ratio
    )


def write_yolo_label(scene: Scene, class_to_id: dict[str, int]) -> str:
    """Format a scene's labels as a YOLO ``.txt`` payload."""
    lines: list[str] = []
    for label in scene.labels:
        if label.class_name not in class_to_id:
            continue
        cls_id = class_to_id[label.class_name]
        b = label.yolo
        lines.append(f"{cls_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


__all__ = [
    "Scene",
    "SceneLabel",
    "compose_scene",
    "render_random_scene",
    "write_yolo_label",
]
