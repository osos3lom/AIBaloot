"""Render dense fanned hands: the scene the 2-3 card compositor never produced.

:mod:`hakim_vision.synthetic.scene` lays 2 or 3 cards on a background with at
most 10% corner occlusion. A held hand looks nothing like that. Thirteen cards
pivot around a point below the frame, each one covering all but a narrow strip
of the card behind it, fingers cover the bottom, and the only thing left of most
cards is the rank/suit index - sometimes half of it.

This module renders that scene directly, so the detector trains on the geometry
it actually has to survive:

* **Fan geometry.** Cards share one pivot and a per-card angular step, so the
  overlap pattern is the real one rather than random scatter.
* **Draw-order occlusion.** Each card's two index quads are tested against the
  union of everything drawn on top of it - later cards, and the hand. A label
  survives only if enough of it is still visible, and its box is the bounding
  box of the *visible* remainder, not of the whole index.
* **Hand, shadow, and camera.** A skin-toned palm and fingers, per-card contact
  shadows, a keystone homography for camera tilt, and photometric noise, because
  the reported failures are photos of a hand, not composites.

Card templates come from :mod:`hakim_vision.datasets.hf_cards`; backgrounds are
any directory of images.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon
from shapely.ops import unary_union

from hakim_vision.datasets.remap import normalise_card_name
from hakim_vision.geometry import voc_to_yolo
from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    REF_CARD,
    REF_CORNER_HL,
    REF_CORNER_LR,
)
from hakim_vision.synthetic.scene import Scene, SceneLabel

logger = logging.getLogger(__name__)

#: Class name for ranks 2-6, matching the deployed 33-class head.
OTHER_CLASS: str = "other"

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


@dataclass(frozen=True)
class FanConfig:
    """Ranges the renderer samples one fanned hand from.

    Defaults describe a phone photo of a held Baloot hand: eight cards or more,
    fanned 35-105 degrees around a pivot near the bottom of the frame, shot
    slightly off-axis.
    """

    canvas: int = 704
    """Output edge length. Match the detector's ``imgsz`` to avoid a resample."""

    cards_min: int = 5
    cards_max: int = 13
    spread_deg: tuple[float, float] = (25.0, 105.0)
    """Total angle from the first card to the last. Low values crowd the indices."""

    tilt_deg: tuple[float, float] = (-40.0, 40.0)
    """Rotation of the fan as a whole."""

    card_height_frac: tuple[float, float] = (0.30, 0.68)
    """Card height as a fraction of the canvas."""

    grip_frac: tuple[float, float] = (0.02, 0.55)
    """Pivot distance below the card's bottom edge, in card heights."""

    pivot_x_frac: tuple[float, float] = (0.25, 0.75)
    pivot_y_frac: tuple[float, float] = (0.80, 1.30)
    """Pivot position. Beyond 1.0 puts it below the frame, as in a real photo."""

    perspective: float = 0.10
    """Keystone strength, as a fraction of the canvas. 0 disables camera tilt."""

    reverse_draw_prob: float = 0.3
    """Chance the fan is stacked the other way, exposing bottom-right indices.

    Mirroring a fan means changing which card is on top, not which way the angles
    run. Negating the spread while still drawing left-to-right produces a hand
    whose every index is buried - physically possible, useless to train on.
    """

    upside_down_prob: float = 0.06
    """Per-card chance of being held rotated 180 degrees."""

    hand_prob: float = 0.75
    shadow_strength: float = 0.35
    min_visible: float = 0.45
    """Fraction of an index that must survive occlusion for it to stay labelled."""

    min_box_px: int = 7
    """Drop boxes whose visible remainder is smaller than this on either side."""

    blur_sigma: tuple[float, float] = (0.0, 1.3)
    noise_sigma: tuple[float, float] = (0.0, 7.0)
    jpeg_quality: tuple[int, int] = (45, 95)
    brightness: tuple[float, float] = (0.72, 1.20)


class CardAtlas:
    """Canonical BGRA card templates, keyed by deck name (``"10H"``, ``"AS"``)."""

    def __init__(self, templates: dict[str, list[NDArray[np.uint8]]]) -> None:
        if not templates:
            raise ValueError("card atlas is empty")
        self._templates = templates
        self._names = sorted(templates)

    @classmethod
    def from_directory(cls, directory: Path) -> CardAtlas:
        """Load ``<DECK NAME>_<n>.png`` templates written by ``build_card_atlas``."""
        directory = Path(directory)
        templates: dict[str, list[NDArray[np.uint8]]] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                logger.warning("skipping non-BGRA template: %s", path)
                continue
            if image.shape[:2] != (CARD_HEIGHT, CARD_WIDTH):
                image = cv2.resize(image, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_AREA)
            templates.setdefault(path.stem.rsplit("_", 1)[0], []).append(image.astype(np.uint8))
        return cls(templates)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def sample(self, name: str, rng: np.random.Generator) -> NDArray[np.uint8]:
        """One template for a deck class, chosen at random among its variants."""
        variants = self._templates[name]
        return variants[int(rng.integers(len(variants)))]


def _homography(matrix: np.ndarray) -> NDArray[np.float32]:
    """Narrow any OpenCV / NumPy matrix to the float32 3x3 the warps expect."""
    result: NDArray[np.float32] = np.asarray(matrix, dtype=np.float32)
    return result


def _translation(dx: float, dy: float) -> NDArray[np.float32]:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotation_scale(angle_deg: float, scale: float) -> NDArray[np.float32]:
    radians = math.radians(angle_deg)
    cos, sin = math.cos(radians) * scale, math.sin(radians) * scale
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _keystone(canvas: int, strength: float, rng: np.random.Generator) -> NDArray[np.float32]:
    """A mild perspective warp standing in for an off-axis camera."""
    if strength <= 0.0:
        return np.eye(3, dtype=np.float32)
    span = canvas * strength
    source = np.array([[0, 0], [canvas, 0], [canvas, canvas], [0, canvas]], dtype=np.float32)
    offsets = rng.uniform(-span, span, size=(4, 2)).astype(np.float32)
    return _homography(cv2.getPerspectiveTransform(source, source + offsets))


def _project(matrix: NDArray[np.float32], points: NDArray[np.float32]) -> NDArray[np.float32]:
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    result: NDArray[np.float32] = projected.astype(np.float32)
    return result


def _polygon(points: np.ndarray) -> Polygon | None:
    if points.shape[0] < 3:
        return None
    poly = Polygon(points.tolist())
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if poly.is_valid and poly.area > 0 else None


def _card_placements(
    count: int,
    config: FanConfig,
    rng: np.random.Generator,
) -> list[NDArray[np.float32]]:
    """One homography per card, ordered back to front.

    Every card starts in canonical card space, is moved so the fan pivot sits
    ``grip`` below its bottom edge, then rotated about that pivot by its share of
    the spread. A single keystone applies to all of them, keeping the fan on one
    consistent image plane.
    """
    canvas = config.canvas
    scale = float(rng.uniform(*config.card_height_frac)) * canvas / CARD_HEIGHT
    grip = float(rng.uniform(*config.grip_frac)) * CARD_HEIGHT
    spread = float(rng.uniform(*config.spread_deg))
    tilt = float(rng.uniform(*config.tilt_deg))
    pivot_x = float(rng.uniform(*config.pivot_x_frac)) * canvas
    pivot_y = float(rng.uniform(*config.pivot_y_frac)) * canvas

    keystone = _keystone(canvas, config.perspective, rng)
    to_pivot = _translation(-CARD_WIDTH / 2.0, -(CARD_HEIGHT + grip))
    place = _translation(pivot_x, pivot_y)
    step = spread / max(count - 1, 1)

    placements: list[NDArray[np.float32]] = []
    for index in range(count):
        angle = tilt + (index - (count - 1) / 2.0) * step
        card = to_pivot
        if rng.random() < config.upside_down_prob:
            flip = _translation(CARD_WIDTH / 2.0, CARD_HEIGHT / 2.0) @ _rotation_scale(180.0, 1.0)
            card = card @ flip @ _translation(-CARD_WIDTH / 2.0, -CARD_HEIGHT / 2.0)
        placements.append(_homography(keystone @ place @ _rotation_scale(angle, scale) @ card))

    # Drawing left-to-right leaves each card's left strip clear, which is where
    # the top-left index sits. Reversing puts the leftmost card on top instead,
    # clearing the right strip and its bottom-right index - the other way a real
    # hand is stacked.
    if rng.random() < config.reverse_draw_prob:
        placements.reverse()
    return placements


def _hand_mask(
    canvas: int,
    placements: Sequence[NDArray[np.float32]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.uint8], Polygon | None]:
    """A skin-toned palm and fingers gripping the bottom of the fan."""
    layer = np.zeros((canvas, canvas, 4), dtype=np.uint8)
    bottoms = np.concatenate([_project(matrix, REF_CARD[[2, 3]]) for matrix in placements], axis=0)
    grip_x = float(np.median(bottoms[:, 0]))
    grip_y = float(np.median(bottoms[:, 1]))
    # Skin reads as B < G < R at a fairly fixed ratio. Sampling the channels
    # independently drifts into greys and purples, which would teach the model
    # that any blob occludes a card.
    tone = float(rng.uniform(95.0, 235.0))
    skin = (
        int(np.clip(tone * rng.uniform(0.58, 0.70), 0, 255)),
        int(np.clip(tone * rng.uniform(0.72, 0.84), 0, 255)),
        int(np.clip(tone, 0, 255)),
    )

    span = canvas * float(rng.uniform(0.14, 0.26))
    cv2.ellipse(
        layer,
        (int(grip_x), int(grip_y + span * 0.5)),
        (int(span), int(span * 0.85)),
        float(rng.uniform(-30, 30)),
        0,
        360,
        (*skin, 255),
        -1,
    )
    for _ in range(int(rng.integers(2, 5))):
        angle = math.radians(float(rng.uniform(-70, 70)))
        length = span * float(rng.uniform(0.7, 1.5))
        thickness = int(span * float(rng.uniform(0.16, 0.30)))
        start = (int(grip_x - math.sin(angle) * length), int(grip_y - math.cos(angle) * length))
        end = (int(grip_x + math.sin(angle) * length * 0.2), int(grip_y + span * 0.4))
        cv2.line(layer, start, end, (*skin, 255), thickness, lineType=cv2.LINE_AA)

    blurred: NDArray[np.uint8] = cv2.GaussianBlur(layer, (0, 0), sigmaX=1.5).astype(np.uint8)
    contours, _ = cv2.findContours(blurred[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [poly for contour in contours if (poly := _polygon(contour.reshape(-1, 2)))]
    return blurred, unary_union(polygons) if polygons else None


def _composite(base: NDArray[np.uint8], overlay: NDArray[np.uint8]) -> NDArray[np.uint8]:
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    blended = overlay[:, :, :3].astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)
    result: NDArray[np.uint8] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _drop_shadow(
    base: NDArray[np.uint8],
    card_layer: NDArray[np.uint8],
    strength: float,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Darken whatever sits under a card, so the cards behind it read as behind."""
    if strength <= 0.0:
        return base
    offset = _translation(float(rng.uniform(-6, 6)), float(rng.uniform(2, 10)))[:2]
    shifted = cv2.warpAffine(card_layer[:, :, 3], offset, (base.shape[1], base.shape[0]))
    mask = cv2.GaussianBlur(shifted, (0, 0), sigmaX=float(rng.uniform(3, 9)))
    factor = 1.0 - strength * (mask.astype(np.float32) / 255.0)[:, :, None]
    darkened: NDArray[np.uint8] = np.clip(base.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return darkened


def _visible_box(
    quad: NDArray[np.float32],
    occluder: Polygon | None,
    config: FanConfig,
) -> tuple[int, int, int, int] | None:
    """Bounding box of the part of an index quad that is still visible."""
    region = _polygon(quad)
    if region is None:
        return None
    frame = Polygon(
        [(0, 0), (config.canvas, 0), (config.canvas, config.canvas), (0, config.canvas)]
    )
    region = region.intersection(frame)
    if region.is_empty or region.area <= 0:
        return None

    full_area = region.area
    visible = region.difference(occluder) if occluder is not None else region
    if visible.is_empty or visible.area / full_area < config.min_visible:
        return None

    xmin, ymin, xmax, ymax = visible.bounds
    box = (math.floor(xmin), math.floor(ymin), math.ceil(xmax), math.ceil(ymax))
    if box[2] - box[0] < config.min_box_px or box[3] - box[1] < config.min_box_px:
        return None
    return box


def _photometric(
    image: NDArray[np.uint8],
    config: FanConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Blur, noise, and JPEG artefacts - what a phone camera adds for free."""
    output: NDArray[np.uint8] = image
    sigma = float(rng.uniform(*config.blur_sigma))
    if sigma > 0.05:
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=sigma).astype(np.uint8)

    noise = float(rng.uniform(*config.noise_sigma))
    if noise > 0.1:
        grain = rng.normal(0.0, noise, size=output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
    encoded, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if encoded:
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is not None:
            output = decoded.astype(np.uint8)
    return output


def _background(
    config: FanConfig,
    rng: np.random.Generator,
    backgrounds: Sequence[NDArray[np.uint8]] | None,
) -> NDArray[np.uint8]:
    """A photo background if one was supplied, otherwise a plausible table."""
    size = config.canvas
    if backgrounds:
        chosen = backgrounds[int(rng.integers(len(backgrounds)))]
        return cv2.resize(chosen, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)

    base = np.array(
        [rng.uniform(40, 140), rng.uniform(60, 165), rng.uniform(80, 195)], dtype=np.float32
    )
    gradient = np.linspace(0.75, 1.25, size, dtype=np.float32)[:, None, None]
    canvas = np.clip(base[None, None, :] * gradient, 0, 255)
    canvas = np.repeat(canvas, size, axis=1)
    grain = rng.normal(0.0, 9.0, size=(size, size, 3)).astype(np.float32)
    table = np.clip(canvas + grain, 0, 255).astype(np.uint8)
    smoothed: NDArray[np.uint8] = cv2.GaussianBlur(table, (0, 0), sigmaX=1.6).astype(np.uint8)
    return smoothed


def render_fan_scene(
    atlas: CardAtlas,
    *,
    rng: np.random.Generator,
    config: FanConfig | None = None,
    backgrounds: Sequence[NDArray[np.uint8]] | None = None,
) -> Scene:
    """Render one fanned hand and the corner-index labels that survive occlusion."""
    config = config or FanConfig()
    names = atlas.names
    count = int(rng.integers(config.cards_min, min(config.cards_max, len(names)) + 1))
    chosen = [names[index] for index in rng.choice(len(names), size=count, replace=False)]
    placements = _card_placements(count, config, rng)

    canvas = _background(config, rng, backgrounds)
    layers: list[NDArray[np.uint8]] = []
    card_polygons: list[Polygon | None] = []

    for name, matrix in zip(chosen, placements, strict=True):
        template = atlas.sample(name, rng).copy()
        gain = float(rng.uniform(*config.brightness))
        template[:, :, :3] = np.clip(template[:, :, :3].astype(np.float32) * gain, 0, 255).astype(
            np.uint8
        )
        layer = cv2.warpPerspective(
            template,
            matrix,
            (config.canvas, config.canvas),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        ).astype(np.uint8)
        canvas = _composite(_drop_shadow(canvas, layer, config.shadow_strength, rng), layer)
        layers.append(layer)
        card_polygons.append(_polygon(_project(matrix, REF_CARD)))

    hand_polygon: Polygon | None = None
    if rng.random() < config.hand_prob:
        hand_layer, hand_polygon = _hand_mask(config.canvas, placements, rng)
        canvas = _composite(canvas, hand_layer)

    labels: list[SceneLabel] = []
    for index, (name, matrix) in enumerate(zip(chosen, placements, strict=True)):
        covering = [poly for poly in card_polygons[index + 1 :] if poly is not None]
        if hand_polygon is not None:
            covering.append(hand_polygon)
        occluder = unary_union(covering) if covering else None
        class_name = normalise_card_name(name) or OTHER_CLASS

        for corner in (REF_CORNER_HL, REF_CORNER_LR):
            box = _visible_box(_project(matrix, corner), occluder, config)
            if box is None:
                continue
            xmin, ymin, xmax, ymax = box
            labels.append(
                SceneLabel(
                    class_name=class_name,
                    yolo=voc_to_yolo((config.canvas, config.canvas), (xmin, xmax, ymin, ymax)),
                    voc=box,
                )
            )

    return Scene(
        image=_photometric(canvas, config, rng),
        labels=labels,
        canvas_size=config.canvas,
    )


def load_backgrounds(directory: Path | None, *, limit: int = 400) -> list[NDArray[np.uint8]]:
    """Read up to ``limit`` background photos from a directory, if given."""
    if directory is None:
        return []
    paths = [p for p in sorted(Path(directory).rglob("*")) if p.suffix.lower() in _IMAGE_SUFFIXES]
    images: list[NDArray[np.uint8]] = []
    for path in paths[:limit]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image.astype(np.uint8))
    return images


__all__ = [
    "OTHER_CLASS",
    "CardAtlas",
    "FanConfig",
    "load_backgrounds",
    "render_fan_scene",
]
