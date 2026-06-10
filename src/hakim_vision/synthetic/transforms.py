"""Per-card affine augmentation, OpenCV-only.

Applies a uniform random affine (rotation, scale, translation) to a BGRA card
layer and its corner-hull keypoints using ``cv2.warpAffine`` plus plain numpy
point arrays. Fully deterministic given an injected RNG. Photometric / elastic
effects can be layered on top in a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    SCENE_SIZE,
)


@dataclass(frozen=True)
class AugmentRange:
    """Uniform ranges for a single random affine transform of one card."""

    rotation_deg: tuple[float, float] = (-30.0, 30.0)
    scale: tuple[float, float] = (0.5, 0.8)
    translate_frac: tuple[float, float] = (-0.15, 0.15)


#: Shared default augmentation range; ``AugmentRange`` is frozen, so a single
#: module-level instance is safe to reuse as a default argument value.
_DEFAULT_AUGMENT = AugmentRange()


@dataclass(frozen=True)
class PlacedCard:
    """A card laid out on the scene canvas, with tracked keypoints.

    Attributes:
        image: BGRA scene-sized image, mostly transparent except where the card
            is rendered after the affine warp.
        card_corners: ``(4, 2)`` array — the four card outline points after
            transform, in scene coordinates.
        hull_hl_points: ``(K_hl, 2)`` array — top-left corner-symbol hull
            points after transform, in scene coordinates.
        hull_lr_points: ``(K_lr, 2)`` array — bottom-right corner-symbol hull
            points after transform, in scene coordinates.
    """

    image: NDArray[np.uint8]
    card_corners: NDArray[np.float32]
    hull_hl_points: NDArray[np.float32]
    hull_lr_points: NDArray[np.float32]


def _place_card_on_canvas(
    card_bgra: NDArray[np.uint8],
    canvas_size: int,
) -> tuple[NDArray[np.uint8], int, int]:
    """Center the canonical card on a transparent ``canvas_size`` canvas."""
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    dx = (canvas_size - CARD_WIDTH) // 2
    dy = (canvas_size - CARD_HEIGHT) // 2
    canvas[dy : dy + CARD_HEIGHT, dx : dx + CARD_WIDTH, :] = card_bgra
    return canvas, dx, dy


def _apply_affine_points(
    matrix: NDArray[np.float32],
    points: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply a ``2x3`` affine matrix to ``(N, 2)`` points."""
    if points.size == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([points.astype(np.float32), ones], axis=1)  # (N, 3)
    transformed: NDArray[np.float32] = (matrix @ homog.T).T.astype(np.float32)
    return transformed


def random_affine_card(
    card_bgra: NDArray[np.uint8],
    hull_hl: NDArray[np.intp] | NDArray[np.float32],
    hull_lr: NDArray[np.intp] | NDArray[np.float32],
    *,
    rng: np.random.Generator,
    canvas_size: int = SCENE_SIZE,
    aug: AugmentRange = _DEFAULT_AUGMENT,
) -> PlacedCard:
    """Place the card on a scene-sized canvas and apply a random affine.

    Args:
        card_bgra: ``(CARD_HEIGHT, CARD_WIDTH, 4)`` BGRA card.
        hull_hl: Top-left corner-symbol hull, ``(K_hl, 1, 2)`` or ``(K_hl, 2)``.
        hull_lr: Bottom-right corner-symbol hull, same shape conventions.
        rng: ``numpy.random.Generator`` for reproducibility.
        canvas_size: Target scene canvas (square).
        aug: Uniform ranges for rotation / scale / translation.

    Returns:
        A ``PlacedCard`` with the warped image and the transformed keypoints.
    """
    if card_bgra.shape != (CARD_HEIGHT, CARD_WIDTH, 4):
        raise ValueError(f"card must be {(CARD_HEIGHT, CARD_WIDTH, 4)}, got {card_bgra.shape}")

    # Lay the card out at full size on a placement canvas at least as large as
    # the card itself, so a requested output smaller than the card (e.g. a 256px
    # preview) never truncates the source. The affine then warps this placement
    # canvas into the requested ``canvas_size`` output.
    place_size = max(canvas_size, CARD_HEIGHT, CARD_WIDTH)
    canvas, dx, dy = _place_card_on_canvas(card_bgra, place_size)

    # Card outline keypoints in placement-canvas coordinates.
    card_corners = np.array(
        [
            [dx, dy],
            [dx + CARD_WIDTH, dy],
            [dx + CARD_WIDTH, dy + CARD_HEIGHT],
            [dx, dy + CARD_HEIGHT],
        ],
        dtype=np.float32,
    )
    # Hull points, lifted into placement-canvas coordinates by the same offset.
    offset = np.array([dx, dy], dtype=np.float32)
    hull_hl_pts = np.asarray(hull_hl).reshape(-1, 2).astype(np.float32) + offset
    hull_lr_pts = np.asarray(hull_lr).reshape(-1, 2).astype(np.float32) + offset

    # Random affine parameters. Rotation/scale pivot on the card centre (the
    # placement-canvas centre); a recentre term maps that pivot to the output
    # centre, so the card stays centred when the placement canvas is larger than
    # the requested output. When ``canvas_size >= card`` this reduces exactly to
    # a plain rotate-scale-translate about the output centre.
    angle = float(rng.uniform(*aug.rotation_deg))
    scale = float(rng.uniform(*aug.scale))
    tx_frac = float(rng.uniform(*aug.translate_frac))
    ty_frac = float(rng.uniform(*aug.translate_frac))
    centre = (place_size / 2.0, place_size / 2.0)
    recentre = (canvas_size - place_size) / 2.0

    matrix = cv2.getRotationMatrix2D(centre, angle, scale).astype(np.float32)
    matrix[0, 2] += recentre + tx_frac * canvas_size
    matrix[1, 2] += recentre + ty_frac * canvas_size

    warped = cv2.warpAffine(
        canvas,
        matrix,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return PlacedCard(
        image=warped.astype(np.uint8, copy=False),
        card_corners=_apply_affine_points(matrix, card_corners),
        hull_hl_points=_apply_affine_points(matrix, hull_hl_pts),
        hull_lr_points=_apply_affine_points(matrix, hull_lr_pts),
    )


__all__ = ["AugmentRange", "PlacedCard", "random_affine_card"]
