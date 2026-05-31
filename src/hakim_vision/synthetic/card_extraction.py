"""Card extraction from a photo or video frame.

Ported and modernized from `extract_card` in the legacy notebook. Key fixes
relative to the original:

- `cv2.findContours` API: in OpenCV 4.x it returns `(contours, hierarchy)`, not
  `(_, contours, hierarchy)` as in OpenCV 3.x. The original code broke silently.
- `np.int0` is removed in NumPy 2.x — replaced with `np.intp`.
- `np.int` is removed — replaced with builtin `int`.
- Side-effecting `cv2.imshow` / `cv2.imwrite` calls removed from the core
  function so it is unit-testable; callers may save / display the returned
  arrays themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    REF_CARD,
    REF_CARD_ROT,
)
from hakim_vision.synthetic.focus import variance_of_laplacian

# Area-fit tolerance: the largest contour's area divided by the minimum-area
# bounding rectangle's area must exceed this for the contour to be treated as
# a card. The legacy notebook used 0.95.
_RECTANGULARITY_THRESHOLD: float = 0.95


@dataclass(frozen=True)
class ExtractedCard:
    """A successfully extracted card, with diagnostic metadata."""

    image: NDArray[np.uint8]
    """The warped BGRA card image of shape (CARD_HEIGHT, CARD_WIDTH, 4)."""

    focus: float
    """Variance-of-Laplacian focus score of the source frame."""

    rectangularity: float
    """Ratio of contour area to its min-area bounding-box area (0..1)."""


def _alpha_mask(border: int) -> NDArray[np.uint8]:
    """Build the soft border alpha mask used to feather card edges."""
    mask = np.full((CARD_HEIGHT, CARD_WIDTH), 255, dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (CARD_WIDTH - 1, CARD_HEIGHT - 1), 0, border)
    # Diagonal clips on the two right-hand corners, matching the legacy mask.
    cv2.line(mask, (CARD_WIDTH - border * 3, 0), (CARD_WIDTH, border * 3), 0, border)
    cv2.line(
        mask,
        (CARD_WIDTH - border * 3, CARD_HEIGHT),
        (CARD_WIDTH, CARD_HEIGHT - border * 3),
        0,
        border,
    )
    return mask


_ALPHA_MASK: NDArray[np.uint8] = _alpha_mask(border=2)


def extract_card(
    frame: NDArray[np.uint8],
    *,
    min_focus: float = 120.0,
    rectangularity_threshold: float = _RECTANGULARITY_THRESHOLD,
) -> ExtractedCard | None:
    """Attempt to extract a single playing card from a BGR frame.

    Pipeline:

    1. Reject too-blurry frames via Laplacian variance.
    2. Bilateral-filter + Canny edges.
    3. Take the largest external contour.
    4. Verify that contour fills ≥ `rectangularity_threshold` of its
       minimum-area bounding rectangle (rules out non-rectangular junk).
    5. Perspective-warp the card to a canonical `(CARD_WIDTH, CARD_HEIGHT)`
       canvas and attach a feathered alpha channel.

    Args:
        frame: BGR image as a NumPy array of dtype uint8.
        min_focus: Minimum Laplacian variance to accept the frame.
        rectangularity_threshold: Minimum contour-to-bbox area ratio.

    Returns:
        An `ExtractedCard` on success, or `None` if the frame is too blurry
        or no rectangular card-like contour was found.

    Raises:
        ValueError: if `frame` is not a BGR uint8 image.
    """
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
    ):
        raise ValueError("frame must be a BGR uint8 image of shape (H, W, 3)")

    focus = variance_of_laplacian(frame)
    if focus < min_focus:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)

    contours, _hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.intp)

    area_cnt = float(cv2.contourArea(cnt))
    area_box = float(cv2.contourArea(box))
    if area_box <= 0:
        return None

    rectangularity = area_cnt / area_box
    if rectangularity < rectangularity_threshold:
        return None

    (_, _), (w_r, h_r), _theta = rect
    ref = REF_CARD if w_r > h_r else REF_CARD_ROT
    perspective = cv2.getPerspectiveTransform(box.astype(np.float32), ref)
    warped = cv2.warpPerspective(frame, perspective, (CARD_WIDTH, CARD_HEIGHT))
    warped_bgra = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)

    # Re-project the source contour into the warped frame and use it to build
    # the alpha channel — this gives a clean cut-out even for slightly
    # imperfect rectangles.
    cnt_warped = cv2.perspectiveTransform(
        cnt.reshape(1, -1, 2).astype(np.float32), perspective
    ).astype(np.intp)

    alpha = np.zeros(warped_bgra.shape[:2], dtype=np.uint8)
    cv2.drawContours(alpha, cnt_warped, 0, 255, -1)
    alpha = cv2.bitwise_and(alpha, _ALPHA_MASK)
    warped_bgra[:, :, 3] = alpha

    return ExtractedCard(image=warped_bgra, focus=focus, rectangularity=rectangularity)


__all__ = ["ExtractedCard", "extract_card"]
