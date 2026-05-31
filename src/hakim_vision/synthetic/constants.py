"""Reference card geometry constants, ported from the legacy notebook.

The original code measured a physical card by hand (millimetres, then scaled by
``zoom=4`` to a pixel canvas). We preserve those exact values so generated
datasets remain compatible with the prior project.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- Hand-measured card geometry (millimetres before zoom). --------------------
_ZOOM: int = 4
_CARD_W_MM: int = 57
_CARD_H_MM: int = 87

#: Card width in pixels on the reference canvas.
CARD_WIDTH: int = _CARD_W_MM * _ZOOM  # 228
#: Card height in pixels on the reference canvas.
CARD_HEIGHT: int = _CARD_H_MM * _ZOOM  # 348

# --- Corner indicator (the small rank/suit symbol in the corner). --------------
# Inner rectangle (top-left of the card) where the rank/suit symbol sits.
CORNER_X_MIN: int = int(2 * _ZOOM)
CORNER_X_MAX: int = int(10.5 * _ZOOM)
CORNER_Y_MIN: int = int(2.5 * _ZOOM)
CORNER_Y_MAX: int = int(23 * _ZOOM)

# --- Reference quads used by `cv2.getPerspectiveTransform`. --------------------
#: Reference card outline, clockwise from top-left.
REF_CARD: NDArray[np.float32] = np.array(
    [[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]],
    dtype=np.float32,
)
#: Same outline rotated 90° (used when the detected card is taller than wide).
REF_CARD_ROT: NDArray[np.float32] = np.array(
    [[CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT], [0, 0]],
    dtype=np.float32,
)

#: Top-left corner indicator rectangle (used to locate rank/suit symbols).
REF_CORNER_HL: NDArray[np.float32] = np.array(
    [
        [CORNER_X_MIN, CORNER_Y_MIN],
        [CORNER_X_MAX, CORNER_Y_MIN],
        [CORNER_X_MAX, CORNER_Y_MAX],
        [CORNER_X_MIN, CORNER_Y_MAX],
    ],
    dtype=np.float32,
)

#: Bottom-right corner indicator rectangle (180° mirror of REF_CORNER_HL).
REF_CORNER_LR: NDArray[np.float32] = np.array(
    [
        [CARD_WIDTH - CORNER_X_MAX, CARD_HEIGHT - CORNER_Y_MAX],
        [CARD_WIDTH - CORNER_X_MIN, CARD_HEIGHT - CORNER_Y_MAX],
        [CARD_WIDTH - CORNER_X_MIN, CARD_HEIGHT - CORNER_Y_MIN],
        [CARD_WIDTH - CORNER_X_MAX, CARD_HEIGHT - CORNER_Y_MIN],
    ],
    dtype=np.float32,
)

#: Default output scene size (square canvas, pixels).
SCENE_SIZE: int = 720


__all__ = [
    "CARD_HEIGHT",
    "CARD_WIDTH",
    "CORNER_X_MAX",
    "CORNER_X_MIN",
    "CORNER_Y_MAX",
    "CORNER_Y_MIN",
    "REF_CARD",
    "REF_CARD_ROT",
    "REF_CORNER_HL",
    "REF_CORNER_LR",
    "SCENE_SIZE",
]
