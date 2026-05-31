"""Corner-symbol hull detection and keypoint utilities.

Ported from `findHull`, `hull_to_kps`, `kps_to_polygon`, `kps_to_BB` in the
legacy notebook. Two substantive changes from the original:

- We do not depend on `imgaug` (unmaintained). Keypoints are plain `(N, 2)`
  float arrays; polygons are `shapely.geometry.Polygon`; bounding boxes are
  a `(xmin, ymin, xmax, ymax)` tuple.
- The OpenCV 4 `findContours` signature is used (`(contours, hierarchy)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from hakim_vision.synthetic.constants import REF_CORNER_HL


@dataclass(frozen=True)
class HullDetection:
    """Result of corner-symbol hull detection."""

    hull: NDArray[np.intp]
    """Convex hull as a `(K, 1, 2)` contour in image coordinates."""

    ok: bool
    """Whether the hull area falls inside the configured plausible range."""

    area: float
    """Hull area in pixels."""


def find_corner_hull(
    card_image: NDArray[np.uint8],
    corner: NDArray[np.float32] = REF_CORNER_HL,
    *,
    min_contour_area: float = 30.0,
    min_solidity: float = 0.3,
    min_hull_area: float = 940.0,
    max_hull_area: float = 2120.0,
    center_x_tolerance: float = 0.3,
    center_y_tolerance: float = 0.4,
) -> HullDetection | None:
    """Find the convex hull around the rank/suit symbol in a card corner.

    Args:
        card_image: BGR uint8 image of a card warped to the reference canvas.
        corner: 4×2 float32 array delimiting the rectangular corner zone
            (`REF_CORNER_HL` for the top-left, `REF_CORNER_LR` for the
            bottom-right corner).
        min_contour_area: Reject small noise contours below this area.
        min_solidity: Reject contours whose `area / convex_hull_area` is below
            this — drops thin, jagged contours.
        min_hull_area: Lower bound on the merged hull area; outside this range
            the result is marked `ok=False`.
        max_hull_area: Upper bound on the merged hull area.
        center_x_tolerance: Keep only contours whose centre of gravity is within
            `±tol*width` of the corner-zone centre.
        center_y_tolerance: Same as `center_x_tolerance` but for height.

    Returns:
        A `HullDetection` if at least one contour passed the filters;
        otherwise `None`.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    corner_i = corner.astype(np.intp)

    x1, y1 = int(corner_i[0][0]), int(corner_i[0][1])
    x2, y2 = int(corner_i[2][0]), int(corner_i[2][1])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    zone = card_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 200)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    concat: NDArray[np.intp] | None = None
    for c in contours:
        area = float(cv2.contourArea(c))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        solidity = area / hull_area

        moments = cv2.moments(c)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        center_ok = (
            abs(w / 2 - cx) < w * center_x_tolerance
            and abs(h / 2 - cy) < h * center_y_tolerance
        )
        if area >= min_contour_area and solidity > min_solidity and center_ok:
            concat = c if concat is None else np.concatenate((concat, c))

    if concat is None:
        return None

    hull = cv2.convexHull(concat)
    hull_area = float(cv2.contourArea(hull))
    hull_in_image = (hull + corner_i[0]).astype(np.intp)
    ok = min_hull_area <= hull_area <= max_hull_area
    return HullDetection(hull=hull_in_image, ok=ok, area=hull_area)


def hull_to_points(
    hull: NDArray[np.intp],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> NDArray[np.float32]:
    """Flatten an OpenCV contour `(K, 1, 2)` into an `(K, 2)` float array.

    Optionally translates points by `(offset_x, offset_y)` — used when placing
    a card hull into a larger scene canvas.
    """
    pts = hull.reshape(-1, 2).astype(np.float32)
    if offset_x or offset_y:
        pts = pts + np.array([offset_x, offset_y], dtype=np.float32)
    return pts


def points_to_polygon(points: NDArray[np.float32]) -> Polygon:
    """Convert an `(N, 2)` point array to a `shapely` polygon."""
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(
            f"need (N, 2) with N >= 3 to form a polygon, got shape {points.shape}"
        )
    return Polygon(points.tolist())


def points_to_bbox(
    points: NDArray[np.float32],
    image_size: tuple[int, int],
    *,
    expand: int = 3,
) -> tuple[int, int, int, int] | None:
    """Compute an integer axis-aligned bbox clipped to the image.

    Args:
        points: `(N, 2)` array of keypoints in image coordinates.
        image_size: `(width, height)` of the parent image, for clipping.
        expand: Pixels of padding to apply on every side.

    Returns:
        `(xmin, ymin, xmax, ymax)` integer tuple, or `None` if the box is
        degenerate after clipping.
    """
    if points.size == 0:
        return None
    width, height = image_size
    xs, ys = points[:, 0], points[:, 1]
    xmin = max(0, int(xs.min() - expand))
    xmax = min(width, int(xs.max() + expand))
    ymin = max(0, int(ys.min() - expand))
    ymax = min(height, int(ys.max() + expand))
    if xmin >= xmax or ymin >= ymax:
        return None
    return xmin, ymin, xmax, ymax


__all__ = [
    "HullDetection",
    "find_corner_hull",
    "hull_to_points",
    "points_to_bbox",
    "points_to_polygon",
]
