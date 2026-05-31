"""Tests for the synthetic-pipeline modules ported from the legacy notebook."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from hakim_vision.synthetic import (
    CARD_HEIGHT,
    CARD_WIDTH,
    ExtractedCard,
    extract_card,
    find_corner_hull,
    hull_to_points,
    points_to_bbox,
    points_to_polygon,
    variance_of_laplacian,
)


def _make_card_on_background(
    bg_shape: tuple[int, int] = (480, 640),
    card_color: tuple[int, int, int] = (240, 240, 240),
    bg_color: tuple[int, int, int] = (20, 60, 20),
) -> NDArray[np.uint8]:
    """Render a sharp-edged white-ish card on a dark uniform background."""
    h, w = bg_shape
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    # Draw a high-contrast filled rectangle "card" near the centre.
    top_left = (w // 4, h // 4)
    bottom_right = (3 * w // 4, 3 * h // 4)
    cv2.rectangle(img, top_left, bottom_right, card_color, thickness=-1)
    # Add a thin border so Canny has strong edges.
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), thickness=2)
    return img


# ---------------------------------------------------------------------------
# variance_of_laplacian
# ---------------------------------------------------------------------------

def test_focus_higher_for_sharp_than_blurred() -> None:
    img = _make_card_on_background()
    sharp = variance_of_laplacian(img)
    blurred_img = cv2.GaussianBlur(img, (25, 25), 0)
    blurred = variance_of_laplacian(blurred_img)
    assert sharp > blurred
    assert blurred >= 0.0


def test_focus_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2-D or 3-D"):
        variance_of_laplacian(np.zeros(7, dtype=np.uint8))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_card
# ---------------------------------------------------------------------------

def test_extract_card_on_blurred_image_returns_none() -> None:
    img = _make_card_on_background()
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    # min_focus chosen well above what a heavily blurred uniform card produces.
    result = extract_card(blurred, min_focus=1e9)
    assert result is None


def test_extract_card_returns_canonical_size() -> None:
    img = _make_card_on_background()
    result = extract_card(img, min_focus=0.0)
    assert isinstance(result, ExtractedCard)
    assert result.image.shape == (CARD_HEIGHT, CARD_WIDTH, 4)
    assert result.image.dtype == np.uint8
    assert 0.0 <= result.rectangularity <= 1.0 + 1e-6
    assert result.focus >= 0.0


def test_extract_card_rejects_non_bgr_input() -> None:
    with pytest.raises(ValueError, match="BGR uint8"):
        extract_card(np.zeros((10, 10), dtype=np.uint8))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# hull / bbox helpers
# ---------------------------------------------------------------------------

def test_hull_to_points_translates() -> None:
    hull = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.intp)
    pts = hull_to_points(hull, offset_x=5, offset_y=7)
    assert pts.shape == (4, 2)
    np.testing.assert_array_equal(pts[0], [5.0, 7.0])
    np.testing.assert_array_equal(pts[2], [15.0, 17.0])


def test_points_to_polygon_requires_three_points() -> None:
    with pytest.raises(ValueError, match="N >= 3"):
        points_to_polygon(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))


def test_points_to_polygon_area_matches() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    poly = points_to_polygon(square)
    assert poly.area == pytest.approx(100.0)


def test_points_to_bbox_clips_and_expands() -> None:
    pts = np.array([[5, 5], [50, 60]], dtype=np.float32)
    bbox = points_to_bbox(pts, image_size=(100, 100), expand=3)
    assert bbox == (2, 2, 53, 63)


def test_points_to_bbox_degenerate_returns_none() -> None:
    pts = np.empty((0, 2), dtype=np.float32)
    assert points_to_bbox(pts, image_size=(100, 100)) is None


def test_points_to_bbox_clipped_to_zero_returns_none() -> None:
    # All points off the left edge with expand small enough to stay clipped.
    pts = np.array([[-10, 5], [-5, 6]], dtype=np.float32)
    assert points_to_bbox(pts, image_size=(100, 100), expand=1) is None


# ---------------------------------------------------------------------------
# find_corner_hull (smoke; uses synthetic card image)
# ---------------------------------------------------------------------------

def test_find_corner_hull_returns_none_on_blank_card() -> None:
    # A blank white card has no internal edges → no hull.
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), 255, dtype=np.uint8)
    assert find_corner_hull(card) is None
