"""End-to-end scene-rendering smoke tests.

These fulfil the §3.2 plan item: a smoke test that generates ≥ 4 full scenes
and asserts label validity. Card images and backgrounds are synthetic so the
test runs in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hakim_vision.synthetic import (
    CARD_HEIGHT,
    CARD_WIDTH,
    REF_CORNER_HL,
    REF_CORNER_LR,
    Backgrounds,
    Cards,
    Scene,
    pack_backgrounds,
    pack_cards,
    random_affine_card,
    render_random_scene,
    write_yolo_label,
)


def _make_fixture_assets(tmp_path: Path) -> tuple[Backgrounds, Cards]:
    """Create tiny synthetic backgrounds + cards shards for testing."""
    bg_src = tmp_path / "bg_src"
    bg_src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        arr = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_src / f"{i}.png"), arr)

    cards_root = tmp_path / "cards_src"
    for cls in ("Ah", "7s", "Kd"):
        d = cards_root / cls
        d.mkdir(parents=True)
        for i in range(2):
            bgra = np.full((CARD_HEIGHT, CARD_WIDTH, 4), 255, dtype=np.uint8)
            bgra[..., :3] = rng.integers(0, 255, size=(CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{i}.png"), bgra)

    bg_shards = pack_backgrounds(bg_src, tmp_path / "bg_shards", shard_size=10)
    card_shards = pack_cards(
        cards_root, tmp_path / "card_shards", shard_size=10, extract_hulls=False
    )
    return (
        Backgrounds(bg_shards, rng=np.random.default_rng(1)),
        Cards(card_shards, rng=np.random.default_rng(2)),
    )


# ---------------------------------------------------------------------------
# random_affine_card
# ---------------------------------------------------------------------------


def test_random_affine_card_shapes() -> None:
    rng = np.random.default_rng(0)
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 4), 255, dtype=np.uint8)
    placed = random_affine_card(card, REF_CORNER_HL, REF_CORNER_LR, rng=rng, canvas_size=512)
    assert placed.image.shape == (512, 512, 4)
    assert placed.card_corners.shape == (4, 2)
    assert placed.hull_hl_points.shape[1] == 2
    assert placed.hull_lr_points.shape[1] == 2


def test_random_affine_card_is_deterministic_under_seed() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 4), 200, dtype=np.uint8)
    a = random_affine_card(card, REF_CORNER_HL, REF_CORNER_LR, rng=np.random.default_rng(7))
    b = random_affine_card(card, REF_CORNER_HL, REF_CORNER_LR, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_allclose(a.card_corners, b.card_corners)


# ---------------------------------------------------------------------------
# End-to-end scene rendering — the §3.2 smoke gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cards", [2, 3])
def test_render_n_scenes_yields_valid_labels(tmp_path: Path, n_cards: int) -> None:
    backgrounds, cards = _make_fixture_assets(tmp_path)
    rng = np.random.default_rng(42)
    scenes = [
        render_random_scene(cards, backgrounds, rng=rng, n_cards=n_cards, canvas_size=512)
        for _ in range(4)
    ]
    assert len(scenes) == 4
    for scene in scenes:
        assert isinstance(scene, Scene)
        assert scene.image.shape == (512, 512, 3)
        assert scene.image.dtype == np.uint8
        # At least one of the 4 scenes should produce labels — fixture geometry
        # may legitimately drop occluded corners for some seeds.
    assert any(len(s.labels) > 0 for s in scenes)
    # Every emitted label has valid normalized YOLO coordinates.
    for scene in scenes:
        for label in scene.labels:
            assert 0.0 <= label.yolo.cx <= 1.0
            assert 0.0 <= label.yolo.cy <= 1.0
            assert 0.0 < label.yolo.w <= 1.0
            assert 0.0 < label.yolo.h <= 1.0
            xmin, ymin, xmax, ymax = label.voc
            assert 0 <= xmin < xmax <= 512
            assert 0 <= ymin < ymax <= 512


def test_write_yolo_label_format(tmp_path: Path) -> None:
    backgrounds, cards = _make_fixture_assets(tmp_path)
    rng = np.random.default_rng(123)
    scene = render_random_scene(cards, backgrounds, rng=rng, n_cards=2, canvas_size=256)
    class_to_id = {name: i for i, name in enumerate(cards.class_names)}
    payload = write_yolo_label(scene, class_to_id)
    if scene.labels:
        for line in payload.strip().splitlines():
            parts = line.split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            assert cls_id in class_to_id.values()
            for token in parts[1:]:
                v = float(token)
                assert 0.0 <= v <= 1.0
    else:
        assert payload == ""
