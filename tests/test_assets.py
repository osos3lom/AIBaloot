"""Round-trip tests for pack -> Backgrounds/Cards loaders."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hakim_vision.synthetic import (
    CARD_HEIGHT,
    CARD_WIDTH,
    Backgrounds,
    CardSample,
    Cards,
    pack_backgrounds,
    pack_cards,
)


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------

def _write_random_texture(path: Path, *, size: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def test_pack_backgrounds_and_load(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        _write_random_texture(src / f"tex_{i}.png", seed=i)

    out = tmp_path / "shards"
    shards = pack_backgrounds(src, out, shard_size=2, shard_prefix="bg")
    assert len(shards) == 3  # 5 images, shard_size=2 → 3 shards
    for shard in shards:
        assert shard.exists()

    backgrounds = Backgrounds(shards, rng=np.random.default_rng(0))
    assert len(backgrounds) == 5
    sample = backgrounds.sample(size=128)
    assert sample.shape == (128, 128, 3)
    assert sample.dtype == np.uint8


def test_backgrounds_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.tar"
    import tarfile

    with tarfile.open(empty, mode="w"):
        pass
    with pytest.raises(ValueError, match="0 images"):
        Backgrounds([empty])


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def _write_canonical_card(path: Path, *, seed: int = 0) -> None:
    """Write a canonical-sized BGRA "card" with a noisy interior."""
    rng = np.random.default_rng(seed)
    bgra = np.full((CARD_HEIGHT, CARD_WIDTH, 4), 255, dtype=np.uint8)
    noise = rng.integers(0, 255, size=(CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8)
    bgra[..., :3] = noise
    cv2.imwrite(str(path), bgra)


def test_pack_cards_no_hulls_round_trip(tmp_path: Path) -> None:
    cards_root = tmp_path / "cards"
    (cards_root / "Ah").mkdir(parents=True)
    (cards_root / "6s").mkdir(parents=True)
    for i in range(3):
        _write_canonical_card(cards_root / "Ah" / f"{i}.png", seed=i)
    for i in range(2):
        _write_canonical_card(cards_root / "6s" / f"{i}.png", seed=10 + i)

    out = tmp_path / "shards"
    shards = pack_cards(
        cards_root, out, shard_size=2, shard_prefix="cards", extract_hulls=False
    )
    assert len(shards) >= 1

    cards = Cards(shards, rng=np.random.default_rng(0))
    assert set(cards.class_names) == {"Ah", "6s"}
    assert cards.counts_by_class["Ah"] == 3
    assert cards.counts_by_class["6s"] == 2
    assert len(cards) == 5

    sample = cards.sample("Ah")
    assert isinstance(sample, CardSample)
    assert sample.name == "Ah"
    assert sample.image.shape[:2] == (CARD_HEIGHT, CARD_WIDTH)
    assert sample.hull_hl.ndim == 3 and sample.hull_hl.shape[2] == 2
    assert sample.hull_lr.ndim == 3 and sample.hull_lr.shape[2] == 2


def test_cards_unknown_class_raises(tmp_path: Path) -> None:
    cards_root = tmp_path / "cards"
    (cards_root / "Ah").mkdir(parents=True)
    _write_canonical_card(cards_root / "Ah" / "0.png", seed=0)
    shards = pack_cards(cards_root, tmp_path / "shards", extract_hulls=False)

    cards = Cards(shards)
    with pytest.raises(KeyError, match="unknown card class"):
        cards.sample("ZZ")


def test_cards_rejects_wrong_sized_images(tmp_path: Path) -> None:
    cards_root = tmp_path / "cards"
    (cards_root / "Ah").mkdir(parents=True)
    # Write a too-small image which must be skipped by the packer.
    small = np.zeros((10, 10, 4), dtype=np.uint8)
    cv2.imwrite(str(cards_root / "Ah" / "tiny.png"), small)
    out = tmp_path / "shards"
    shards = pack_cards(cards_root, out, extract_hulls=False)
    # Either no shard written, or the shard is empty.
    if shards:
        with pytest.raises(ValueError, match="0 classes"):
            Cards(shards)


def test_backgrounds_deterministic_under_seed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        _write_random_texture(src / f"{i}.png", seed=i)
    shards = pack_backgrounds(src, tmp_path / "shards", shard_size=10)

    bg_a = Backgrounds(shards, rng=np.random.default_rng(42))
    bg_b = Backgrounds(shards, rng=np.random.default_rng(42))
    a = bg_a.sample(size=32)
    b = bg_b.sample(size=32)
    np.testing.assert_array_equal(a, b)
