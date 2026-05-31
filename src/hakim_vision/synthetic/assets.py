"""Background-texture and card-image asset loaders.

Ported from the `Backgrounds` and `Cards` classes in the legacy notebook, with
three structural changes:

1. **No pickle.** Assets live in WebDataset tar shards
   (`backgrounds-{000..NNN}.tar`, `cards-{000..NNN}.tar`). Pickle loading was an
   RCE risk; this also makes the artifacts portable across Python versions and
   streamable from object storage (R2 / S3 / HF Hub).
2. **Injected RNG.** Both loaders accept a `numpy.random.Generator`, so dataset
   generation is reproducible from a single seed. The original code used the
   module-level `random` global, which makes runs non-deterministic across
   threads.
3. **Lazy + indexed.** Backgrounds are loaded eagerly into memory (they're
   small — ~5,640 DTD textures); cards keep an on-disk index and decode on
   demand so the dataset fits in RAM even with thousands of source photos per
   class.

A card shard sample is expected to contain:

    {key}.cls         : utf-8 card name, e.g. "Ah", "2s", "10d"
    {key}.png         : BGRA card image of shape (CARD_HEIGHT, CARD_WIDTH, 4)
    {key}.hull_hl.npy : (K, 1, 2) int32 contour of the top-left corner hull
    {key}.hull_lr.npy : (K, 1, 2) int32 contour of the bottom-right corner hull

A background shard sample contains just `{key}.png` (any solid 3-channel image;
the loader will resize to the requested scene canvas).
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CardSample:
    """A single source card image with the two corner-symbol hulls."""

    name: str
    """Card class name, e.g. ``"Ah"``."""

    image: NDArray[np.uint8]
    """BGRA card image."""

    hull_hl: NDArray[np.intp]
    """Top-left corner hull as a `(K, 1, 2)` contour."""

    hull_lr: NDArray[np.intp]
    """Bottom-right corner hull as a `(K, 1, 2)` contour."""


def _iter_shard_samples(shard_path: Path) -> Iterator[dict[str, bytes]]:
    """Yield WebDataset samples (`{ext: bytes, ...}`) from a tar shard.

    A sample is the group of files sharing the same stem (everything before the
    first `.`). Files are expected in sorted order within a shard, which `tar`
    guarantees if it was built by the `pack` CLI.
    """
    current_key: str | None = None
    current: dict[str, bytes] = {}
    with tarfile.open(shard_path, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if "." not in name:
                continue
            key, _, ext = name.partition(".")
            if current_key is None:
                current_key = key
            if key != current_key:
                if current:
                    yield current
                current = {}
                current_key = key
            f = tar.extractfile(member)
            if f is not None:
                current[ext] = f.read()
        if current:
            yield current


def _decode_image(data: bytes) -> NDArray[np.uint8]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("failed to decode image payload")
    return img


def _decode_npy(data: bytes) -> NDArray[np.intp]:
    return np.load(io.BytesIO(data)).astype(np.intp)


class Backgrounds:
    """Loader for background textures.

    Loads all images eagerly into a list and serves random ones via the
    supplied RNG. Images are resized to a square canvas on demand.
    """

    def __init__(
        self,
        shards: Iterable[Path] | Path,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(shards, Path):
            shards = [shards]
        rng = rng if rng is not None else np.random.default_rng()
        images: list[NDArray[np.uint8]] = []
        for shard in shards:
            for sample in _iter_shard_samples(shard):
                for ext, payload in sample.items():
                    if ext in {"png", "jpg", "jpeg", "webp"}:
                        images.append(_decode_image(payload))
                        break
        if not images:
            raise ValueError("Backgrounds loader found 0 images across shards")
        self._images: list[NDArray[np.uint8]] = images
        self._rng = rng

    def __len__(self) -> int:
        return len(self._images)

    def sample(self, size: int) -> NDArray[np.uint8]:
        """Return a random background resized to a `(size, size, 3)` canvas."""
        idx = int(self._rng.integers(0, len(self._images)))
        bg = self._images[idx]
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        elif bg.shape[2] == 4:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR)
        return cv2.resize(bg, (size, size), interpolation=cv2.INTER_AREA)


class Cards:
    """Loader for source card images and their corner hulls.

    Builds an index of `(card_name -> list[(shard_path, key)])` and decodes
    samples on demand. Per-class counts are exposed via `counts_by_class`.
    """

    def __init__(
        self,
        shards: Iterable[Path] | Path,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(shards, Path):
            shards = [shards]
        shards = list(shards)
        rng = rng if rng is not None else np.random.default_rng()

        index: dict[str, list[tuple[Path, str]]] = {}
        for shard in shards:
            with tarfile.open(shard, mode="r:*") as tar:
                key_to_cls: dict[str, str] = {}
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name
                    if "." not in name:
                        continue
                    key, _, ext = name.partition(".")
                    if ext == "cls":
                        f = tar.extractfile(member)
                        if f is not None:
                            key_to_cls[key] = f.read().decode("utf-8").strip()
                for key, cls in key_to_cls.items():
                    index.setdefault(cls, []).append((shard, key))

        if not index:
            raise ValueError("Cards loader found 0 classes across shards")
        self._index = index
        self._rng = rng

    @property
    def class_names(self) -> list[str]:
        return sorted(self._index.keys())

    @property
    def counts_by_class(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._index.items()}

    def __len__(self) -> int:
        return sum(len(v) for v in self._index.values())

    def sample(self, card_name: str | None = None) -> CardSample:
        """Sample one card (optionally constrained to a class)."""
        if card_name is None:
            card_name = self.class_names[int(self._rng.integers(0, len(self._index)))]
        if card_name not in self._index:
            raise KeyError(f"unknown card class {card_name!r}")
        entries = self._index[card_name]
        shard, key = entries[int(self._rng.integers(0, len(entries)))]
        return self._load_sample(shard, key, card_name)

    def _load_sample(self, shard: Path, key: str, card_name: str) -> CardSample:
        image: NDArray[np.uint8] | None = None
        hull_hl: NDArray[np.intp] | None = None
        hull_lr: NDArray[np.intp] | None = None
        with tarfile.open(shard, mode="r:*") as tar:
            for member in tar:
                if not member.isfile() or "." not in member.name:
                    continue
                k, _, ext = member.name.partition(".")
                if k != key:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                payload = f.read()
                if ext in {"png", "jpg", "jpeg", "webp"}:
                    image = _decode_image(payload)
                elif ext == "hull_hl.npy":
                    hull_hl = _decode_npy(payload)
                elif ext == "hull_lr.npy":
                    hull_lr = _decode_npy(payload)
        if image is None or hull_hl is None or hull_lr is None:
            raise ValueError(
                f"shard {shard.name} key {key!r} ({card_name}) is missing required files"
            )
        return CardSample(name=card_name, image=image, hull_hl=hull_hl, hull_lr=hull_lr)


__all__ = ["Backgrounds", "CardSample", "Cards"]
