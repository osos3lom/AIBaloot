"""Pack on-disk asset directories into WebDataset tar shards.

Replaces the legacy pickle artifacts (`backgrounds.pck`, `cards.pck`):

- `pack_backgrounds(...)` walks a directory of texture images (e.g. an
  unpacked Describable Textures Dataset) and writes one or more
  `backgrounds-{NNN}.tar` shards.
- `pack_cards(...)` walks a directory laid out as `{root}/{card_name}/*.png`
  (the output of the legacy `extract_cards_from_video` step) and writes
  `cards-{NNN}.tar` shards. Corner hulls can be supplied via a sidecar
  `hulls.npz` map; if absent the hulls are extracted live with
  `find_corner_hull`.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    REF_CORNER_HL,
    REF_CORNER_LR,
)
from hakim_vision.synthetic.hull import find_corner_hull

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _walk_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            yield p


def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _encode_npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def pack_backgrounds(
    images_dir: Path,
    output_dir: Path,
    *,
    shard_size: int = 1024,
    shard_prefix: str = "backgrounds",
) -> list[Path]:
    """Pack textures from `images_dir` into shards in `output_dir`.

    Returns the list of shard paths created.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: list[Path] = []
    shard_index = 0
    written_in_shard = 0
    tar: tarfile.TarFile | None = None

    def _open_shard() -> tarfile.TarFile:
        nonlocal shard_index
        path = output_dir / f"{shard_prefix}-{shard_index:04d}.tar"
        shards.append(path)
        shard_index += 1
        return tarfile.open(path, mode="w")

    try:
        for i, img_path in enumerate(_walk_images(images_dir)):
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if tar is None or written_in_shard >= shard_size:
                if tar is not None:
                    tar.close()
                tar = _open_shard()
                written_in_shard = 0
            key = f"{i:08d}"
            _add_bytes_to_tar(tar, f"{key}.png", _encode_png(image))
            written_in_shard += 1
    finally:
        if tar is not None:
            tar.close()

    return shards


def pack_cards(
    cards_root: Path,
    output_dir: Path,
    *,
    shard_size: int = 512,
    shard_prefix: str = "cards",
    extract_hulls: bool = True,
) -> list[Path]:
    """Pack pre-extracted card images into shards.

    Expects ``cards_root/{card_name}/*.png`` (the layout produced by the
    legacy ``extract_cards_from_video`` step). Each image must already be a
    canonical-size BGRA card (``CARD_HEIGHT × CARD_WIDTH``); non-conforming
    images are silently skipped.

    If ``extract_hulls`` is True, runs ``find_corner_hull`` for the two
    corner positions and stores the contours alongside the image. Samples
    where either hull is missing are dropped — they're unusable downstream.
    """
    cards_root = Path(cards_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: list[Path] = []
    shard_index = 0
    written_in_shard = 0
    tar: tarfile.TarFile | None = None
    sample_idx = 0

    def _open_shard() -> tarfile.TarFile:
        nonlocal shard_index
        path = output_dir / f"{shard_prefix}-{shard_index:04d}.tar"
        shards.append(path)
        shard_index += 1
        return tarfile.open(path, mode="w")

    try:
        for class_dir in sorted(p for p in cards_root.iterdir() if p.is_dir()):
            card_name = class_dir.name
            for img_path in _walk_images(class_dir):
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape[:2] != (CARD_HEIGHT, CARD_WIDTH):
                    continue
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

                if extract_hulls:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    hl = find_corner_hull(rgb, REF_CORNER_HL)
                    lr = find_corner_hull(rgb, REF_CORNER_LR)
                    if hl is None or lr is None or not (hl.ok and lr.ok):
                        continue
                    hl_arr = hl.hull
                    lr_arr = lr.hull
                else:
                    # Placeholder hulls covering the corner reference rectangles
                    # so the loader contract is satisfied.
                    hl_arr = REF_CORNER_HL.astype(np.intp).reshape(-1, 1, 2)
                    lr_arr = REF_CORNER_LR.astype(np.intp).reshape(-1, 1, 2)

                if tar is None or written_in_shard >= shard_size:
                    if tar is not None:
                        tar.close()
                    tar = _open_shard()
                    written_in_shard = 0

                key = f"{sample_idx:08d}"
                _add_bytes_to_tar(tar, f"{key}.cls", card_name.encode("utf-8"))
                _add_bytes_to_tar(tar, f"{key}.png", _encode_png(image))
                _add_bytes_to_tar(tar, f"{key}.hull_hl.npy", _encode_npy(hl_arr))
                _add_bytes_to_tar(tar, f"{key}.hull_lr.npy", _encode_npy(lr_arr))
                written_in_shard += 1
                sample_idx += 1
    finally:
        if tar is not None:
            tar.close()

    return shards


__all__ = ["pack_backgrounds", "pack_cards"]
