"""Detect and prune near-duplicate images across and within dataset splits using dHash."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from hakim_vision.datasets.yolo_layout import DatasetLayout, discover_layout

logger = logging.getLogger(__name__)


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for an image as an integer."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize to (width = hash_size + 1, height = hash_size)
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]

    # Pack boolean 2D array into integer
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculate the Hamming distance between two integer hashes."""
    return (hash1 ^ hash2).bit_count()


@dataclass
class DuplicateMatch:
    """A pair of near-duplicate images."""

    image_a: Path
    split_a: str
    image_b: Path
    split_b: str
    distance: int

    def to_dict(self) -> dict[str, object]:
        return {
            "image_a": str(self.image_a),
            "split_a": self.split_a,
            "image_b": str(self.image_b),
            "split_b": self.split_b,
            "distance": self.distance,
        }


@dataclass
class DedupeReport:
    """Outcome of dataset deduplication scan."""

    root: Path
    total_scanned: int = 0
    threshold: int = 4
    cross_split_leaks: list[DuplicateMatch] = field(default_factory=list)
    intra_split_matches: list[DuplicateMatch] = field(default_factory=list)

    @property
    def leaked_eval_images(self) -> list[Path]:
        """Unique eval (valid/test) images that match train images."""
        leaked: set[Path] = set()
        for match in self.cross_split_leaks:
            if match.split_a == "train" and match.split_b in {"val", "valid", "test"}:
                leaked.add(match.image_b)
            elif match.split_b == "train" and match.split_a in {"val", "valid", "test"}:
                leaked.add(match.image_a)
        return sorted(leaked)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "total_scanned": self.total_scanned,
            "threshold": self.threshold,
            "cross_split_leaks_count": len(self.cross_split_leaks),
            "intra_split_matches_count": len(self.intra_split_matches),
            "leaked_eval_images_count": len(self.leaked_eval_images),
            "leaked_eval_images": [str(p) for p in self.leaked_eval_images],
        }


def scan_dataset_duplicates(
    source: Path,
    threshold: int = 4,
    layout: DatasetLayout | None = None,
) -> DedupeReport:
    """Scan dataset for near-duplicate images across and within splits."""
    resolved = layout or discover_layout(source)
    report = DedupeReport(root=resolved.root, threshold=threshold)

    # 1. Compute hashes for each split
    split_hashes: dict[str, list[tuple[Path, int]]] = {}
    for split in resolved.splits:
        entries: list[tuple[Path, int]] = []
        for img_path in sorted(split.images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                continue
            h = compute_dhash(img_path)
            entries.append((img_path, h))
        split_hashes[split.name] = entries
        report.total_scanned += len(entries)

    # 2. Check cross-split and intra-split distances
    split_names = list(split_hashes.keys())
    for i, s1 in enumerate(split_names):
        list1 = split_hashes[s1]
        # Intra-split
        for idx_a in range(len(list1)):
            path_a, hash_a = list1[idx_a]
            for idx_b in range(idx_a + 1, len(list1)):
                path_b, hash_b = list1[idx_b]
                dist = hamming_distance(hash_a, hash_b)
                if dist <= threshold:
                    report.intra_split_matches.append(DuplicateMatch(path_a, s1, path_b, s1, dist))

        # Cross-split
        for j in range(i + 1, len(split_names)):
            s2 = split_names[j]
            list2 = split_hashes[s2]
            for path_a, hash_a in list1:
                for path_b, hash_b in list2:
                    dist = hamming_distance(hash_a, hash_b)
                    if dist <= threshold:
                        report.cross_split_leaks.append(
                            DuplicateMatch(path_a, s1, path_b, s2, dist)
                        )

    return report


def prune_leaked_images(
    report: DedupeReport,
    layout: DatasetLayout | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Delete leaked eval images (and their label files) from valid/test splits."""
    to_delete = report.leaked_eval_images
    pruned: list[Path] = []

    for img_path in to_delete:
        lbl_path = (
            img_path.parent.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        )
        if not lbl_path.is_file():
            # In roboflow layout: <root>/<split>/labels/<stem>.txt
            lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"

        if not dry_run:
            if img_path.is_file():
                img_path.unlink()
            if lbl_path.is_file():
                lbl_path.unlink()
        pruned.append(img_path)

    logger.info("Pruned %d leaked eval images (dry_run=%s)", len(pruned), dry_run)
    return pruned


__all__ = [
    "DedupeReport",
    "DuplicateMatch",
    "compute_dhash",
    "hamming_distance",
    "prune_leaked_images",
    "scan_dataset_duplicates",
]
