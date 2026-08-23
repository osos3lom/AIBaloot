"""Map a third-party card dataset onto the 32-card Baloot deck.

Public playing-card datasets are 52-card poker decks. Baloot uses 32 cards
(A K Q J 10 9 8 7), so ranks 2-6 and jokers must be dropped and the remaining
classes re-indexed to the Baloot order. This module suggests that mapping and
then materialises a new dataset from it.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from hakim_vision.datasets.yolo_layout import DatasetLayout, discover_layout
from hakim_vision.models.yolo_export import get_baloot_classes

#: Rank spellings seen in the wild, normalised to the Baloot rank token.
RANK_ALIASES: dict[str, str] = {
    "a": "A",
    "ace": "A",
    "1": "A",
    "k": "K",
    "king": "K",
    "q": "Q",
    "queen": "Q",
    "j": "J",
    "jack": "J",
    "knave": "J",
    "10": "10",
    "t": "10",
    "ten": "10",
    "9": "9",
    "nine": "9",
    "8": "8",
    "eight": "8",
    "7": "7",
    "seven": "7",
}

#: Suit spellings, normalised to the Baloot suit letter.
SUIT_ALIASES: dict[str, str] = {
    "h": "h",
    "heart": "h",
    "hearts": "h",
    "♥": "h",
    "d": "d",
    "diamond": "d",
    "diamonds": "d",
    "♦": "d",
    "c": "c",
    "club": "c",
    "clubs": "c",
    "♣": "c",
    "s": "s",
    "spade": "s",
    "spades": "s",
    "♠": "s",
}

#: Ranks that exist in a poker deck but never in Baloot.
NON_BALOOT_RANKS: frozenset[str] = frozenset({"2", "3", "4", "5", "6"})

_SEPARATORS = re.compile(r"[\s_\-./]+")
_COMPACT = re.compile(r"^(10|[2-9]|[akqjt])([hdcs])$")
_COMPACT_REVERSED = re.compile(r"^([hdcs])(10|[2-9]|[akqjt])$")


@dataclass
class RemapSuggestion:
    """What we think a source class should become."""

    source_id: int
    source_name: str
    target: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "target": self.target,
            "reason": self.reason,
        }


@dataclass
class RemapResult:
    """Outcome of materialising a remapped dataset."""

    output_root: Path
    data_yaml: Path
    images_written: int = 0
    labels_written: int = 0
    boxes_kept: int = 0
    boxes_dropped: int = 0
    images_skipped_empty: int = 0
    per_split: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": str(self.output_root),
            "data_yaml": str(self.data_yaml),
            "images_written": self.images_written,
            "labels_written": self.labels_written,
            "boxes_kept": self.boxes_kept,
            "boxes_dropped": self.boxes_dropped,
            "images_skipped_empty": self.images_skipped_empty,
            "per_split": dict(self.per_split),
        }


def normalise_card_name(name: str) -> str | None:
    """Turn a source class name into a Baloot `<rank><suit>` token, or None.

    Understands ``10c``, ``C10``, ``ten of clubs``, ``10_of_Clubs``, ``AS`` and
    similar. Returns None for anything outside the 32-card deck — a poker ``2c``
    is a real card but has no Baloot token, so it maps to None just like a joker.
    """
    text = name.strip().lower()
    if not text:
        return None

    compact = _SEPARATORS.sub("", text)
    match = _COMPACT.match(compact)
    if match:
        rank = RANK_ALIASES.get(match.group(1))
        suit = SUIT_ALIASES.get(match.group(2))
        return f"{rank}{suit}" if rank and suit else None

    match = _COMPACT_REVERSED.match(compact)
    if match:
        suit = SUIT_ALIASES.get(match.group(1))
        rank = RANK_ALIASES.get(match.group(2))
        return f"{rank}{suit}" if rank and suit else None

    words = [word for word in _SEPARATORS.split(text) if word and word != "of"]
    rank_token: str | None = None
    suit_token: str | None = None
    for word in words:
        if rank_token is None and word in RANK_ALIASES:
            rank_token = RANK_ALIASES[word]
            continue
        if suit_token is None and word in SUIT_ALIASES:
            suit_token = SUIT_ALIASES[word]
    if rank_token and suit_token:
        return f"{rank_token}{suit_token}"
    return None


def _is_non_baloot_rank(name: str) -> bool:
    compact = _SEPARATORS.sub("", name.strip().lower())
    head = compact[:2] if compact[:2] == "10" else compact[:1]
    return head in NON_BALOOT_RANKS


def suggest_mapping(source_classes: list[str]) -> list[RemapSuggestion]:
    """Propose a Baloot target for every source class.

    The result is advisory: the studio shows it as an editable table, because a
    silent wrong mapping is far more expensive than a visible one.
    """
    baloot = set(get_baloot_classes())
    suggestions: list[RemapSuggestion] = []

    for index, name in enumerate(source_classes):
        normalised = normalise_card_name(name)
        if normalised and normalised in baloot:
            reason = "exact" if normalised == name.strip() else "normalised"
            suggestions.append(RemapSuggestion(index, name, normalised, reason))
        elif _is_non_baloot_rank(name):
            suggestions.append(
                RemapSuggestion(index, name, None, "rank not in the 32-card Baloot deck")
            )
        else:
            suggestions.append(RemapSuggestion(index, name, None, "unrecognised class name"))

    return suggestions


def mapping_to_index_map(
    source_classes: list[str], mapping: dict[str, str | None]
) -> dict[int, int]:
    """Convert a `{source name: baloot name}` mapping to `{source id: baloot id}`."""
    baloot_index = {name: index for index, name in enumerate(get_baloot_classes())}
    index_map: dict[int, int] = {}
    for source_id, source_name in enumerate(source_classes):
        target = mapping.get(source_name)
        if target is None:
            continue
        if target not in baloot_index:
            raise ValueError(f"'{target}' is not one of the 32 Baloot classes")
        index_map[source_id] = baloot_index[target]
    return index_map


def _place_image(source: Path, destination: Path, link_mode: str) -> None:
    """Hardlink, symlink, or copy an image into the new dataset."""
    if destination.exists():
        return
    if link_mode in {"auto", "link"}:
        try:
            os.link(source, destination)
            return
        except (OSError, NotImplementedError):
            pass
        try:
            destination.symlink_to(source.resolve())
            return
        except (OSError, NotImplementedError):
            if link_mode == "link":
                raise
    shutil.copy2(source, destination)


def _write_data_yaml(output_root: Path, split_names: list[str]) -> Path:
    """Write the Ultralytics config for the remapped dataset."""
    classes = get_baloot_classes()
    lines = [f"path: {output_root.resolve().as_posix()}"]
    for split in split_names:
        key = split if split != "test" else "test"
        lines.append(f"{key}: images/{split}")
    if "val" not in split_names:
        # Ultralytics requires a val entry; reuse train rather than fail late.
        lines.append("val: images/train")
    lines.append("names:")
    lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))

    path = output_root / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def remap_dataset(
    root: Path,
    output_root: Path,
    mapping: dict[str, str | None],
    *,
    layout: DatasetLayout | None = None,
    link_mode: str = "auto",
    drop_empty: bool = True,
) -> RemapResult:
    """Write a Baloot-class dataset from `root` into `output_root`.

    Args:
        mapping: source class name -> Baloot class name, or None to drop it.
        link_mode: ``auto`` (hardlink, else symlink, else copy), ``link``, or ``copy``.
        drop_empty: skip images whose boxes were all dropped by the mapping.
    """
    resolved = layout or discover_layout(root)
    if not resolved.class_names:
        raise ValueError(
            "The source dataset has no class names; add a data.yaml or classes.txt first."
        )

    index_map = mapping_to_index_map(resolved.class_names, mapping)
    if not index_map:
        raise ValueError("The mapping keeps no classes at all.")

    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    result = RemapResult(output_root=output_root, data_yaml=output_root / "data.yaml")

    for split in resolved.splits:
        images_out = output_root / "images" / split.name
        labels_out = output_root / "labels" / split.name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        kept_in_split = 0

        for image_path in sorted(split.images_dir.iterdir()):
            if not image_path.is_file():
                continue
            if split.labels_dir is None:
                continue
            label_path = split.labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue

            kept_lines: list[str] = []
            for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw_line.split()
                if len(parts) < 5:
                    continue
                try:
                    source_id = int(float(parts[0]))
                except ValueError:
                    continue
                target_id = index_map.get(source_id)
                if target_id is None:
                    result.boxes_dropped += 1
                    continue
                kept_lines.append(" ".join([str(target_id), *parts[1:]]))

            if not kept_lines and drop_empty:
                result.images_skipped_empty += 1
                continue

            (labels_out / f"{image_path.stem}.txt").write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8"
            )
            _place_image(image_path, images_out / image_path.name, link_mode)
            result.images_written += 1
            result.labels_written += 1
            result.boxes_kept += len(kept_lines)
            kept_in_split += 1

        result.per_split[split.name] = kept_in_split

    result.data_yaml = _write_data_yaml(output_root, [split.name for split in resolved.splits])
    return result


__all__ = [
    "NON_BALOOT_RANKS",
    "RemapResult",
    "RemapSuggestion",
    "mapping_to_index_map",
    "normalise_card_name",
    "remap_dataset",
    "suggest_mapping",
]
