"""Discover the shape of a YOLO-format dataset on disk.

Public playing-card datasets ship in several near-identical layouts (Roboflow
exports, Ultralytics conventions, hand-rolled folders). This module normalises
them into one :class:`DatasetLayout` so the rest of the pipeline never has to
guess where images and labels live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Image suffixes we treat as dataset samples.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

#: Split directory names, mapped to the canonical split they represent.
SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class SplitPaths:
    """Where one split's images and labels live."""

    name: str
    images_dir: Path
    labels_dir: Path | None


@dataclass
class DatasetLayout:
    """A normalised view of a YOLO dataset directory."""

    root: Path
    splits: list[SplitPaths] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    config_path: Path | None = None

    @property
    def split_names(self) -> list[str]:
        return [split.name for split in self.splits]


def _read_scalar(value: str) -> str:
    """Strip inline comments and surrounding quotes from a YAML scalar."""
    text = value.split("#", 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def read_data_yaml(path: Path) -> dict[str, object]:
    """Parse the flat subset of `data.yaml` that YOLO datasets actually use.

    Handles ``key: value`` scalars, ``names: [a, b]`` inline lists, ``names:``
    block lists (``- a``) and index maps (``0: a``). Anything more exotic is
    ignored rather than guessed at, and callers fall back to directory scanning.
    """
    result: dict[str, object] = {}
    names_list: list[str] = []
    names_map: dict[int, str] = {}
    in_names = False

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indented = raw_line[:1].isspace()
        line = raw_line.strip()

        if in_names and (indented or line.startswith("-")):
            if line.startswith("-"):
                names_list.append(_read_scalar(line[1:]))
            elif ":" in line:
                index_text, name_text = line.split(":", 1)
                try:
                    names_map[int(index_text.strip())] = _read_scalar(name_text)
                except ValueError:
                    continue
            continue

        in_names = False
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "names":
            if value.startswith("["):
                inline = value.strip("[]")
                names_list = [_read_scalar(part) for part in inline.split(",") if part.strip()]
            else:
                in_names = True
            continue

        result[key] = _read_scalar(value)

    if names_map and not names_list:
        names_list = [names_map[index] for index in sorted(names_map)]
    if names_list:
        result["names"] = names_list
    return result


def _labels_dir_for(images_dir: Path) -> Path | None:
    """Find the labels directory paired with an images directory."""
    candidates = [
        images_dir.parent / "labels",
        images_dir.with_name("labels"),
        images_dir.parent.parent / "labels" / images_dir.name,
        images_dir,  # labels sitting beside the images
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def _has_images(directory: Path) -> bool:
    return any(
        item.suffix.lower() in IMAGE_SUFFIXES for item in directory.iterdir() if item.is_file()
    )


def _discover_splits(root: Path) -> list[SplitPaths]:
    """Scan for `<split>/images`, `images/<split>`, or a flat image folder."""
    found: dict[str, SplitPaths] = {}

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        canonical = SPLIT_ALIASES.get(child.name.lower())

        # Layout A: <root>/<split>/images
        if canonical:
            images_dir = child / "images" if (child / "images").is_dir() else child
            if images_dir.is_dir() and _has_images(images_dir):
                found[canonical] = SplitPaths(canonical, images_dir, _labels_dir_for(images_dir))
            continue

        # Layout B: <root>/images/<split>
        if child.name.lower() == "images":
            subdirs = [item for item in sorted(child.iterdir()) if item.is_dir()]
            for subdir in subdirs:
                sub_canonical = SPLIT_ALIASES.get(subdir.name.lower())
                if sub_canonical and _has_images(subdir):
                    found[sub_canonical] = SplitPaths(
                        sub_canonical, subdir, _labels_dir_for(subdir)
                    )
            if not subdirs and _has_images(child):
                found["train"] = SplitPaths("train", child, _labels_dir_for(child))

    return [found[name] for name in ("train", "val", "test") if name in found]


def discover_layout(root: Path) -> DatasetLayout:
    """Normalise the dataset at `root` into a :class:`DatasetLayout`.

    Raises:
        FileNotFoundError: if `root` is not a directory.
        ValueError: if no split containing images can be found.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    # A single-directory dataset (one folder of images) is still worth reading;
    # unwrap a lone top-level folder first, which is how most zips extract.
    entries = [item for item in root.iterdir() if not item.name.startswith(".")]
    directories = [item for item in entries if item.is_dir()]
    if len(directories) == 1 and not any(item.is_file() for item in entries):
        root = directories[0]

    config_path: Path | None = None
    class_names: list[str] = []
    for candidate in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        path = root / candidate
        if path.is_file():
            config_path = path
            parsed = read_data_yaml(path)
            names = parsed.get("names")
            if isinstance(names, list):
                class_names = [str(name) for name in names]
            break

    if not class_names:
        for candidate in ("classes.txt", "obj.names", "cards.names"):
            path = root / candidate
            if path.is_file():
                class_names = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                config_path = config_path or path
                break

    splits = _discover_splits(root)
    if not splits:
        raise ValueError(
            f"No image splits found under {root}. Expected train/val folders "
            "with images, e.g. train/images or images/train."
        )

    return DatasetLayout(root=root, splits=splits, class_names=class_names, config_path=config_path)


__all__ = [
    "IMAGE_SUFFIXES",
    "SPLIT_ALIASES",
    "DatasetLayout",
    "SplitPaths",
    "discover_layout",
    "read_data_yaml",
]
