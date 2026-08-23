"""Count and validate a YOLO dataset before anyone spends GPU hours on it.

Everything here is derived from the user's actual files: no sampling estimates,
no placeholder numbers. If a split has 3 orphan labels, this reports 3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from hakim_vision.datasets.yolo_layout import (
    IMAGE_SUFFIXES,
    DatasetLayout,
    SplitPaths,
    discover_layout,
)

#: Issue codes surfaced to the UI and CLI.
ISSUE_MISSING_LABEL = "missing_label"
ISSUE_ORPHAN_LABEL = "orphan_label"
ISSUE_EMPTY_LABEL = "empty_label"
ISSUE_MALFORMED_LINE = "malformed_line"
ISSUE_OUT_OF_RANGE = "out_of_range"
ISSUE_UNKNOWN_CLASS = "unknown_class"
ISSUE_NO_LABELS_DIR = "no_labels_dir"

#: How many offending file names to keep per issue, for display.
MAX_ISSUE_EXAMPLES = 5


@dataclass
class Issue:
    """One category of dataset problem, with a few examples."""

    code: str
    split: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def record(self, example: str) -> None:
        self.count += 1
        if len(self.examples) < MAX_ISSUE_EXAMPLES:
            self.examples.append(example)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "split": self.split,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass
class SplitReport:
    """Per-split counts."""

    name: str
    images: int = 0
    labelled_images: int = 0
    instances: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "images": self.images,
            "labelled_images": self.labelled_images,
            "instances": self.instances,
            "class_counts": {str(key): value for key, value in sorted(self.class_counts.items())},
        }


@dataclass
class DatasetReport:
    """The full picture of a dataset directory."""

    root: Path
    splits: list[SplitReport] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    sample_images: list[Path] = field(default_factory=list)
    config_path: Path | None = None

    @property
    def total_images(self) -> int:
        return sum(split.images for split in self.splits)

    @property
    def total_instances(self) -> int:
        return sum(split.instances for split in self.splits)

    @property
    def class_counts(self) -> Counter[int]:
        combined: Counter[int] = Counter()
        for split in self.splits:
            combined.update(split.class_counts)
        return combined

    def class_label(self, index: int) -> str:
        """Human name for a class id, falling back to the raw id."""
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"class_{index}"

    def to_dict(self) -> dict[str, object]:
        counts = self.class_counts
        return {
            "root": str(self.root),
            "config_path": str(self.config_path) if self.config_path else None,
            "class_names": list(self.class_names),
            "splits": [split.to_dict() for split in self.splits],
            "totals": {
                "images": self.total_images,
                "instances": self.total_instances,
                "classes_present": len(counts),
            },
            "class_counts": [
                {"id": index, "name": self.class_label(index), "count": count}
                for index, count in sorted(counts.items())
            ],
            "issues": [issue.to_dict() for issue in self.issues if issue.count],
            "sample_images": [str(path) for path in self.sample_images],
        }


def _iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _parse_label_file(
    path: Path,
    report: SplitReport,
    malformed: Issue,
    out_of_range: Issue,
    unknown_class: Issue,
    class_count: int,
) -> int:
    """Add one label file's boxes to `report`; return the number of valid boxes."""
    valid = 0
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        # YOLO detection: class cx cy w h. Segmentation adds polygon points.
        if len(parts) < 5:
            malformed.record(f"{path.name}:{line_number}")
            continue
        try:
            class_id = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:5]]
        except ValueError:
            malformed.record(f"{path.name}:{line_number}")
            continue

        if any(value < -0.001 or value > 1.001 for value in coordinates):
            out_of_range.record(f"{path.name}:{line_number}")
            continue
        if class_id < 0 or (class_count and class_id >= class_count):
            unknown_class.record(f"{path.name}:{line_number} -> id {class_id}")
            continue

        report.class_counts[class_id] += 1
        valid += 1
    return valid


def _inspect_split(
    split: SplitPaths, class_count: int, issues: list[Issue]
) -> tuple[SplitReport, list[Path]]:
    report = SplitReport(name=split.name)
    images = _iter_images(split.images_dir)
    report.images = len(images)

    missing = Issue(ISSUE_MISSING_LABEL, split.name)
    orphan = Issue(ISSUE_ORPHAN_LABEL, split.name)
    empty = Issue(ISSUE_EMPTY_LABEL, split.name)
    malformed = Issue(ISSUE_MALFORMED_LINE, split.name)
    out_of_range = Issue(ISSUE_OUT_OF_RANGE, split.name)
    unknown_class = Issue(ISSUE_UNKNOWN_CLASS, split.name)

    if split.labels_dir is None:
        no_labels = Issue(ISSUE_NO_LABELS_DIR, split.name)
        no_labels.record(str(split.images_dir))
        issues.append(no_labels)
        return report, images[:6]

    seen_labels: set[str] = set()
    for image_path in images:
        label_path = split.labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing.record(image_path.name)
            continue
        seen_labels.add(label_path.name)
        boxes = _parse_label_file(
            label_path, report, malformed, out_of_range, unknown_class, class_count
        )
        if boxes:
            report.labelled_images += 1
            report.instances += boxes
        else:
            empty.record(label_path.name)

    for label_path in sorted(split.labels_dir.glob("*.txt")):
        if label_path.name not in seen_labels and label_path.stem != "classes":
            orphan.record(label_path.name)

    issues.extend([missing, orphan, empty, malformed, out_of_range, unknown_class])
    return report, images[:6]


def inspect_dataset(root: Path, layout: DatasetLayout | None = None) -> DatasetReport:
    """Walk every split and count images, boxes, classes, and problems."""
    resolved = layout or discover_layout(root)
    issues: list[Issue] = []
    splits: list[SplitReport] = []
    samples: list[Path] = []

    class_count = len(resolved.class_names)
    for split in resolved.splits:
        split_report, split_samples = _inspect_split(split, class_count, issues)
        splits.append(split_report)
        if len(samples) < 6:
            samples.extend(split_samples[: 6 - len(samples)])

    report = DatasetReport(
        root=resolved.root,
        splits=splits,
        class_names=list(resolved.class_names),
        issues=issues,
        sample_images=samples,
        config_path=resolved.config_path,
    )

    # A dataset with no names file still has class ids; name them by id so the
    # rest of the pipeline (and the mapping UI) has something to show.
    if not report.class_names:
        highest = max(report.class_counts, default=-1)
        report.class_names = [f"class_{index}" for index in range(highest + 1)]

    return report


__all__ = [
    "DatasetReport",
    "Issue",
    "SplitReport",
    "inspect_dataset",
]
