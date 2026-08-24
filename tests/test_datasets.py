"""Tests for dataset discovery, inspection, and Baloot remapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hakim_vision.datasets import (
    discover_layout,
    inspect_dataset,
    normalise_card_name,
    remap_dataset,
    suggest_mapping,
)
from hakim_vision.datasets.remap import mapping_to_index_map
from hakim_vision.datasets.yolo_layout import read_data_yaml
from hakim_vision.models.yolo_export import get_baloot_classes


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _roboflow_dataset(root: Path, classes: list[str]) -> Path:
    """A `train/images` + `train/labels` layout, as Roboflow exports it."""
    names = ", ".join(f"'{name}'" for name in classes)
    _write(root / "data.yaml", f"train: train/images\nval: valid/images\nnames: [{names}]\n")

    for split, count in (("train", 3), ("valid", 2)):
        for index in range(count):
            _write(root / split / "images" / f"img_{index}.jpg", "x")
            _write(
                root / split / "labels" / f"img_{index}.txt",
                f"{index % len(classes)} 0.5 0.5 0.2 0.3\n0 0.25 0.25 0.1 0.1\n",
            )
    return root


def test_read_data_yaml_handles_inline_block_and_index_names(tmp_path: Path):
    inline = _write(tmp_path / "a.yaml", "train: t\nnames: ['Ah', '10c']\n")
    assert read_data_yaml(inline)["names"] == ["Ah", "10c"]

    block = _write(tmp_path / "b.yaml", "names:\n  - Ah\n  - 10c\ntrain: t\n")
    assert read_data_yaml(block)["names"] == ["Ah", "10c"]

    indexed = _write(tmp_path / "c.yaml", "names:\n  0: Ah\n  1: 10c\n")
    assert read_data_yaml(indexed)["names"] == ["Ah", "10c"]

    commented = _write(tmp_path / "d.yaml", "path: /data  # root\n")
    assert read_data_yaml(commented)["path"] == "/data"


def test_discover_layout_reads_roboflow_style(tmp_path: Path):
    _roboflow_dataset(tmp_path / "cards", ["Ah", "2c"])
    layout = discover_layout(tmp_path / "cards")

    assert layout.split_names == ["train", "val"]
    assert layout.class_names == ["Ah", "2c"]
    assert layout.splits[0].labels_dir is not None


def test_discover_layout_reads_images_split_layout(tmp_path: Path):
    root = tmp_path / "cards"
    _write(root / "classes.txt", "Ah\nKh\n")
    _write(root / "images" / "train" / "a.jpg", "x")
    _write(root / "labels" / "train" / "a.txt", "0 0.5 0.5 0.2 0.2\n")

    layout = discover_layout(root)
    assert layout.split_names == ["train"]
    assert layout.class_names == ["Ah", "Kh"]


def test_discover_layout_unwraps_a_single_top_level_folder(tmp_path: Path):
    _roboflow_dataset(tmp_path / "zip_root" / "cards", ["Ah"])
    layout = discover_layout(tmp_path / "zip_root")
    assert layout.root.name == "cards"


def test_discover_layout_rejects_a_directory_without_images(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No image splits"):
        discover_layout(tmp_path / "empty")

    with pytest.raises(FileNotFoundError):
        discover_layout(tmp_path / "nope")


def test_inspect_counts_images_boxes_and_classes(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "cards", ["Ah", "2c"])
    report = inspect_dataset(root)

    assert report.total_images == 5
    assert report.total_instances == 10  # 2 boxes per image
    assert report.class_counts[0] == 8
    assert report.class_label(1) == "2c"
    assert [split.name for split in report.splits] == ["train", "val"]


def test_inspect_flags_missing_orphan_empty_and_malformed_labels(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "cards", ["Ah"])
    _write(root / "train" / "images" / "unlabelled.jpg", "x")
    _write(root / "train" / "labels" / "ghost.txt", "0 0.5 0.5 0.2 0.2\n")
    _write(root / "train" / "labels" / "img_0.txt", "")
    _write(root / "train" / "labels" / "img_1.txt", "not a box\n")
    _write(root / "train" / "labels" / "img_2.txt", "0 5.0 0.5 0.2 0.2\n")

    codes = {issue.code: issue.count for issue in inspect_dataset(root).issues if issue.count}
    assert codes["missing_label"] == 1
    assert codes["orphan_label"] == 1
    assert codes["empty_label"] == 3  # the empty file plus two files with no valid box
    assert codes["malformed_line"] == 1
    assert codes["out_of_range"] == 1


def test_inspect_flags_class_ids_outside_the_name_list(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "cards", ["Ah"])
    _write(root / "train" / "labels" / "img_0.txt", "7 0.5 0.5 0.2 0.2\n")

    codes = {issue.code for issue in inspect_dataset(root).issues if issue.count}
    assert "unknown_class" in codes


def test_inspect_names_classes_by_id_when_the_dataset_has_no_names(tmp_path: Path):
    root = tmp_path / "cards"
    _write(root / "images" / "train" / "a.jpg", "x")
    _write(root / "labels" / "train" / "a.txt", "2 0.5 0.5 0.2 0.2\n")

    report = inspect_dataset(root)
    assert report.class_label(2) == "class_2"
    assert report.total_instances == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Ah", "Ah"),
        ("10c", "10c"),
        ("AS", "As"),
        ("c10", "10c"),
        ("ten of clubs", "10c"),
        ("Queen_of_Hearts", "Qh"),
        ("jack-of-spades", "Js"),
        ("king of diamonds", "Kd"),
        ("2c", None),  # a real card, but not one of the 32
        ("joker", None),
        ("card", None),
        ("", None),
    ],
)
def test_normalise_card_name(source: str, expected: str | None):
    assert normalise_card_name(source) == expected


def test_suggest_mapping_keeps_baloot_ranks_and_drops_the_rest():
    suggestions = suggest_mapping(["Ah", "10c", "2c", "5s", "joker", "QUEEN of hearts"])
    by_name = {item.source_name: item for item in suggestions}

    assert by_name["Ah"].target == "Ah"
    assert by_name["10c"].target == "10c"
    assert by_name["QUEEN of hearts"].target == "Qh"
    assert by_name["2c"].target is None
    assert "Baloot deck" in by_name["5s"].reason
    assert by_name["joker"].target is None


def test_mapping_to_index_map_uses_baloot_ordering():
    classes = get_baloot_classes()
    index_map = mapping_to_index_map(["x", "y"], {"x": classes[0], "y": classes[5]})
    assert index_map == {0: 0, 1: 5}

    with pytest.raises(ValueError, match="not one of the 32"):
        mapping_to_index_map(["x"], {"x": "3h"})


def test_remap_writes_a_baloot_dataset_and_drops_unmapped_boxes(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "poker", ["Ah", "2c"])
    output = tmp_path / "baloot"

    result = remap_dataset(root, output, {"Ah": "Ah", "2c": None}, link_mode="copy")

    assert result.data_yaml.is_file()
    assert result.boxes_kept > 0
    assert result.boxes_dropped > 0
    assert (output / "images" / "train").is_dir()
    assert (output / "labels" / "val").is_dir()

    yaml_text = result.data_yaml.read_text(encoding="utf-8")
    assert "0: Ah" in yaml_text
    assert "31: 7s" in yaml_text

    # Every surviving box points at the Baloot index for Ah, which is 0.
    for label_file in (output / "labels" / "train").glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            assert line.split()[0] == "0"


def test_remap_reindexes_to_the_target_class_id(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "poker", ["Ah", "spare"])
    output = tmp_path / "baloot"
    target = get_baloot_classes()[9]

    remap_dataset(root, output, {"Ah": target, "spare": None}, link_mode="copy")

    ids = {
        line.split()[0]
        for label_file in (output / "labels" / "train").glob("*.txt")
        for line in label_file.read_text(encoding="utf-8").splitlines()
    }
    assert ids == {"9"}


def test_remap_keeps_empty_images_when_asked(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "poker", ["Ah", "2c"])
    dropped = remap_dataset(root, tmp_path / "a", {"Ah": None, "2c": "Ah"}, link_mode="copy")
    kept = remap_dataset(
        root, tmp_path / "b", {"Ah": None, "2c": "Ah"}, link_mode="copy", drop_empty=False
    )
    assert kept.images_written > dropped.images_written


def test_remap_refuses_a_mapping_that_keeps_nothing(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "poker", ["Ah"])
    with pytest.raises(ValueError, match="keeps no classes"):
        remap_dataset(root, tmp_path / "out", {"Ah": None}, unmapped="drop")


def test_remap_mapping_round_trips_through_json(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "poker", ["Ah", "2c"])
    report = inspect_dataset(root)
    mapping = {item.source_name: item.target for item in suggest_mapping(report.class_names)}

    serialised = json.loads(json.dumps(mapping))
    result = remap_dataset(root, tmp_path / "out", serialised, link_mode="copy", unmapped="other")
    assert result.images_written > 0


def test_remap_unmapped_other_mode(tmp_path: Path):
    root = _roboflow_dataset(tmp_path / "poker", ["Ah", "2c", "3d"])
    output = tmp_path / "baloot"
    result = remap_dataset(
        root, output, {"Ah": "Ah", "2c": None, "3d": None}, link_mode="copy", unmapped="other"
    )

    assert result.data_yaml.is_file()
    yaml_text = result.data_yaml.read_text(encoding="utf-8")
    assert "32: other" in yaml_text

    # Check that labels contain 0 (for Ah) and 32 (for other)
    target_ids = set()
    for label_file in (output / "labels" / "train").glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            target_ids.add(line.split()[0])
    assert "0" in target_ids
    assert "32" in target_ids


def test_remap_box_hygiene_filters_tiny_boxes_and_clips(tmp_path: Path):
    root = tmp_path / "raw"
    _write(root / "data.yaml", "train: train/images\nval: train/images\nnames: ['Ah']\n")
    _write(root / "train" / "images" / "test.jpg", "x")
    # 1 normal box, 1 tiny box (< 6/416 = 0.0144), 1 out-of-bounds box to clip, 1 zero-width box
    _write(
        root / "train" / "labels" / "test.txt",
        "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.005 0.005\n0 1.05 -0.05 0.3 0.3\n0 0.5 0.5 0.0 0.2\n",
    )

    output = tmp_path / "clean"
    remap_dataset(root, output, {"Ah": "Ah"}, link_mode="copy", min_box_pixels=6.0, imgsz=416)

    lines = (output / "labels" / "train" / "test.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # normal box and clipped box (tiny and zero-width dropped)

    # Verify clipped box coordinates stay within [0, 1]
    clipped_parts = lines[1].split()
    cx, cy = float(clipped_parts[1]), float(clipped_parts[2])
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0


def test_preview_dataset(tmp_path: Path):
    import cv2
    import numpy as np

    from hakim_vision.datasets.preview import preview_dataset

    root = tmp_path / "cards"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "c1.jpg"), dummy_img)
    (lbl_dir / "c1.txt").write_text("0 0.5 0.5 0.2 0.2\n32 0.8 0.8 0.1 0.1\n", encoding="utf-8")
    _write(root / "data.yaml", "train: images/train\nnames: {0: 'Ah', 32: 'other'}\n")

    out_dir = tmp_path / "previews"
    rendered = preview_dataset(root, out_dir, count=1)
    assert len(rendered) == 1
    assert rendered[0].is_file()


def test_dedupe_scan_and_prune(tmp_path: Path):
    import cv2
    import numpy as np

    from hakim_vision.datasets.dedupe import (
        compute_dhash,
        hamming_distance,
        prune_leaked_images,
        scan_dataset_duplicates,
    )

    root = tmp_path / "ds"
    train_dir = root / "train" / "images"
    val_dir = root / "valid" / "images"
    train_lbl = root / "train" / "labels"
    val_lbl = root / "valid" / "labels"
    for d in (train_dir, val_dir, train_lbl, val_lbl):
        d.mkdir(parents=True)

    _write(root / "data.yaml", "train: train/images\nval: valid/images\nnames: ['Ah']\n")

    # Create two identical images (one in train, one in valid)
    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img1[20:50, 20:50] = 255
    cv2.imwrite(str(train_dir / "card1.jpg"), img1)
    (train_lbl / "card1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    cv2.imwrite(str(val_dir / "card1_leak.jpg"), img1)
    (val_lbl / "card1_leak.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Compute dHash
    h1 = compute_dhash(train_dir / "card1.jpg")
    h2 = compute_dhash(val_dir / "card1_leak.jpg")
    assert hamming_distance(h1, h2) == 0

    # Scan duplicates
    report = scan_dataset_duplicates(root, threshold=2)
    assert len(report.cross_split_leaks) == 1
    assert len(report.leaked_eval_images) == 1

    # Prune leaks
    pruned = prune_leaked_images(report, dry_run=False)
    assert len(pruned) == 1
    assert not (val_dir / "card1_leak.jpg").exists()
    assert not (val_lbl / "card1_leak.txt").exists()
    assert (train_dir / "card1.jpg").exists()
