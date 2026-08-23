"""Dataset ingestion: discover, inspect, and remap third-party card datasets."""

from hakim_vision.datasets.inspection import DatasetReport, Issue, SplitReport, inspect_dataset
from hakim_vision.datasets.remap import (
    RemapResult,
    RemapSuggestion,
    normalise_card_name,
    remap_dataset,
    suggest_mapping,
)
from hakim_vision.datasets.yolo_layout import DatasetLayout, SplitPaths, discover_layout

__all__ = [
    "DatasetLayout",
    "DatasetReport",
    "Issue",
    "RemapResult",
    "RemapSuggestion",
    "SplitPaths",
    "SplitReport",
    "discover_layout",
    "inspect_dataset",
    "normalise_card_name",
    "remap_dataset",
    "suggest_mapping",
]
