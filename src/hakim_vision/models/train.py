"""Train a Baloot card detector with Ultralytics YOLO.

This module runs real training. It has no simulation mode: when Ultralytics or
a dataset is missing it raises, so nothing downstream can mistake a placeholder
for a trained model.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Model checkpoints that make sense as a starting point for card detection.
SUPPORTED_MODELS: tuple[str, ...] = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "rtdetr-l.pt",
)

#: Columns in Ultralytics' results.csv we surface as live metrics.
METRIC_COLUMNS: dict[str, str] = {
    "train/box_loss": "box_loss",
    "train/cls_loss": "cls_loss",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


@dataclass
class TrainingConfig:
    """Hyper-parameters for one training run."""

    data: Path
    model: str = "yolo11n.pt"
    epochs: int = 120
    imgsz: int = 704
    batch: int = 32
    device: str = ""
    workers: int = 8
    patience: int = 20
    project: Path = Path("runs/hakim")
    name: str = "baloot"
    resume: bool = False
    seed: int = 42
    fliplr: float = 0.0
    flipud: float = 0.0
    degrees: float = 180.0
    cos_lr: bool = True
    close_mosaic: int = 10
    scale: float = 0.5
    amp: bool = True
    half: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        """Ultralytics keyword arguments for `YOLO.train()`."""
        kwargs: dict[str, Any] = {
            "data": str(self.data),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "patience": self.patience,
            "project": str(self.project),
            "name": self.name,
            "seed": self.seed,
            "resume": self.resume,
            "exist_ok": True,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "degrees": self.degrees,
            "cos_lr": self.cos_lr,
            "close_mosaic": self.close_mosaic,
            "scale": self.scale,
            "amp": self.amp,
            "half": self.half,
        }
        if self.device:
            kwargs["device"] = self.device
        return kwargs


@dataclass
class TrainingOutcome:
    """Where a finished run put its artefacts."""

    run_dir: Path
    best_weights: Path | None = None
    last_weights: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "best_weights": str(self.best_weights) if self.best_weights else None,
            "last_weights": str(self.last_weights) if self.last_weights else None,
            "metrics": dict(self.metrics),
        }


class UltralyticsMissingError(RuntimeError):
    """Raised when training is requested without Ultralytics installed."""

    def __init__(self) -> None:
        super().__init__(
            "Ultralytics is not installed. Install the training extra first:\n"
            "  uv sync --extra train\n"
            "or:  pip install 'ultralytics>=8.3'"
        )


def build_train_command(config: TrainingConfig) -> list[str]:
    """The equivalent `yolo` CLI invocation, for users who prefer the terminal."""
    command = [
        "yolo",
        "detect",
        "train",
        f"model={config.model}",
        f"data={config.data}",
        f"epochs={config.epochs}",
        f"imgsz={config.imgsz}",
        f"batch={config.batch}",
        f"project={config.project}",
        f"name={config.name}",
        f"seed={config.seed}",
    ]
    if config.device:
        command.append(f"device={config.device}")
    return command


def read_results_csv(run_dir: Path) -> list[dict[str, float]]:
    """Parse Ultralytics' `results.csv` into per-epoch metric dicts.

    Returns an empty list while the file does not exist yet — a run that has not
    finished its first epoch simply has no metrics, which is not an error.
    """
    results_path = Path(run_dir) / "results.csv"
    if not results_path.is_file():
        return []

    rows: list[dict[str, float]] = []
    with results_path.open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, float] = {}
            for column, value in raw_row.items():
                if column is None or value is None:
                    continue
                key = column.strip()
                name = METRIC_COLUMNS.get(key, "epoch" if key == "epoch" else None)
                if name is None:
                    continue
                try:
                    row[name] = float(value)
                except ValueError:
                    continue
            if row:
                rows.append(row)
    return rows


def train_detector(config: TrainingConfig) -> TrainingOutcome:
    """Fine-tune a detector on a Baloot-class dataset.

    Raises:
        UltralyticsMissingError: if the training extra is not installed.
        FileNotFoundError: if the dataset config does not exist.
    """
    data_path = Path(config.data).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise UltralyticsMissingError() from error

    logger.info("Training %s on %s for %d epochs", config.model, data_path, config.epochs)
    model = YOLO(config.model)
    results = model.train(**config.to_kwargs())

    run_dir = Path(getattr(results, "save_dir", config.project / config.name))
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    history = read_results_csv(run_dir)

    return TrainingOutcome(
        run_dir=run_dir,
        best_weights=best if best.is_file() else None,
        last_weights=last if last.is_file() else None,
        metrics=history[-1] if history else {},
    )


__all__ = [
    "METRIC_COLUMNS",
    "SUPPORTED_MODELS",
    "TrainingConfig",
    "TrainingOutcome",
    "UltralyticsMissingError",
    "build_train_command",
    "read_results_csv",
    "train_detector",
]
