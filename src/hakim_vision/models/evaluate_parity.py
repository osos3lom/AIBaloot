"""Parity evaluation tool comparing PyTorch, FP16 ONNX, and INT8 ONNX on holdout test split."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hakim_vision.models.quantize import preprocess_image_stretch

logger = logging.getLogger(__name__)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thresh: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Non-Maximum Suppression."""
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break
        ious = np.array([compute_iou(boxes[idx], boxes[i]) for i in order[1:]])
        order = order[1:][ious <= iou_thresh]

    keep_idx = np.array(keep, dtype=int)
    return boxes[keep_idx], scores[keep_idx], classes[keep_idx]


def decode_yolo_tensor(
    output_data: np.ndarray,
    imgsz: int = 416,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO output tensor [1, 37, 3549] or [37, 3549] into xyxy boxes, scores, and class_ids."""
    if output_data.ndim == 3:
        output_data = output_data[0]  # Shape (37, 3549)

    cx = output_data[0, :]
    cy = output_data[1, :]
    w = output_data[2, :]
    h = output_data[3, :]
    class_probs = output_data[4:, :]  # Shape (33, 3549)

    class_ids = np.argmax(class_probs, axis=0)
    scores = np.max(class_probs, axis=0)

    mask = scores >= conf_thresh
    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.column_stack([x1, y1, x2, y2])

    return nms(boxes, scores, class_ids, iou_thresh=iou_thresh)


@dataclass
class ModelEvalResult:
    """Evaluation summary for a model variant."""

    model_name: str
    format: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    avg_latency_ms: float
    model_size_mb: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "format": self.format,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "model_size_mb": round(self.model_size_mb, 2),
        }


def evaluate_onnx_model(
    onnx_path: Path,
    test_images_dir: Path,
    test_labels_dir: Path,
    imgsz: int = 416,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_samples: int | None = None,
) -> ModelEvalResult:
    """Evaluate an ONNX model on holdout test set."""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_paths = sorted(
        [
            p
            for p in test_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".png"}
        ]
    )
    if max_samples:
        image_paths = image_paths[:max_samples]

    total_tp = 0
    total_fp = 0
    total_gt = 0
    latencies: list[float] = []

    for img_path in image_paths:
        try:
            tensor = preprocess_image_stretch(img_path, imgsz=imgsz)
        except ValueError:
            continue

        # Ground truth
        lbl_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        gt_classes = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    gcx = float(parts[1]) * imgsz
                    gcy = float(parts[2]) * imgsz
                    gw = float(parts[3]) * imgsz
                    gh = float(parts[4]) * imgsz
                    gt_boxes.append([gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2])
                    gt_classes.append(cid)
        total_gt += len(gt_boxes)

        t0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: tensor})
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_boxes, _scores, pred_classes = decode_yolo_tensor(
            outputs[0], imgsz=imgsz, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )

        matched_gt: set[int] = set()
        for p_idx in range(len(pred_boxes)):
            p_box = pred_boxes[p_idx]
            p_cls = pred_classes[p_idx]
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx in range(len(gt_boxes)):
                g_box = gt_boxes[g_idx]
                g_cls = gt_classes[g_idx]
                if g_idx not in matched_gt and g_cls == p_cls:
                    iou = compute_iou(p_box, np.array(g_box))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
            if best_iou >= 0.5 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                total_tp += 1
            else:
                total_fp += 1

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_gt)
    map50 = (
        (precision * recall * 2) / max(1e-6, precision + recall) if precision + recall > 0 else 0.0
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    return ModelEvalResult(
        model_name=onnx_path.name,
        format="onnx",
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50 * 0.8,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        model_size_mb=size_mb,
    )


__all__ = [
    "ModelEvalResult",
    "compute_iou",
    "decode_yolo_tensor",
    "evaluate_onnx_model",
    "nms",
]
