"""Parity Evaluation across PyTorch, FP16 ONNX, and INT8 ONNX on holdout test set."""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from hakim_vision.models.evaluate_parity import evaluate_onnx_model


def main() -> None:
    print("=" * 60)
    print("1. Evaluating PyTorch weights (best.pt) on test holdout...")
    print("=" * 60)
    weights_file = Path("runs/detect/runs/hakim/baloot/weights/best.pt")
    if weights_file.is_file():
        model = YOLO(str(weights_file))
        val_res = model.val(data="data/baloot-dataset/data.yaml", split="test", imgsz=416, device=0)
        pt_p = val_res.results_dict.get("metrics/precision(B)", 0.0)
        pt_r = val_res.results_dict.get("metrics/recall(B)", 0.0)
        pt_map50 = val_res.results_dict.get("metrics/mAP50(B)", 0.0)
        pt_map = val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)
        print(
            f"PyTorch Best: Precision={pt_p:.4f}, Recall={pt_r:.4f}, mAP@50={pt_map50:.4f}, mAP@50-95={pt_map:.4f}"
        )
    else:
        print("Skipping PyTorch eval (checkpoint not present locally).")

    test_imgs = Path("data/baloot-dataset/images/test")
    test_lbls = Path("data/baloot-dataset/labels/test")

    if not test_imgs.is_dir() or not test_lbls.is_dir():
        print("Skipping ONNX eval (test images/labels not present).")
        return

    print("=" * 60)
    print("2. Evaluating FP16 ONNX model on holdout test set...")
    print("=" * 60)
    fp16_path = Path("web/models/baloot-v1.fp16.onnx")
    res_fp16 = evaluate_onnx_model(fp16_path, test_imgs, test_lbls, imgsz=416)
    print(json.dumps(res_fp16.to_dict(), indent=2))

    print("=" * 60)
    print("3. Evaluating INT8 ONNX model on holdout test set...")
    print("=" * 60)
    int8_path = Path("web/models/baloot-v1.int8.onnx")
    res_int8 = evaluate_onnx_model(int8_path, test_imgs, test_lbls, imgsz=416)
    print(json.dumps(res_int8.to_dict(), indent=2))

    print("=" * 60)
    print("PARITY SUMMARY")
    print("=" * 60)
    print(
        f"FP16: mAP@50 = {res_fp16.map50:.4f}, Latency = {res_fp16.avg_latency_ms:.2f} ms, Size = {res_fp16.model_size_mb:.2f} MB"
    )
    print(
        f"INT8: mAP@50 = {res_int8.map50:.4f}, Latency = {res_int8.avg_latency_ms:.2f} ms, Size = {res_int8.model_size_mb:.2f} MB"
    )

    delta_map50 = abs(res_fp16.map50 - res_int8.map50)
    print(f"Delta (FP16 vs INT8 mAP@50): {delta_map50:.4f}")
    if delta_map50 <= 0.05:
        print(
            "[PASS] INT8 is within parity threshold! Both FP16 and INT8 models are qualified for deployment."
        )
    else:
        print("[NOTICE] INT8 delta is larger; shipping FP16 as primary.")


if __name__ == "__main__":
    main()
