"""Smoke test on real photos in data/test-photos/"""

from __future__ import annotations

from pathlib import Path

import cv2
import onnxruntime as ort

from hakim_vision.models.evaluate_parity import decode_yolo_tensor
from hakim_vision.models.quantize import preprocess_image_stretch
from hakim_vision.models.yolo_export import get_baloot_detection_classes


def run_smoke_test_photo(
    onnx_path: Path,
    photo_path: Path,
    imgsz: int = 704,
    conf_thresh: float = 0.3,
) -> None:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    class_names = get_baloot_detection_classes()

    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"Error loading {photo_path}")
        return

    h_orig, w_orig = img.shape[:2]
    tensor = preprocess_image_stretch(photo_path, imgsz=imgsz)

    outputs = session.run([output_name], {input_name: tensor})
    boxes, scores, class_ids = decode_yolo_tensor(outputs[0], imgsz=imgsz, conf_thresh=conf_thresh)

    scale_x = w_orig / imgsz
    scale_y = h_orig / imgsz

    detected_cards = []
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        cid = class_ids[i]
        cname = class_names[cid]
        x1, y1, x2, y2 = box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
        detected_cards.append(
            (cname, float(score), (round(x1), round(y1), round(x2 - x1), round(y2 - y1)))
        )

    print(f"\nPhoto: {photo_path.name} ({w_orig}x{h_orig})")
    print(f"Detected {len(detected_cards)} cards:")
    for cname, score, b in detected_cards:
        print(f"  - Card: {cname:6s} | Confidence: {score * 100:.1f}% | Box: {b}")


if __name__ == "__main__":
    fp16_model = Path("web/models/baloot-v1.fp16.onnx")
    for photo in Path("data/test-photos").glob("*.jpg"):
        run_smoke_test_photo(fp16_model, photo)
