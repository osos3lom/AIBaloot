from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hakim_vision.models import export_yolo_to_onnx, get_baloot_classes
from hakim_vision.models.evaluate_parity import (
    ModelEvalResult,
    compute_iou,
    decode_yolo_tensor,
    nms,
)
from hakim_vision.models.quantize import (
    BalootCalibrationDataReader,
    _decode_tail_nodes,
    _reject_non_float32_model,
    preprocess_image_stretch,
)
from hakim_vision.models.yolo_export import (
    generate_dataset_yaml,
    get_baloot_detection_classes,
)


def test_get_baloot_classes() -> None:
    cls = get_baloot_classes(include_other=False)
    assert len(cls) == 32
    assert "Ah" in cls
    assert "7s" in cls
    assert "other" not in cls

    cls_33 = get_baloot_classes(include_other=True)
    assert len(cls_33) == 33
    assert cls_33[-1] == "other"

    det_cls = get_baloot_detection_classes()
    assert len(det_cls) == 33
    assert det_cls[-1] == "other"


def test_generate_dataset_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "dataset.yaml"
    generate_dataset_yaml(yaml_file, dataset_root=tmp_path, include_other=True)
    assert yaml_file.exists()
    content = yaml_file.read_text(encoding="utf-8")
    assert "32: other" in content


def test_export_yolo_to_onnx_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_yolo_to_onnx(model_path=tmp_path / "non_existent.pt")


def test_reject_non_float32_model(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    # Build a mock ONNX graph with a float16 initializer
    tensor_fp16 = helper.make_tensor("w_fp16", TensorProto.FLOAT16, [1], [0])
    graph_fp16 = helper.make_graph([], "fp16_graph", [], [], [tensor_fp16])
    model_fp16 = helper.make_model(graph_fp16)
    fp16_file = tmp_path / "model_fp16.onnx"
    onnx.save(model_fp16, str(fp16_file))

    with pytest.raises(ValueError, match="float16 initializers"):
        _reject_non_float32_model(fp16_file)

    # Build a mock ONNX graph with a float32 initializer
    tensor_fp32 = helper.make_tensor("w_fp32", TensorProto.FLOAT, [1], [0.0])
    graph_fp32 = helper.make_graph([], "fp32_graph", [], [], [tensor_fp32])
    model_fp32 = helper.make_model(graph_fp32)
    fp32_file = tmp_path / "model_fp32.onnx"
    onnx.save(model_fp32, str(fp32_file))

    # Should not raise
    _reject_non_float32_model(fp32_file)


def test_decode_tail_nodes_stops_at_first_conv(tmp_path: Path) -> None:
    """The decode tail must stay float32, or class scores share a scale with box coords.

    Graph: Conv -> Sigmoid -> Concat -> output. Everything from the output back to and
    including the Conv is excluded; the Conv feeding it is the boundary and nothing
    upstream of it is.
    """
    import onnx
    from onnx import TensorProto, helper

    weight = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    nodes = [
        helper.make_node("Conv", ["stem_out", "w"], ["conv_out"], name="/head/Conv"),
        helper.make_node("Relu", ["images", "w"], ["stem_out"], name="/stem/Relu"),
        helper.make_node("Sigmoid", ["conv_out"], ["sig_out"], name="/head/Sigmoid"),
        helper.make_node(
            "Concat", ["conv_out", "sig_out"], ["output0"], name="/head/Concat", axis=1
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "tail_graph",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 2, 1, 1])],
        [weight],
    )
    model_file = tmp_path / "tail.onnx"
    onnx.save(helper.make_model(graph), str(model_file))

    excluded = set(_decode_tail_nodes(model_file))

    assert excluded == {"/head/Concat", "/head/Sigmoid", "/head/Conv"}
    assert "/stem/Relu" not in excluded


def test_preprocess_and_calibration_reader(tmp_path: Path) -> None:
    import cv2

    img_path = tmp_path / "sample.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = (255, 128, 64)
    cv2.imwrite(str(img_path), img)

    tensor = preprocess_image_stretch(img_path, imgsz=416)
    assert tensor.shape == (1, 3, 416, 416)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0

    reader = BalootCalibrationDataReader([img_path], imgsz=416)
    batch = reader.get_next()
    assert batch is not None
    assert "images" in batch
    assert batch["images"].shape == (1, 3, 416, 416)

    # Exhausted reader returns None
    assert reader.get_next() is None
    # Rewind resets reader
    reader.rewind()
    assert reader.get_next() is not None


def test_evaluate_parity_utilities() -> None:
    # 1. IoU tests
    box_a = np.array([0.0, 0.0, 10.0, 10.0])
    box_b = np.array([0.0, 0.0, 10.0, 10.0])
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    box_c = np.array([5.0, 0.0, 15.0, 10.0])
    assert compute_iou(box_a, box_c) == pytest.approx(50.0 / 150.0)

    box_d = np.array([20.0, 20.0, 30.0, 30.0])
    assert compute_iou(box_a, box_d) == 0.0

    # 2. NMS tests
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]])
    scores = np.array([0.9, 0.8, 0.95])
    classes = np.array([0, 0, 1])
    k_boxes, _scores, _classes = nms(boxes, scores, classes, iou_thresh=0.45)
    assert len(k_boxes) == 2  # The overlapping box with score 0.8 is suppressed

    # 3. Decode YOLO tensor test
    mock_tensor = np.zeros((37, 3549), dtype=np.float32)
    # First anchor: cx=100, cy=100, w=50, h=50, class 0 has prob 0.9
    mock_tensor[0, 0] = 100.0
    mock_tensor[1, 0] = 100.0
    mock_tensor[2, 0] = 50.0
    mock_tensor[3, 0] = 50.0
    mock_tensor[4 + 0, 0] = 0.9

    pred_b, pred_s, pred_c = decode_yolo_tensor(mock_tensor, imgsz=416, conf_thresh=0.5)
    assert len(pred_b) == 1
    assert pred_c[0] == 0
    assert pred_s[0] == pytest.approx(0.9)
    assert pred_b[0].tolist() == [75.0, 75.0, 125.0, 125.0]

    # 4. ModelEvalResult serialization
    res = ModelEvalResult(
        model_name="test.onnx",
        format="onnx",
        precision=0.99,
        recall=0.98,
        map50=0.985,
        map50_95=0.88,
        avg_latency_ms=12.34,
        model_size_mb=5.12,
    )
    d = res.to_dict()
    assert d["map50"] == 0.985
    assert d["format"] == "onnx"
