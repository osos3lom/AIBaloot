"""Static INT8 ONNX Quantization for YOLO Card Detector."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_image_stretch(image_path: Path, imgsz: int = 416) -> np.ndarray:
    """Preprocess image by stretch-resizing to imgsz x imgsz (RGB float32 normalized [0, 1])."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Stretch resize (aspect ratio not preserved, matching Roboflow stretch preprocessing)
    resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    chw = np.transpose(tensor, (2, 0, 1))  # (3, imgsz, imgsz)
    return np.expand_dims(chw, axis=0)  # (1, 3, imgsz, imgsz)


class BalootCalibrationDataReader:
    """Calibration data reader for onnxruntime static quantization."""

    def __init__(
        self,
        image_paths: list[Path],
        input_name: str = "images",
        imgsz: int = 416,
    ) -> None:
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.imgsz = imgsz
        self._iterator: Iterator[Path] = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            path = next(self._iterator)
        except StopIteration:
            return None

        tensor = preprocess_image_stretch(path, imgsz=self.imgsz)
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self._iterator = iter(self.image_paths)


def _reject_non_float32_model(model_path: Path) -> None:
    """Quantizing an fp16 graph yields silently wrong dequantization scales.

    Calibration assumes float32 tensors, so an fp16 input produces a model that runs and
    looks plausible but emits class scores outside [0, 1]. Fail loudly instead.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(model_path), load_external_data=False)
    float16_count = sum(
        1 for tensor in model.graph.initializer if tensor.data_type == TensorProto.FLOAT16
    )
    if float16_count:
        raise ValueError(
            f"{model_path} holds {float16_count} float16 initializers. Static quantization "
            "requires a float32 model - export with half=False and quantize from that."
        )


def _decode_tail_nodes(model_path: Path) -> list[str]:
    """Names of the detection head's decode ops, which must stay float32.

    The graph output is a Concat of box coordinates (0..imgsz) and class probabilities
    (0..1). One shared activation scale cannot represent both: the scale is set by the
    coordinate range, so every probability quantizes to zero (or, at smaller imgsz, to a
    single step above 1.0). Excluding the decode tail keeps the Conv weights quantized -
    where the size saving actually comes from - while the arithmetic after them runs in
    float.

    The head's final Convs are excluded too. They emit the raw class logits, whose range
    is set by the most negative background logit across the calibration set; quantizing
    them to 8 bits costs enough resolution to push peak confidences from ~0.89 to ~0.06.
    They are three small Convs, so keeping them float barely moves the file size.

    Derived from the graph rather than hardcoded so it survives a re-export: walk back
    from the outputs, collecting every node up to and including the first Conv on each
    path.
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    producer = {out: node for node in model.graph.node for out in node.output}

    excluded: list[str] = []
    seen: set[str] = set()
    frontier = [output.name for output in model.graph.output]

    while frontier:
        node = producer.get(frontier.pop())
        if node is None or node.name in seen:
            continue
        seen.add(node.name)
        excluded.append(node.name)
        if node.op_type != "Conv":
            frontier.extend(node.input)

    return excluded


def quantize_onnx_static(
    input_model: Path | str,
    output_model: Path | str,
    calibration_images: list[Path],
    imgsz: int = 416,
    input_name: str = "images",
) -> Path:
    """Perform static INT8 quantization on an ONNX model using representative calibration images."""
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is not installed. Install with `uv sync --extra train`."
        ) from error

    input_path = Path(input_model)
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not calibration_images:
        raise ValueError("At least 1 calibration image is required for static quantization.")

    _reject_non_float32_model(input_path)

    reader = BalootCalibrationDataReader(
        image_paths=calibration_images,
        input_name=input_name,
        imgsz=imgsz,
    )

    excluded_nodes = _decode_tail_nodes(input_path)

    logger.info(
        "Quantizing %s -> %s (static INT8 with %d calibration images, "
        "%d decode-tail nodes left in float32)",
        input_path,
        output_path,
        len(calibration_images),
        len(excluded_nodes),
    )

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        nodes_to_exclude=excluded_nodes,
        # The u8s8 convolution kernels accumulate into int16, which saturates on any
        # target without VNNI - including the WASM SIMD build this model is for. The
        # saturation is silent: the graph runs, boxes stay plausible, and class scores
        # collapse to ~0.05 instead of ~0.90. Halving the activation range avoids it at
        # a cost of one bit.
        reduce_range=True,
    )

    logger.info("Static INT8 ONNX model written to %s", output_path)
    return output_path


__all__ = [
    "BalootCalibrationDataReader",
    "preprocess_image_stretch",
    "quantize_onnx_static",
]
