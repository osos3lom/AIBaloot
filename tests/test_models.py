from pathlib import Path

from hakim_vision.models import export_yolo_to_onnx, get_baloot_classes
from hakim_vision.models.yolo_export import generate_dataset_yaml


def test_get_baloot_classes():
    cls = get_baloot_classes()
    assert len(cls) == 32
    assert "Ah" in cls
    assert "7s" in cls


def test_generate_dataset_yaml(tmp_path: Path):
    yaml_file = tmp_path / "dataset.yaml"
    generate_dataset_yaml(yaml_file, dataset_root=tmp_path)
    assert yaml_file.exists()
    content = yaml_file.read_text()
    assert "2: Ah" in content or "0: Ahs" in content or "Ah" in content
    assert "31" in content


def test_export_yolo_to_onnx_placeholder(tmp_path: Path):
    model_pt = tmp_path / "model.pt"
    model_pt.touch()
    onnx_out = tmp_path / "model.ongx"
    result = export_yolo_to_onnx(
        model_path=model_pt,
        output_path=onnx_out,
        imgsz=640,
    )
    assert result.exists()
