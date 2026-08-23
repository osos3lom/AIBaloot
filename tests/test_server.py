"""Tests for the studio's local job API."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from hakim_vision.jobs import STATUS_DONE, STATUS_FAILED, JobManager
from hakim_vision.models.train import (
    TrainingConfig,
    UltralyticsMissingError,
    build_train_command,
    read_results_csv,
)
from hakim_vision.server import TOKEN_HEADER, create_server


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    """Poll until `predicate()` is true. Jobs are subprocesses, so this is I/O."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ---- jobs -----------------------------------------------------------------


def test_job_manager_captures_output_and_exit_code(tmp_path: Path):
    manager = JobManager(cwd=tmp_path)
    job = manager.start("test", [sys.executable, "-c", "print('hello studio')"])

    assert _wait_for(lambda: job.status != "running")
    assert job.status == STATUS_DONE
    assert job.exit_code == 0
    assert any("hello studio" in line for line in job.tail())


def test_job_manager_marks_a_failing_command_failed(tmp_path: Path):
    manager = JobManager(cwd=tmp_path)
    job = manager.start("test", [sys.executable, "-c", "raise SystemExit(3)"])

    assert _wait_for(lambda: job.status != "running")
    assert job.status == STATUS_FAILED
    assert job.exit_code == 3


def test_job_manager_stops_a_running_job(tmp_path: Path):
    manager = JobManager(cwd=tmp_path)
    job = manager.start("test", [sys.executable, "-c", "import time; time.sleep(30)"])

    assert manager.stop(job.id) is True
    assert _wait_for(lambda: job.finished_at is not None)
    assert job.status == "stopped"
    assert manager.stop(job.id) is False


def test_job_log_is_bounded(tmp_path: Path):
    manager = JobManager(cwd=tmp_path)
    job = manager.start("test", [sys.executable, "-c", "for i in range(2000): print(i)"])
    assert _wait_for(lambda: job.status != "running")
    assert len(job.log) <= job.log.maxlen  # type: ignore[operator]


def test_cli_argv_uses_the_running_interpreter():
    argv = JobManager.cli_argv("version")
    assert argv[:3] == [sys.executable, "-m", "hakim_vision.cli"]


# ---- training helpers -----------------------------------------------------


def test_build_train_command_reflects_the_config():
    command = build_train_command(
        TrainingConfig(data=Path("d.yaml"), model="yolo11s.pt", epochs=7, device="0")
    )
    assert "model=yolo11s.pt" in command
    assert "epochs=7" in command
    assert "device=0" in command


def test_read_results_csv_returns_nothing_before_the_first_epoch(tmp_path: Path):
    assert read_results_csv(tmp_path) == []


def test_read_results_csv_maps_ultralytics_columns(tmp_path: Path):
    (tmp_path / "results.csv").write_text(
        "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),junk\n"
        "1,1.5,0.42,0.31,x\n"
        "2,1.1,0.55,0.40,y\n",
        encoding="utf-8",
    )
    rows = read_results_csv(tmp_path)
    assert [row["epoch"] for row in rows] == [1.0, 2.0]
    assert rows[-1]["map50"] == 0.55
    assert "junk" not in rows[0]


def test_train_detector_rejects_a_missing_dataset(tmp_path: Path):
    from hakim_vision.models.train import train_detector

    with pytest.raises(FileNotFoundError):
        train_detector(TrainingConfig(data=tmp_path / "missing.yaml"))


def test_ultralytics_missing_error_explains_the_fix():
    assert "uv sync --extra train" in str(UltralyticsMissingError())


# ---- HTTP API -------------------------------------------------------------


@pytest.fixture
def studio(tmp_path: Path) -> Iterator[tuple[str, str, Path]]:
    """A running studio server on an ephemeral port."""
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "studio.html").write_text("<h1>studio</h1>", encoding="utf-8")

    server, context = create_server(web_dir, 0, tmp_path, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, context.token, tmp_path
    finally:
        context.jobs.stop_all()
        server.shutdown()
        server.server_close()


def _request(
    url: str, *, token: str | None = None, payload: dict | None = None, origin: str | None = None
):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header(TOKEN_HEADER, token)
    if origin:
        request.add_header("Origin", origin)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode())


def test_status_is_public_and_reports_the_environment(studio):
    base, _, _ = studio
    status, payload = _request(f"{base}/api/status")

    assert status == 200
    assert payload["backend"] is True
    assert payload["authenticated"] is False
    assert len(payload["baloot_classes"]) == 32
    assert "ultralytics" in payload


def test_status_reports_authenticated_with_a_token(studio):
    base, token, _ = studio
    _, payload = _request(f"{base}/api/status", token=token)
    assert payload["authenticated"] is True


def test_endpoints_reject_a_missing_or_wrong_token(studio):
    base, _, _ = studio
    for url in (f"{base}/api/jobs", f"{base}/api/dataset/inspect"):
        with pytest.raises(urllib.error.HTTPError) as error:
            _request(url, payload={"path": "."} if "inspect" in url else None)
        assert error.value.code == 403

    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/jobs", token="wrong")
    assert error.value.code == 403


def test_endpoints_reject_a_foreign_origin(studio):
    base, token, _ = studio
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/jobs", token=token, origin="https://evil.example")
    assert error.value.code == 403


def test_inspect_endpoint_returns_a_report_and_suggestions(studio):
    base, token, root = studio
    dataset = root / "cards"
    (dataset / "train" / "images").mkdir(parents=True)
    (dataset / "train" / "labels").mkdir(parents=True)
    (dataset / "data.yaml").write_text("names: ['Ah', '2c']\n", encoding="utf-8")
    (dataset / "train" / "images" / "a.jpg").write_text("x", encoding="utf-8")
    (dataset / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _, payload = _request(
        f"{base}/api/dataset/inspect", token=token, payload={"path": str(dataset)}
    )
    assert payload["report"]["totals"]["images"] == 1
    targets = {item["source_name"]: item["target"] for item in payload["suggestions"]}
    assert targets == {"Ah": "Ah", "2c": None}


def test_inspect_endpoint_reports_a_missing_folder(studio):
    base, token, root = studio
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/dataset/inspect", token=token, payload={"path": str(root / "nope")})
    assert error.value.code == 404


def test_train_endpoint_rejects_an_unknown_model(studio):
    base, token, root = studio
    data_yaml = root / "data.yaml"
    data_yaml.write_text("names: []\n", encoding="utf-8")

    with pytest.raises(urllib.error.HTTPError) as error:
        _request(
            f"{base}/api/train",
            token=token,
            payload={"data": str(data_yaml), "model": "definitely-not-a-model.pt"},
        )
    assert error.value.code == 400


def test_train_endpoint_rejects_a_missing_data_yaml(studio):
    base, token, root = studio
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/train", token=token, payload={"data": str(root / "nope.yaml")})
    assert error.value.code == 404


def test_image_endpoint_refuses_paths_outside_inspected_datasets(studio):
    base, token, root = studio
    secret = root / "secret.png"
    secret.write_text("not really an image", encoding="utf-8")

    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/image?path={secret}", token=token)
    assert error.value.code == 403


def test_static_files_are_still_served(studio):
    base, _, _ = studio
    with urllib.request.urlopen(f"{base}/studio.html", timeout=10) as response:
        assert "studio" in response.read().decode()


def _dataset(root: Path) -> Path:
    """A minimal inspectable dataset."""
    dataset = root / "cards"
    (dataset / "train" / "images").mkdir(parents=True, exist_ok=True)
    (dataset / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset / "data.yaml").write_text("names: ['Ah', '2c']\n", encoding="utf-8")
    (dataset / "train" / "images" / "a.jpg").write_bytes(
        bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b" stub jpeg"
    )
    (dataset / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return dataset


def test_remap_endpoint_starts_a_job_and_records_the_mapping(studio):
    base, token, root = studio
    dataset = _dataset(root)
    output = root / "baloot"

    _, payload = _request(
        f"{base}/api/dataset/remap",
        token=token,
        payload={"path": str(dataset), "output": str(output), "mapping": {"Ah": "Ah", "2c": None}},
    )
    assert payload["job"]["kind"] == "remap"
    assert (root / "baloot.mapping.json").is_file()

    _, jobs = _request(f"{base}/api/jobs", token=token)
    assert any(job["id"] == payload["job"]["id"] for job in jobs["jobs"])


def test_remap_endpoint_requires_a_mapping_object(studio):
    base, token, root = studio
    dataset = _dataset(root)
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(
            f"{base}/api/dataset/remap",
            token=token,
            payload={"path": str(dataset), "output": str(root / "out"), "mapping": "nope"},
        )
    assert error.value.code == 400


def test_export_endpoint_rejects_missing_weights(studio):
    base, token, root = studio
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/export", token=token, payload={"weights": str(root / "nope.pt")})
    assert error.value.code == 404


def test_export_endpoint_starts_a_job_for_real_weights(studio):
    base, token, root = studio
    weights = root / "best.pt"
    weights.write_text("stub", encoding="utf-8")
    _, payload = _request(f"{base}/api/export", token=token, payload={"weights": str(weights)})
    assert payload["job"]["kind"] == "export"


def test_image_endpoint_serves_files_from_an_inspected_dataset(studio):
    base, token, root = studio
    dataset = _dataset(root)
    _request(f"{base}/api/dataset/inspect", token=token, payload={"path": str(dataset)})

    image = dataset / "train" / "images" / "a.jpg"
    request = urllib.request.Request(f"{base}/api/image?path={image}")
    request.add_header(TOKEN_HEADER, token)
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.read()


def test_image_endpoint_rejects_non_image_paths(studio):
    base, token, root = studio
    dataset = _dataset(root)
    _request(f"{base}/api/dataset/inspect", token=token, payload={"path": str(dataset)})
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/image?path={dataset / 'data.yaml'}", token=token)
    assert error.value.code == 400


def test_unknown_endpoints_return_404(studio):
    base, token, _ = studio
    for method_payload in (None, {}):
        with pytest.raises(urllib.error.HTTPError) as error:
            _request(f"{base}/api/nope", token=token, payload=method_payload)
        assert error.value.code == 404


def test_malformed_json_bodies_are_rejected(studio):
    base, token, _ = studio
    request = urllib.request.Request(f"{base}/api/train", data=b"{not json", method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header(TOKEN_HEADER, token)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 400


def test_stopping_an_unknown_job_reports_false(studio):
    base, token, _ = studio
    _, payload = _request(f"{base}/api/jobs/deadbeef/stop", token=token, payload={})
    assert payload["stopped"] is False


def test_job_detail_for_an_unknown_id_is_404(studio):
    base, token, _ = studio
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/api/jobs/deadbeef", token=token)
    assert error.value.code == 404
