"""Localhost JSON API behind the browser studio.

The studio is a static page: opened from the filesystem it can inspect datasets
the user drags in, but it cannot train. Served by `hakim-vision studio` it also
gets this API, and the same buttons run real jobs on the machine.

Security posture: bound to the loopback interface, every mutating endpoint
requires a per-run token, and cross-origin requests are refused. This process
runs training commands, so it is not something to expose on a network.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from hakim_vision import __version__
from hakim_vision.datasets import discover_layout, inspect_dataset, suggest_mapping
from hakim_vision.datasets.yolo_layout import IMAGE_SUFFIXES
from hakim_vision.jobs import JobManager
from hakim_vision.models.train import SUPPORTED_MODELS, TrainingConfig, build_train_command
from hakim_vision.models.train import read_results_csv as read_metrics
from hakim_vision.models.yolo_export import get_baloot_classes

logger = logging.getLogger(__name__)

#: Largest JSON request body we will read, in bytes.
MAX_BODY_BYTES = 1_000_000
#: Header carrying the per-run token.
TOKEN_HEADER = "X-Hakim-Token"  # noqa: S105 - header name, not a secret


@dataclass
class StudioContext:
    """Shared state for the studio API."""

    web_dir: Path
    project_root: Path
    token: str
    jobs: JobManager
    #: Dataset roots the user has inspected; only these may be read back.
    allowed_roots: set[Path]

    def allow(self, root: Path) -> None:
        self.allowed_roots.add(root.resolve())

    def is_allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved.is_relative_to(root) for root in self.allowed_roots)


def _environment_status() -> dict[str, object]:
    """Report what is actually installed, without importing torch eagerly."""
    import importlib.util

    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # pragma: no cover - odd import states
            return False

    status: dict[str, object] = {
        "ultralytics": installed("ultralytics"),
        "torch": installed("torch"),
        "cuda": False,
        "device_name": "cpu",
    }
    if status["torch"]:
        try:
            import torch

            status["cuda"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                status["device_name"] = str(torch.cuda.get_device_name(0))
        except Exception as error:  # pragma: no cover - driver-dependent
            logger.debug("torch probe failed: %s", error)
    return status


class StudioHandler(SimpleHTTPRequestHandler):
    """Static files plus the `/api/*` endpoints."""

    context: StudioContext  # injected by `create_server`
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(self.context.web_dir), **kwargs)

    # -- plumbing ---------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed

    def _authorised(self) -> bool:
        """Reject cross-origin callers and anyone without this run's token."""
        origin = self.headers.get("Origin")
        if origin:
            host = self.headers.get("Host", "")
            if urlparse(origin).netloc != host:
                return False
        supplied = self.headers.get(TOKEN_HEADER, "")
        if not supplied:
            supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return secrets.compare_digest(supplied, self.context.token)

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if not route.path.startswith("/api/"):
            super().do_GET()
            return

        query = parse_qs(route.query)
        if route.path == "/api/status":
            self._send_json(self._status_payload())
            return

        if not self._authorised():
            self._error("Invalid or missing studio token.", HTTPStatus.FORBIDDEN)
            return

        if route.path == "/api/jobs":
            self._send_json({"jobs": [job.to_dict() for job in self.context.jobs.all()]})
            return

        if route.path.startswith("/api/jobs/"):
            self._job_detail(route.path.rsplit("/", 1)[-1])
            return

        if route.path == "/api/image":
            self._serve_dataset_image(query.get("path", [""])[0])
            return

        self._error("Unknown endpoint.", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = urlparse(self.path)
        if not self._authorised():
            self._error("Invalid or missing studio token.", HTTPStatus.FORBIDDEN)
            return

        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as error:
            self._error(f"Bad request body: {error}")
            return

        try:
            if route.path == "/api/dataset/inspect":
                self._inspect(payload)
            elif route.path == "/api/dataset/remap":
                self._remap(payload)
            elif route.path == "/api/train":
                self._train(payload)
            elif route.path == "/api/export":
                self._export(payload)
            elif route.path.startswith("/api/jobs/") and route.path.endswith("/stop"):
                job_id = route.path.split("/")[3]
                self._send_json({"stopped": self.context.jobs.stop(job_id)})
            else:
                self._error("Unknown endpoint.", HTTPStatus.NOT_FOUND)
        except FileNotFoundError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)
        except (ValueError, OSError) as error:
            self._error(str(error))

    # -- endpoints --------------------------------------------------------

    def _status_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": True,
            "version": __version__,
            "project_root": str(self.context.project_root),
            "models": list(SUPPORTED_MODELS),
            "baloot_classes": get_baloot_classes(),
            "authenticated": self._authorised(),
        }
        payload.update(_environment_status())
        return payload

    def _inspect(self, payload: dict[str, Any]) -> None:
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            raise ValueError("Provide the dataset folder path.")
        root = Path(raw_path).expanduser()
        layout = discover_layout(root)
        report = inspect_dataset(layout.root, layout=layout)
        self.context.allow(layout.root)
        self._send_json(
            {
                "report": report.to_dict(),
                "suggestions": [item.to_dict() for item in suggest_mapping(report.class_names)],
            }
        )

    def _remap(self, payload: dict[str, Any]) -> None:
        raw_path = str(payload.get("path", "")).strip()
        output = str(payload.get("output", "")).strip()
        mapping = payload.get("mapping")
        if not raw_path or not output:
            raise ValueError("Both 'path' and 'output' are required.")
        if not isinstance(mapping, dict):
            raise ValueError("'mapping' must be an object of source -> Baloot class.")

        source = Path(raw_path).expanduser()
        destination = Path(output).expanduser()
        mapping_file = destination.parent / f"{destination.name}.mapping.json"
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        mapping_file.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

        argv = self.context.jobs.cli_argv(
            "dataset",
            "remap",
            "--source",
            str(source),
            "--output",
            str(destination),
            "--mapping",
            str(mapping_file),
        )
        job = self.context.jobs.start("remap", argv, run_dir=destination)
        self.context.allow(destination)
        self._send_json({"job": job.to_dict()})

    def _train(self, payload: dict[str, Any]) -> None:
        data = str(payload.get("data", "")).strip()
        if not data:
            raise ValueError("Provide the dataset's data.yaml path.")
        data_path = Path(data).expanduser()
        if not data_path.is_file():
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

        model = str(payload.get("model", "yolo11n.pt"))
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model}'.")

        config = TrainingConfig(
            data=data_path,
            model=model,
            epochs=max(1, int(payload.get("epochs", 50))),
            imgsz=max(64, int(payload.get("imgsz", 640))),
            batch=max(1, int(payload.get("batch", 16))),
            device=str(payload.get("device", "")),
            project=Path(str(payload.get("project", self.context.project_root / "runs"))),
            name=str(payload.get("name", "baloot")),
        )

        argv = self.context.jobs.cli_argv(
            "train",
            "--data",
            str(config.data),
            "--model",
            config.model,
            "--epochs",
            str(config.epochs),
            "--imgsz",
            str(config.imgsz),
            "--batch",
            str(config.batch),
            "--project",
            str(config.project),
            "--name",
            config.name,
        )
        if config.device:
            argv.extend(["--device", config.device])

        job = self.context.jobs.start("train", argv, run_dir=config.project / config.name)
        self._send_json(
            {"job": job.to_dict(), "equivalent_cli": " ".join(build_train_command(config))}
        )

    def _export(self, payload: dict[str, Any]) -> None:
        weights = str(payload.get("weights", "")).strip()
        if not weights:
            raise ValueError("Provide the trained weights (.pt) path.")
        weights_path = Path(weights).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        output = str(payload.get("output", "")).strip()
        argv = self.context.jobs.cli_argv(
            "export", "--weights", str(weights_path), "--imgsz", str(int(payload.get("imgsz", 640)))
        )
        if output:
            argv.extend(["--output", str(Path(output).expanduser())])
        job = self.context.jobs.start("export", argv)
        self._send_json({"job": job.to_dict()})

    def _job_detail(self, job_id: str) -> None:
        job = self.context.jobs.get(job_id)
        if job is None:
            self._error("No such job.", HTTPStatus.NOT_FOUND)
            return
        payload = job.to_dict()
        payload["log"] = job.tail()
        payload["metrics"] = read_metrics(job.run_dir) if job.run_dir else []
        self._send_json(payload)

    def _serve_dataset_image(self, raw_path: str) -> None:
        """Return one image from a dataset the user has inspected."""
        if not raw_path:
            self._error("Missing 'path'.")
            return
        path = Path(raw_path).expanduser()
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            self._error("Not an image path.")
            return
        if not self.context.is_allowed(path) or not path.is_file():
            self._error("Path is outside any inspected dataset.", HTTPStatus.FORBIDDEN)
            return

        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"image/{path.suffix.lstrip('.').replace('jpg', 'jpeg')}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def create_server(
    web_dir: Path,
    port: int,
    project_root: Path,
    token: str | None = None,
) -> tuple[ThreadingHTTPServer, StudioContext]:
    """Build a loopback-only studio server. Caller runs `serve_forever()`."""
    context = StudioContext(
        web_dir=Path(web_dir),
        project_root=Path(project_root),
        token=token or secrets.token_urlsafe(24),
        jobs=JobManager(cwd=project_root),
        allowed_roots=set(),
    )

    handler = type("BoundStudioHandler", (StudioHandler,), {"context": context})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server, context


__all__ = ["MAX_BODY_BYTES", "TOKEN_HEADER", "StudioContext", "StudioHandler", "create_server"]
