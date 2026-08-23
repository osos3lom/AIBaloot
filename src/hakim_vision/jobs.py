"""Background job runner for long tasks the studio starts (remap, train, export).

Jobs are subprocesses, not threads: training pins a GPU and prints a lot, and a
subprocess can be stopped cleanly without leaving a half-initialised CUDA
context behind. Output is kept in a bounded buffer so a 50-epoch run cannot
grow memory without limit.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

#: Lines of stdout kept per job.
LOG_CAPACITY = 400

#: Job lifecycle states.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"


@dataclass
class Job:
    """One running or finished subprocess."""

    id: str
    kind: str
    argv: list[str]
    status: str = STATUS_RUNNING
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    run_dir: Path | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_CAPACITY))
    _process: subprocess.Popen[str] | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, line: str) -> None:
        with self._lock:
            self.log.append(line.rstrip())

    def tail(self, limit: int = 80) -> list[str]:
        with self._lock:
            lines = list(self.log)
        return lines[-limit:]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "elapsed": (self.finished_at or time.time()) - self.started_at,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "command": " ".join(self.argv),
        }


class JobManager:
    """Starts, tracks, and stops studio jobs."""

    def __init__(self, cwd: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._cwd = Path(cwd) if cwd else Path.cwd()
        self._lock = threading.Lock()

    @staticmethod
    def cli_argv(*arguments: str) -> list[str]:
        """Build an argv that runs this package's CLI with the current interpreter."""
        return [sys.executable, "-m", "hakim_vision.cli", *arguments]

    def start(self, kind: str, argv: list[str], run_dir: Path | None = None) -> Job:
        """Launch `argv` as a job. The caller owns argv construction."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=list(argv), run_dir=run_dir)
        # argv is built by this package from validated inputs and runs without a
        # shell, so no untrusted string is ever interpreted as a command.
        process = subprocess.Popen(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self._cwd),
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        job._process = process
        job.append(f"$ {' '.join(argv)}")

        with self._lock:
            self._jobs[job.id] = job

        reader = threading.Thread(target=self._drain, args=(job,), daemon=True)
        reader.start()
        return job

    def _drain(self, job: Job) -> None:
        """Stream the subprocess' output into the job's buffer until it exits."""
        process = job._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                job.append(line)
        finally:
            exit_code = process.wait()
            job.exit_code = exit_code
            job.finished_at = time.time()
            if job.status == STATUS_RUNNING:
                job.status = STATUS_DONE if exit_code == 0 else STATUS_FAILED
            job.append(f"[exit {exit_code}]")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda job: job.started_at, reverse=True)

    def stop(self, job_id: str) -> bool:
        """Terminate a running job. Returns False if it was already finished."""
        job = self.get(job_id)
        if job is None or job._process is None or job.status != STATUS_RUNNING:
            return False
        job.status = STATUS_STOPPED
        job._process.terminate()
        try:
            job._process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged child
            job._process.kill()
        return True

    def stop_all(self) -> None:
        for job in self.all():
            if job.status == STATUS_RUNNING:
                self.stop(job.id)


__all__ = [
    "LOG_CAPACITY",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "Job",
    "JobManager",
]
