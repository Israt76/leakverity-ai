"""Run untrusted experiment code in a restricted subprocess.

Isolation used on hosted demos (where containers are not available):
  * separate process, isolated interpreter (-I), throwaway working directory
  * network sockets disabled inside the process
  * CPU-time, address-space and file-size limits; wall-clock timeout
  * minimal environment (no private environment variables or secrets are passed through)
For production, run the same entrypoint inside a disposable container with
`--network none` (see Dockerfile.sandbox and docs/security.md).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TRACER = Path(__file__).with_name("tracer.py")


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    events: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    design: pd.DataFrame | None = None
    error: str | None = None
    workdir: str = ""
    seconds: float = 0.0

    def metrics(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "metric"]

    def fits(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "fit"]

    def split(self) -> dict | None:
        seq = self.summary.get("primary_split_seq")
        return next((e for e in self.events if e["type"] == "split" and e["seq"] == seq), None)

    def loads(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "load"]


def _limits(cpu_seconds: int, mem_bytes: int):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 2**20, 512 * 2**20))
            os.setsid()
        except Exception:  # noqa: BLE001
            pass
    return apply


def run_code(code: str, data_files: dict[str, bytes | str], timeout: int = 300,
             mem_gb: float = 6.0, keep: bool = False) -> RunResult:
    import time
    work = tempfile.mkdtemp(prefix="lv_run_")
    try:
        paths = []
        for name, content in data_files.items():
            p = os.path.join(work, os.path.basename(name))
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(p, mode) as f:
                f.write(content)
            paths.append(p)
        code_path = os.path.join(work, "experiment.py")
        with open(code_path, "w") as f:
            f.write(code)
        docker = os.getenv("LEAKVERITY_SANDBOX", "process").lower() == "docker"
        cfg_path = os.path.join(work, "_lv_config.json")
        with open(cfg_path, "w") as f:
            json.dump({"data_files": [("/job/" + os.path.basename(p)) if docker else p for p in paths]}, f)
        out = os.path.join(work, "_lv_out")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work, "MPLBACKEND": "Agg",
               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"), "PYTHONHASHSEED": "0",
               "LANG": "C.UTF-8"}
        t0 = time.time()
        if docker:
            # one disposable container per run: no network, read-only root, capped memory/CPU/processes
            os.chmod(work, 0o777)
            cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
                   "--memory", f"{int(mem_gb)}g", "--cpus", os.getenv("LEAKVERITY_CPUS", "2"), "--pids-limit", "256",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-v", f"{work}:/job:rw", "-w", "/job",
                   os.getenv("LEAKVERITY_SANDBOX_IMAGE", "leakverity-sandbox"),
                   "/job/experiment.py", "/job/_lv_out", "/job/_lv_config.json"]
            run_kw = {}
        else:
            cmd = [sys.executable, "-I", str(TRACER), code_path, out, cfg_path]
            run_kw = {"env": env, "preexec_fn": _limits(timeout * 2, int(mem_gb * 2**30))}
        try:
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout, **run_kw)
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            return RunResult(False, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                             "", error=f"Timed out after {timeout} s", workdir=work, seconds=timeout)
        secs = time.time() - t0
        trace_path = os.path.join(out, "trace.json")
        if not os.path.exists(trace_path):
            return RunResult(False, stdout, stderr, error=(stderr or f"exit code {rc}")[-3000:], workdir=work, seconds=secs)
        with open(trace_path) as f:
            trace = json.load(f)
        design = None
        dpath = os.path.join(out, "design.csv")
        if os.path.exists(dpath):
            design = pd.read_csv(dpath, dtype={"__lv_index__": str})
        err = trace["summary"].get("error")
        return RunResult(err is None, stdout, stderr, trace["events"], trace["summary"], design, err, work, secs)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
