"""
Subprocess management for batch training.

Training and post-processing are launched as child processes. This module keeps
track of them so Ctrl-C or SIGTERM can stop the full process group instead of
leaving long-running GPU jobs alive.
"""
from __future__ import annotations

import os
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from utils import platform_utils
from utils.resource_limits import ResourceLimiter, cpu_order_for_gpu, resolve_resource_limits, resource_description


child_processes: list[subprocess.Popen] = []
should_exit = False


def signal_handler(sig: int, frame: object) -> None:
    """
        Stop all known child processes and exit with the standard interrupt code.
    """
    del sig, frame
    global should_exit
    should_exit = True
    print("\nReceived stop signal, terminating all child processes...")

    if platform_utils.WINDOWS:
        for proc in list(child_processes):
            _stop_process_group(proc)
        print("All child processes stopped.")
        raise SystemExit(130)

    for proc in list(child_processes):
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    for proc in list(child_processes):
        if proc.poll() is not None:
            continue
        try:
            # Training handles SIGTERM as an interrupt and needs time to close
            # decoder workers and flush its resource summary.
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    print("All child processes stopped.")
    raise SystemExit(130)


def install_signal_handlers() -> None:
    """
        Register cleanup handlers for SIGINT and SIGTERM.
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if platform_utils.WINDOWS:
        signal.signal(signal.SIGBREAK, signal_handler)


def _stop_process_group(process: subprocess.Popen, timeout: float = 5.0) -> None:
    """Stop the task tree on Windows or the independent process group on Linux."""
    if platform_utils.WINDOWS:
        # Snapshot descendants before Ctrl+Break can make their parent exit.
        try:
            parent = psutil.Process(process.pid)
            members = parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            return
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except OSError:
            for member in members:
                try:
                    member.terminate()
                except psutil.NoSuchProcess:
                    pass
        _, alive = psutil.wait_procs(members, timeout=timeout)
        for member in alive:
            try:
                member.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_command(command: list[str], gpu_id: Any, dry_run: bool,
                resource_limits: dict[str, Any] | None = None,
                resource_path: str | None = None) -> int:
    """
        Run a task in its own process group with optional CPU/RAM limits.
    """
    resource_limits = platform_utils.runner_resource_limits(resource_limits)
    command_text = " ".join(command)
    print(command_text)
    if dry_run:
        if resource_limits is not None:
            budget = resolve_resource_limits(resource_limits)
            cpu_ids = (cpu_order_for_gpu(budget["allowed_cpus"], gpu_id if gpu_id is not None else 0)[:budget["cpu_count"]]
                       if budget["cpu_max_percent"] is not None else None)
            print(resource_description(resource_limits))
            preview = " (GPU not fixed; slot 0 preview)" if gpu_id is None and cpu_ids is not None else ""
            print(f"CPU affinity: {cpu_ids if cpu_ids is not None else 'inherited'}{preview}")
        return 0

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    limiter = (ResourceLimiter(resource_limits, gpu_id if gpu_id is not None else 0)
               if resource_limits is not None else None)
    process = None
    exit_code = None
    error = None
    started_at = time.time()
    try:
        if limiter is not None:
            print(resource_description(resource_limits), flush=True)
            command, env = limiter.prepare(command, env)
        group_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                         if platform_utils.WINDOWS else {"start_new_session": True})
        process = subprocess.Popen(command, env=env, **group_options)
        child_processes.append(process)
        if limiter is not None:
            limiter.monitor(process)
        exit_code = process.wait()
        if limiter is not None and limiter.cgroup is not None:
            events = dict(line.split() for line in (limiter.cgroup / "memory.events").read_text().splitlines())
            if int(events.get("oom_kill", 0)):
                limiter.exceeded = "Strict memory limit exceeded: cgroup reported an OOM kill"
        if limiter is not None and limiter.exceeded:
            # The guard's leader may exit before a descendant handles SIGTERM.
            _stop_process_group(process)
            print(limiter.exceeded, flush=True)
            exit_code = 137
        return exit_code
    except BaseException as exc:
        error = str(exc)
        exit_code = 130 if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 1
        if process is not None:
            _stop_process_group(process)
        raise
    finally:
        if process in child_processes:
            child_processes.remove(process)
        if limiter is not None:
            try:
                limiter.close()
            finally:
                if resource_path is not None:
                    report = dict(limiter.report(), exit_code=exit_code, error=error,
                                  status="completed" if exit_code == 0 else "failed",
                                  gpu_id=gpu_id, started_at=started_at, finished_at=time.time())
                    path = Path(resource_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + ".tmp")
                    temporary.write_text(json.dumps(report, indent=2) + "\n")
                    temporary.replace(path)
