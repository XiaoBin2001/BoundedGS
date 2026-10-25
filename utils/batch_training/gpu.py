"""
GPU selection and lock-file helpers for batch training.

The batch runner can choose the least busy GPU from `nvidia-smi`. A small lock
file in the temporary directory prevents two local batch jobs from choosing
the same GPU at the same time.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Any, Optional

import psutil

from utils import platform_utils


GPU_LOCK_DIR = (os.path.join(tempfile.gettempdir(), "3dgs_gpu_locks")
                if platform_utils.WINDOWS else "/tmp/3dgs_gpu_locks")


def query_gpu_stats() -> list[dict[str, int]]:
    """
        Read GPU memory and utilization numbers from `nvidia-smi`.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    stats = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index, memory_used, utilization, memory_total = [part.strip() for part in line.split(",")]
        stats.append(
            {
                "index": int(index),
                "memory_used": int(memory_used),
                "utilization": int(utilization),
                "memory_total": int(memory_total),
            }
        )
    return stats


def process_exists(pid: int) -> bool:
    """
        Return whether a local process id still exists.
    """
    if platform_utils.WINDOWS:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cleanup_stale_lock(lock_path: str) -> bool:
    """
        Remove a GPU lock file when the process that created it is gone.
    """
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    pid = payload.get("pid")
    if pid is None or not process_exists(pid):
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        return True
    return False


def acquire_gpu_lock(config: dict[str, Any], scene_name: str) -> tuple[Optional[int], Optional[str]]:
    """
        Choose a candidate GPU and create its lock file atomically.

        The atomic file creation step is what prevents another process from
        taking the same GPU between selection and job launch.
    """
    os.makedirs(GPU_LOCK_DIR, exist_ok=True)

    if config.get("gpu_id") is not None:
        candidates = [{"index": int(config["gpu_id"]), "memory_used": 0, "utilization": 0}]
    elif not config.get("gpu_auto_select", True):
        return None, None
    else:
        candidates = sorted(
            query_gpu_stats(),
            key=lambda item: (item["memory_used"], item["utilization"], item["index"]),
        )

    for candidate in candidates:
        gpu_index = candidate["index"]
        lock_path = os.path.join(GPU_LOCK_DIR, f"gpu_{gpu_index}.lock")
        if os.path.exists(lock_path):
            cleanup_stale_lock(lock_path)
        payload = {"pid": os.getpid(), "scene": scene_name, "acquired_at": time.time()}
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return gpu_index, lock_path
        except FileExistsError:
            continue

    raise RuntimeError("No free GPU lock could be acquired. All GPUs appear busy.")


def release_gpu_lock(lock_path: Optional[str]) -> None:
    """
        Remove a lock file after the scene job finishes.
    """
    if not lock_path:
        return
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def select_scene_gpu(
    config: dict[str, Any],
    scene_name: str,
    configured_gpu_id: Any,
    dry_run: bool,
) -> tuple[Any, Optional[str]]:
    """
        Return the GPU id to expose to the child process plus its lock path.
    """
    if dry_run:
        return configured_gpu_id, None
    if configured_gpu_id is None and config.get("gpu_auto_select", True):
        return acquire_gpu_lock(config, scene_name)
    if configured_gpu_id is not None:
        return acquire_gpu_lock(config, scene_name)
    return configured_gpu_id, None
