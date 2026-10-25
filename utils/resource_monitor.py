"""Low-overhead process and CUDA resource summaries for training runs."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator

import torch

from utils.experiment_utils import append_log_lines, beijing_now_iso
from utils import platform_utils

if not platform_utils.WINDOWS:
    import resource


RESOURCE_SUMMARY_FILENAME = "resource_summary.json"
PROCESS_MEMORY_SAMPLE_INTERVAL = 10


def _read_proc_memory() -> dict[str, int]:
    if platform_utils.WINDOWS:
        return {}
    values: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                key, separator, raw_value = line.partition(":")
                if separator == "" or key not in {"VmRSS", "VmHWM", "VmSwap"}:
                    continue
                fields = raw_value.strip().split()
                if fields:
                    values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        pass

    try:
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        values["VmHWM"] = max(values.get("VmHWM", 0), peak_rss_bytes)
    except (OSError, ValueError):
        pass
    return values


def _read_total_ram_bytes() -> int | None:
    if platform_utils.WINDOWS:
        return None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * float(fraction)))
    return int(ordered[index])


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def resource_status_for_exception(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if (
        (cuda_oom_type is not None and isinstance(error, cuda_oom_type))
        or "cuda out of memory" in str(error).lower()
    ):
        return "oom"
    return "failed"


class TrainingResourceMonitor:
    """Accumulate allocator high-water marks without polling the GPU."""

    _CHECKPOINT_FIELDS = (
        "started_at_beijing", "workload", "peak_num_gaussians", "final_num_gaussians",
        "_phase_stats", "_structure_transitions", "_bounded_scheduler_stats", "_monitor_warnings",
        "_process_memory_sample_count", "_peak_sampled_swap_bytes",
        "_peak_sampled_rss_plus_swap_bytes", "_peak_cpu_rss_bytes",
        "_overall_peak_allocated", "_overall_peak_reserved", "_overall_peak_active",
        "_overall_peak_inactive_split", "_overall_peak_allocated_context",
        "_overall_peak_reserved_context", "_overall_peak_active_context",
        "_overall_peak_inactive_split_context", "_persistent_peak_allocated",
        "_persistent_peak_reserved", "_persistent_peak_context", "statistics_start_iteration",
    )

    def __init__(self, dataset: Any, opt: Any, runtime_args: Any) -> None:
        self.started_at_beijing = beijing_now_iso()
        self.started_at_monotonic = time.perf_counter()
        self.model_path = str(getattr(dataset, "model_path", "") or "")
        self.experiment: dict[str, Any] = {
            "model_path": self.model_path,
            "source_path": str(getattr(dataset, "source_path", "") or ""),
            "group_grid": int(getattr(opt, "group_grid", 1)),
            "projection_mode": str(getattr(opt, "projection_mode", "full") or "full"),
            "use_compact_parameter_gradients": bool(getattr(opt, "use_compact_parameter_gradients", True)),
            "structure_update_mode": str(
                getattr(opt, "structure_update_mode", "single_pass") or "single_pass"
            ),
            "structure_rebuild_backend": str(
                getattr(opt, "structure_rebuild_backend", "auto") or "auto"
            ),
            "use_memory_lifecycle_trim": bool(
                getattr(opt, "use_memory_lifecycle_trim", True)
            ),
            "leaf_raster_budget_mib": float(
                getattr(opt, "leaf_raster_budget_mib", 0.0)
            ),
            "training_image_upsample_factor": int(
                getattr(dataset, "training_image_upsample_factor", 1)
            ),
            "sh_degree": int(getattr(dataset, "sh_degree", 0)),
            "budget": int(getattr(opt, "budget", 0)),
            "iterations_planned": int(getattr(opt, "iterations", 0)),
            "resume_training": bool(getattr(runtime_args, "resume_training", False)),
        }
        self.workload: dict[str, Any] = {}
        self.completed_iteration = 0
        self.current_iteration = 0
        self.current_camera: dict[str, Any] = {}
        self.current_num_gaussians = 0
        self.peak_num_gaussians = 0
        self.final_num_gaussians = 0
        self.pure_training_seconds: float | None = None
        self.report_seconds: float | None = None
        self.ply_save_seconds = 0.0
        self.monitor_overhead_seconds = 0.0
        self._resume_timing: dict[str, float] = {}
        self._resume_workload: dict[str, Any] = {}
        self.statistics_start_iteration = 0
        self._finalized = False
        self._monitor_warnings: list[str] = []
        if platform_utils.WINDOWS:
            message = "Windows mode: Linux CPU memory statistics disabled; CUDA statistics and timing remain enabled."
            self._monitor_warnings.append(message)
            print(message, flush=True)
        self._cuda_available = bool(torch.cuda.is_available())
        self._cuda_device: int | None = None
        self._cuda_tracking_enabled = self._cuda_available
        self._active_phase: dict[str, Any] | None = None
        self._phase_stats: dict[str, dict[str, Any]] = {}
        self._parameter_adam_resident_bytes = None
        self._long_term_resident_bytes = None
        self._overall_peak_allocated = 0
        self._overall_peak_reserved = 0
        self._overall_peak_active = 0
        self._overall_peak_inactive_split = 0
        self._overall_peak_allocated_context: dict[str, Any] = {}
        self._overall_peak_reserved_context: dict[str, Any] = {}
        self._overall_peak_active_context: dict[str, Any] = {}
        self._overall_peak_inactive_split_context: dict[str, Any] = {}
        self._persistent_peak_allocated = 0
        self._persistent_peak_reserved = 0
        self._persistent_peak_context: dict[str, Any] = {}
        self._structure_transitions: list[dict[str, int]] = []
        self._bounded_scheduler_stats: dict[str, Any] = {
            "iterations_recorded": 0,
            "projection_group_count": 0,
            "projection_candidate_count_sum": 0,
            "projection_candidate_count_max": 0,
            "projection_input_count_sum": 0,
            "projection_input_count_max": 0,
            "projection_indexed_group_count": 0,
            "raster_leaf_count": 0,
            "split_count": 0,
            "count_call_count": 0,
            "forced_leaf_count": 0,
            "maximum_estimated_payload_bytes": 0,
            "maximum_actual_payload_bytes": 0,
            "maximum_raster_leaf_count_per_iteration": 0,
            "maximum_raster_leaf_count_context": {},
            "maximum_actual_payload_context": {},
            "initial_base_group_grid": None,
            "final_base_group_grid": None,
            "base_group_grid_iteration_counts": {},
        }
        self._process_memory_sample_count = 0
        self._peak_sampled_swap_bytes = 0
        self._peak_sampled_rss_plus_swap_bytes = 0
        self._peak_cpu_rss_bytes = 0
        if self._cuda_available:
            try:
                self._cuda_device = int(torch.cuda.current_device())
                torch.cuda.reset_peak_memory_stats(self._cuda_device)
            except Exception as error:
                self._disable_cuda_tracking(error)

    def _disable_cuda_tracking(self, error: BaseException) -> None:
        message = "CUDA resource tracking disabled after {}: {}".format(type(error).__name__, str(error))
        if message not in self._monitor_warnings:
            self._monitor_warnings.append(message)
        self._cuda_tracking_enabled = False
        self._active_phase = None

    def set_output_path(self, model_path: str) -> None:
        self.model_path = str(model_path or "")
        self.experiment["model_path"] = self.model_path

    def keep_completed_summary(self, iteration: int) -> bool:
        """A completed final checkpoint needs no new measurement or metadata write."""
        path = Path(self.model_path) / RESOURCE_SUMMARY_FILENAME
        if not path.is_file():
            return False
        payload = json.loads(path.read_text())
        if (payload.get("status") == "completed"
                and payload.get("progress", {}).get("completed_iteration") == iteration
                and payload.get("progress", {}).get("iterations_planned") == iteration):
            self._finalized = True
            return True
        return False

    def checkpoint_state(self) -> dict[str, Any]:
        """Save raw samples/counters at this node, never a later run's summary."""
        self._harvest_untracked_cuda_peak()
        self._sample_process_memory()
        return {
            "iteration": self.completed_iteration,
            "statistics": deepcopy({key: getattr(self, key) for key in self._CHECKPOINT_FIELDS}),
            "timing": self._timing_payload(),
        }

    def restore_checkpoint_state(self, state: dict[str, Any] | None, iteration: int) -> None:
        if not state:
            self.statistics_start_iteration = iteration
            self._monitor_warnings.append(
                "Checkpoint has no resource history; statistics cover iterations after {} only.".format(iteration)
            )
            return
        if int(state["iteration"]) != iteration:
            raise ValueError("Resource statistics do not match checkpoint iteration.")
        for key in self._CHECKPOINT_FIELDS:
            setattr(self, key, deepcopy(state["statistics"][key]))
        self._resume_timing = dict(state["timing"])
        self._resume_workload = deepcopy(self.workload)
        self.set_training_times(0.0, 0.0)

    def record_image_prefetch(self, stats: dict[str, Any]) -> None:
        combined = dict(stats)
        previous = self._resume_workload.get("image_prefetch", {})
        for key in ("decoded_views", "background_decode_seconds", "main_wait_seconds"):
            combined[key] = previous.get(key, 0) + stats.get(key, 0)
        self.workload["image_prefetch"] = combined


    def configure_scene(self, scene: Any, method_config: dict[str, Any], resumed_iteration: int) -> None:
        self.set_output_path(str(scene.model_path))
        self.experiment["use_gns_pruning"] = bool(method_config["use_gns_pruning"])
        self.experiment["group_grid"] = int(method_config.get("group_grid", 1))
        self.experiment["projection_mode"] = str(method_config.get("projection_mode", "full"))
        self.experiment["structure_update_mode"] = str(
            method_config.get("structure_update_mode", "single_pass")
        )
        self.experiment["structure_rebuild_backend"] = str(
            method_config.get("structure_rebuild_backend", "auto")
        )
        self.experiment["use_memory_lifecycle_trim"] = bool(
            method_config.get("use_memory_lifecycle_trim", True)
        )
        self.experiment["leaf_raster_budget_mib"] = float(
            method_config.get("leaf_raster_budget_mib", 0.0)
        )
        self.experiment["resumed_from_iteration"] = int(resumed_iteration)
        self.completed_iteration = int(resumed_iteration)
        self.current_iteration = int(resumed_iteration)

        train_cameras = list(scene.getTrainCameras())
        widths = [int(camera.image_width) for camera in train_cameras]
        heights = [int(camera.image_height) for camera in train_cameras]
        self.workload["train_views"] = len(train_cameras)
        self.workload["test_views"] = len(scene.getTestCameras())
        if widths and heights:
            self.workload.update(
                {
                    "image_width_min": min(widths),
                    "image_width_max": max(widths),
                    "image_height_min": min(heights),
                    "image_height_max": max(heights),
                }
            )

    def record_jpeg_cache(self, dataset: Any) -> None:
        encoded_images = getattr(dataset, "in_memory_jpeg_images", {})
        self.workload["jpeg_cache_bytes"] = sum(
            int(image.numel()) * int(image.element_size())
            for image in encoded_images.values()
            if isinstance(image, torch.Tensor)
        )

    def record_structure_rebuild_backends(self, gaussians: Any) -> None:
        counts = getattr(gaussians, "structure_rebuild_backend_counts", {})
        previous = self._resume_workload.get("structure_rebuild_backend_counts", {})
        self.workload["structure_rebuild_backend_counts"] = {
            key: int(previous.get(key, 0)) + int(counts.get(key, 0))
            for key in ("gpu", "cpu")
        }

    def record_structure_transition(self, iteration: int, before: int, after: int) -> None:
        """Record only iterations whose structure operation changed Gaussian count."""
        before_count = int(before)
        after_count = int(after)
        if before_count == after_count:
            return
        self._structure_transitions.append(
            {
                "iteration": int(iteration),
                "before": before_count,
                "after": after_count,
                "delta": after_count - before_count,
            }
        )


    def _context(self, phase: str) -> dict[str, Any]:
        context: dict[str, Any] = {
            "phase": str(phase),
            "iteration": int(self.current_iteration),
            "num_gaussians": int(self.current_num_gaussians),
        }
        context.update(self.current_camera)
        return context

    def _update_overall(
        self,
        allocated: int,
        reserved: int,
        phase: str,
        active: int = 0,
        inactive_split: int = 0,
    ) -> None:
        context = self._context(phase)
        if int(allocated) > self._overall_peak_allocated:
            self._overall_peak_allocated = int(allocated)
            self._overall_peak_allocated_context = context
        if int(reserved) > self._overall_peak_reserved:
            self._overall_peak_reserved = int(reserved)
            self._overall_peak_reserved_context = context
        if int(active) > self._overall_peak_active:
            self._overall_peak_active = int(active)
            self._overall_peak_active_context = context
        if int(inactive_split) > self._overall_peak_inactive_split:
            self._overall_peak_inactive_split = int(inactive_split)
            self._overall_peak_inactive_split_context = context

    def _sample_process_memory(self) -> dict[str, int]:
        if platform_utils.WINDOWS:
            return {}
        values = _read_proc_memory()
        self._process_memory_sample_count += 1
        self._peak_cpu_rss_bytes = max(self._peak_cpu_rss_bytes, int(values.get("VmHWM", 0)))
        rss_bytes = int(values.get("VmRSS", 0))
        swap_bytes = int(values.get("VmSwap", 0))
        self._peak_sampled_swap_bytes = max(self._peak_sampled_swap_bytes, swap_bytes)
        self._peak_sampled_rss_plus_swap_bytes = max(
            self._peak_sampled_rss_plus_swap_bytes,
            rss_bytes + swap_bytes,
        )
        return values

    def _read_cuda_allocator_stats(self) -> dict[str, int]:
        stats = torch.cuda.memory_stats_as_nested_dict(self._cuda_device)

        def pool_value(group_name: str, field_name: str) -> int:
            group = stats.get(group_name, {})
            if not isinstance(group, dict):
                return 0
            all_pool = group.get("all", {})
            if not isinstance(all_pool, dict):
                return 0
            return int(all_pool.get(field_name, 0))

        return {
            "allocated_current": pool_value("allocated_bytes", "current"),
            "allocated_peak": pool_value("allocated_bytes", "peak"),
            "reserved_current": pool_value("reserved_bytes", "current"),
            "reserved_peak": pool_value("reserved_bytes", "peak"),
            "active_current": pool_value("active_bytes", "current"),
            "active_peak": pool_value("active_bytes", "peak"),
            "inactive_split_current": pool_value("inactive_split_bytes", "current"),
            "inactive_split_peak": pool_value("inactive_split_bytes", "peak"),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }

    def _harvest_untracked_cuda_peak(self) -> None:
        if not self._cuda_tracking_enabled or self._cuda_device is None:
            return
        try:
            stats = self._read_cuda_allocator_stats()
            self._update_overall(
                stats["allocated_peak"],
                stats["reserved_peak"],
                "setup_or_other",
                stats["active_peak"],
                stats["inactive_split_peak"],
            )
        except Exception as error:
            self._disable_cuda_tracking(error)

    def _reset_cuda_peak(self) -> None:
        if not self._cuda_tracking_enabled or self._cuda_device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(self._cuda_device)
        except Exception as error:
            self._disable_cuda_tracking(error)

    def record_iteration_start(self, iteration: int, num_gaussians: int, gaussians: Any = None) -> None:
        self.current_iteration = int(iteration)
        self.current_camera = {}
        self._parameter_adam_resident_bytes = None
        self._long_term_resident_bytes = None
        if gaussians is not None and getattr(gaussians, "optimizer", None) is not None:
            from utils.compact_gradients import long_term_storage_bytes, parameter_adam_storage_bytes
            self._parameter_adam_resident_bytes = parameter_adam_storage_bytes(gaussians)
            self._long_term_resident_bytes = long_term_storage_bytes(gaussians)
        self.update_gaussian_count(num_gaussians)
        if int(iteration) == 1 or int(iteration) % PROCESS_MEMORY_SAMPLE_INTERVAL == 0:
            self._sample_process_memory()
        if not self._cuda_tracking_enabled or self._cuda_device is None:
            return
        try:
            stats = self._read_cuda_allocator_stats()
            allocated = stats["allocated_current"]
            reserved = stats["reserved_current"]
            if allocated > self._persistent_peak_allocated:
                self._persistent_peak_allocated = allocated
                self._persistent_peak_context = self._context("iteration_start")
            self._persistent_peak_reserved = max(self._persistent_peak_reserved, reserved)
            self._update_overall(
                allocated,
                reserved,
                "iteration_start",
                stats["active_current"],
                stats["inactive_split_current"],
            )
        except Exception as error:
            self._disable_cuda_tracking(error)

    def set_camera(self, camera: Any) -> None:
        self.current_camera = {
            "camera_name": str(getattr(camera, "image_name", "")),
            "image_width": int(getattr(camera, "image_width", 0)),
            "image_height": int(getattr(camera, "image_height", 0)),
        }

    def record_bounded_execution(self, stats: Any) -> None:
        """Aggregate rectangle workload decisions without retaining per-iteration logs."""
        if not isinstance(stats, dict) or not stats:
            return
        gt = stats.get('gt_prefetch')
        if isinstance(gt, dict):
            total = self.workload.setdefault('gt_prefetch', dict(requests=0, upload_bytes=0,
                                                               max_pending=0, cpu_wait_seconds=0.0))
            total['scope'] = gt.get('scope', 'leaf')
            for key in ('requests', 'upload_bytes', 'cpu_wait_seconds', 'cpu_pack_requests'):
                total[key] = total.get(key, 0) + gt.get(key, 0)
            total['max_pending'] = max(total['max_pending'], gt.get('max_pending', 0))
        aggregate = self._bounded_scheduler_stats
        aggregate["iterations_recorded"] += 1
        base_grid = int(stats.get("base_group_grid", 0) or 0)
        if base_grid > 0:
            if aggregate["initial_base_group_grid"] is None:
                aggregate["initial_base_group_grid"] = base_grid
            aggregate["final_base_group_grid"] = base_grid
            grid_key = str(base_grid)
            grid_counts = aggregate["base_group_grid_iteration_counts"]
            grid_counts[grid_key] = int(grid_counts.get(grid_key, 0)) + 1

        for key in (
            "projection_group_count",
            "projection_candidate_count_sum",
            "projection_input_count_sum",
            "projection_indexed_group_count",
            "raster_leaf_count",
            "split_count",
            "count_call_count",
            "forced_leaf_count",
        ):
            aggregate[key] += int(stats.get(key, 0) or 0)

        for key in (
            "projection_candidate_count_max",
            "projection_input_count_max",
            "maximum_estimated_payload_bytes",
            "maximum_actual_payload_bytes",
        ):
            value = int(stats.get(key, 0) or 0)
            if value > int(aggregate[key]):
                aggregate[key] = value
                if key == "maximum_actual_payload_bytes":
                    aggregate["maximum_actual_payload_context"] = self._context(
                        "optimization"
                    )

        raster_leaf_count = int(stats.get("raster_leaf_count", 0) or 0)
        if raster_leaf_count > int(aggregate["maximum_raster_leaf_count_per_iteration"]):
            aggregate["maximum_raster_leaf_count_per_iteration"] = raster_leaf_count
            aggregate["maximum_raster_leaf_count_context"] = self._context(
                "optimization"
            )

    def _bounded_scheduler_payload(self) -> dict[str, Any]:
        stats = dict(self._bounded_scheduler_stats)
        iterations = int(stats["iterations_recorded"])
        stats["average_raster_leaf_count_per_iteration"] = (
            0.0 if iterations == 0 else stats["raster_leaf_count"] / float(iterations)
        )
        stats["average_split_count_per_iteration"] = (
            0.0 if iterations == 0 else stats["split_count"] / float(iterations)
        )
        stats["average_projection_candidates_per_group"] = (
            0.0
            if stats["projection_group_count"] == 0
            else stats["projection_candidate_count_sum"]
            / float(stats["projection_group_count"])
        )
        stats["average_projected_gaussians_per_group"] = (
            0.0
            if stats["projection_group_count"] == 0
            else stats["projection_input_count_sum"]
            / float(stats["projection_group_count"])
        )
        return stats

    def update_gaussian_count(self, num_gaussians: int) -> None:
        self.current_num_gaussians = int(num_gaussians)
        self.final_num_gaussians = int(num_gaussians)
        self.peak_num_gaussians = max(self.peak_num_gaussians, int(num_gaussians))

    def mark_iteration_completed(self, iteration: int, num_gaussians: int) -> None:
        self.completed_iteration = int(iteration)
        self.update_gaussian_count(num_gaussians)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        token = self._begin_phase(str(name))
        try:
            yield
        finally:
            self._end_phase(token)

    def _begin_phase(self, name: str) -> dict[str, Any] | None:
        if not self._cuda_tracking_enabled or self._cuda_device is None:
            return None
        if self._active_phase is not None:
            self._monitor_warnings.append("Nested resource phase ignored: {}".format(name))
            return None
        try:
            stats = self._read_cuda_allocator_stats()
            self._update_overall(
                stats["allocated_peak"],
                stats["reserved_peak"],
                "setup_or_other",
                stats["active_peak"],
                stats["inactive_split_peak"],
            )
            self._reset_cuda_peak()
            if not self._cuda_tracking_enabled:
                return None
            token = {
                "name": name,
                "start_allocated": stats["allocated_current"],
                "parameter_adam_resident_bytes": (getattr(self, "_parameter_adam_resident_bytes", None) if name == "optimization" else None),
                "long_term_resident_bytes": (getattr(self, "_long_term_resident_bytes", None) if name == "optimization" else None),
                "start_num_alloc_retries": stats["num_alloc_retries"],
                "start_num_ooms": stats["num_ooms"],
                "context": self._context(name),
            }
            self._active_phase = token
            return token
        except Exception as error:
            self._disable_cuda_tracking(error)
            return None

    def _end_phase(self, token: dict[str, Any] | None) -> None:
        if token is None or token is not self._active_phase:
            return
        try:
            stats = self._read_cuda_allocator_stats()
            peak_allocated = stats["allocated_peak"]
            peak_reserved = stats["reserved_peak"]
            peak_active = stats["active_peak"]
            peak_inactive_split = stats["inactive_split_peak"]
            incremental_allocated = max(peak_allocated - int(token["start_allocated"]), 0)
            alloc_retries = max(
                stats["num_alloc_retries"] - int(token["start_num_alloc_retries"]),
                0,
            )
            num_ooms = max(stats["num_ooms"] - int(token["start_num_ooms"]), 0)
            phase_name = str(token["name"])
            phase_stats = self._phase_stats.setdefault(
                phase_name,
                {
                    "sample_count": 0,
                    "peak_allocated_bytes": 0,
                    "peak_reserved_bytes": 0,
                    "peak_active_bytes": 0,
                    "peak_inactive_split_bytes": 0,
                    "max_incremental_allocated_bytes": 0,
                    "max_incremental_start_allocated_bytes": 0,
                    "max_incremental_peak_allocated_bytes": 0,
                    "num_alloc_retries": 0,
                    "num_ooms": 0,
                    "incremental_samples": [],
                    "peak_allocated_context": {},
                    "peak_inactive_split_context": {},
                    "max_incremental_context": {},
                },
            )
            resident = token.get("long_term_resident_bytes")
            metric_version = 3
            if resident is None:
                resident = token.get("parameter_adam_resident_bytes")
                metric_version = 2
            if resident is not None:
                working_set = max(peak_allocated - resident, 0)
                if working_set > phase_stats.get("working_set_peak_bytes", -1):
                    phase_stats["working_set_peak_bytes"] = working_set
                    phase_stats["working_set_peak_context"] = dict(token["context"],
                        parameter_adam_resident_bytes=token.get("parameter_adam_resident_bytes"),
                        long_term_resident_bytes=resident,
                        memory_metric_version=metric_version,
                        allocated_peak_bytes=peak_allocated)
            phase_stats["sample_count"] += 1
            phase_stats["incremental_samples"].append(incremental_allocated)
            if peak_allocated > phase_stats["peak_allocated_bytes"]:
                phase_stats["peak_allocated_bytes"] = peak_allocated
                phase_stats["peak_allocated_context"] = dict(token["context"])
            phase_stats["peak_reserved_bytes"] = max(phase_stats["peak_reserved_bytes"], peak_reserved)
            phase_stats["peak_active_bytes"] = max(phase_stats["peak_active_bytes"], peak_active)
            if peak_inactive_split > phase_stats["peak_inactive_split_bytes"]:
                phase_stats["peak_inactive_split_bytes"] = peak_inactive_split
                phase_stats["peak_inactive_split_context"] = dict(token["context"])
            phase_stats["num_alloc_retries"] += alloc_retries
            phase_stats["num_ooms"] += num_ooms
            if incremental_allocated > phase_stats["max_incremental_allocated_bytes"]:
                phase_stats["max_incremental_allocated_bytes"] = incremental_allocated
                phase_stats["max_incremental_start_allocated_bytes"] = int(
                    token["start_allocated"]
                )
                phase_stats["max_incremental_peak_allocated_bytes"] = peak_allocated
                phase_stats["max_incremental_context"] = dict(token["context"])
            self._update_overall(
                peak_allocated,
                peak_reserved,
                phase_name,
                peak_active,
                peak_inactive_split,
            )
        except Exception as error:
            self._disable_cuda_tracking(error)
            return
        finally:
            self._active_phase = None
        self._reset_cuda_peak()

    def set_training_times(self, pure_training_seconds: float, report_seconds: float,
                           ply_save_seconds: float = 0.0) -> None:
        self.pure_training_seconds = self._resume_timing.get("pure_training_seconds", 0.0) + float(pure_training_seconds)
        self.report_seconds = self._resume_timing.get("report_seconds", 0.0) + float(report_seconds)
        self.ply_save_seconds = self._resume_timing.get("ply_save_seconds", 0.0) + float(ply_save_seconds)

    def _timing_payload(self) -> dict[str, Any]:
        return {
            "wall_seconds": self._resume_timing.get("wall_seconds", 0.0) + time.perf_counter() - self.started_at_monotonic,
            "pure_training_seconds": self.pure_training_seconds,
            "report_seconds": self.report_seconds,
            "ply_save_seconds": self.ply_save_seconds,
            "pure_training_excludes_ply_save": True,
            "resource_monitor_overhead_seconds": self._resume_timing.get("resource_monitor_overhead_seconds", 0.0) + self.monitor_overhead_seconds,
        }

    def _cuda_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": self._cuda_available,
            "tracking_enabled": self._cuda_tracking_enabled,
        }
        if not self._cuda_available or self._cuda_device is None:
            return payload

        final_allocated = 0
        final_reserved = 0
        allocator_stats: dict[str, Any] = {}
        try:
            stats = self._read_cuda_allocator_stats()
            final_allocated = stats["allocated_current"]
            final_reserved = stats["reserved_current"]
            allocator_stats = {
                "num_alloc_retries": stats["num_alloc_retries"],
                "num_ooms": stats["num_ooms"],
                "active_bytes": stats["active_current"],
                "inactive_split_bytes": stats["inactive_split_current"],
            }
            self._update_overall(
                final_allocated,
                final_reserved,
                "final",
                stats["active_current"],
                stats["inactive_split_current"],
            )
        except Exception as error:
            self._disable_cuda_tracking(error)
        payload["tracking_enabled"] = self._cuda_tracking_enabled

        phase_payload: dict[str, Any] = {}
        for name, raw_stats in self._phase_stats.items():
            samples = list(raw_stats["incremental_samples"])
            peak_reserved = int(raw_stats["peak_reserved_bytes"])
            peak_inactive_split = int(raw_stats["peak_inactive_split_bytes"])
            phase_payload[name] = {
                "sample_count": int(raw_stats["sample_count"]),
                "peak_allocated_bytes": int(raw_stats["peak_allocated_bytes"]),
                "peak_reserved_bytes": peak_reserved,
                "peak_active_bytes": int(raw_stats["peak_active_bytes"]),
                "peak_inactive_split_bytes": peak_inactive_split,
                "peak_inactive_split_fraction_of_peak_reserved": (
                    0.0 if peak_reserved == 0 else peak_inactive_split / float(peak_reserved)
                ),
                "max_incremental_allocated_bytes": int(raw_stats["max_incremental_allocated_bytes"]),
                "max_incremental_start_allocated_bytes": int(
                    raw_stats["max_incremental_start_allocated_bytes"]
                ),
                "max_incremental_peak_allocated_bytes": int(
                    raw_stats["max_incremental_peak_allocated_bytes"]
                ),
                "incremental_allocated_p50_bytes": _percentile(samples, 0.50),
                "incremental_allocated_p95_bytes": _percentile(samples, 0.95),
                "num_alloc_retries": int(raw_stats["num_alloc_retries"]),
                "num_ooms": int(raw_stats["num_ooms"]),
                "peak_allocated_context": raw_stats["peak_allocated_context"],
                "peak_inactive_split_context": raw_stats["peak_inactive_split_context"],
                "max_incremental_context": raw_stats["max_incremental_context"],
                "working_set_peak_bytes": raw_stats.get("working_set_peak_bytes"),
                "working_set_peak_context": raw_stats.get("working_set_peak_context", {}),
                "memory_metric_version": raw_stats.get("working_set_peak_context", {}).get("memory_metric_version"),
                "working_set_definition": ("peak allocated minus Gaussian parameters, initialized Adam state, "
                                           "and persistent densification statistics; optimization only"),
            }

        try:
            properties = torch.cuda.get_device_properties(self._cuda_device)
            device_name = str(properties.name)
            total_device_memory = int(properties.total_memory)
        except Exception as error:
            self._monitor_warnings.append(
                "Could not read CUDA device properties after {}: {}".format(type(error).__name__, str(error))
            )
            device_name = ""
            total_device_memory = 0
        payload.update(
            {
                "device_index": self._cuda_device,
                "device_name": device_name,
                "total_device_memory_bytes": total_device_memory,
                "overall_peak_allocated_bytes": self._overall_peak_allocated,
                "overall_peak_reserved_bytes": self._overall_peak_reserved,
                "overall_peak_active_bytes": self._overall_peak_active,
                "overall_peak_inactive_split_bytes": self._overall_peak_inactive_split,
                "overall_peak_inactive_split_fraction_of_peak_reserved": (
                    0.0
                    if self._overall_peak_reserved == 0
                    else self._overall_peak_inactive_split / float(self._overall_peak_reserved)
                ),
                "overall_peak_allocated_context": self._overall_peak_allocated_context,
                "overall_peak_reserved_context": self._overall_peak_reserved_context,
                "overall_peak_active_context": self._overall_peak_active_context,
                "overall_peak_inactive_split_context": self._overall_peak_inactive_split_context,
                "persistent_peak_allocated_bytes": self._persistent_peak_allocated,
                "persistent_peak_reserved_bytes": self._persistent_peak_reserved,
                "persistent_peak_context": self._persistent_peak_context,
                "final_allocated_bytes": final_allocated,
                "final_reserved_bytes": final_reserved,
                "phases": phase_payload,
                "allocator": allocator_stats,
            }
        )
        return payload

    def _build_payload(self, status: str, error: BaseException | None) -> dict[str, Any]:
        if self._active_phase is not None:
            self._end_phase(self._active_phase)
        self._harvest_untracked_cuda_peak()
        process_memory = self._sample_process_memory()
        error_payload = None
        if error is not None:
            error_payload = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "context": self._context("exception"),
            }
        self.workload["peak_num_gaussians"] = int(self.peak_num_gaussians)
        self.workload["final_num_gaussians"] = int(self.final_num_gaussians)
        self.workload["structure_transition_count"] = len(self._structure_transitions)
        self.workload["structure_net_added_gaussians"] = sum(
            max(int(item["delta"]), 0) for item in self._structure_transitions
        )
        self.workload["structure_net_removed_gaussians"] = sum(
            max(-int(item["delta"]), 0) for item in self._structure_transitions
        )
        self.workload["structure_transitions"] = self._structure_transitions
        return {
            "schema_version": 3,
            "status": str(status),
            "started_at_beijing": self.started_at_beijing,
            "finished_at_beijing": beijing_now_iso(),
            "experiment": self.experiment,
            "progress": {
                "completed_iteration": int(self.completed_iteration),
                "iterations_planned": int(self.experiment["iterations_planned"]),
                "statistics_start_iteration": self.statistics_start_iteration,
            },
            "workload": self.workload,
            "timing": self._timing_payload(),
            "memory": {
                "measurement_scope": {
                    "cuda": "PyTorch CUDA caching allocator; excludes CUDA context and non-PyTorch allocations",
                    "cuda_peak_method": "allocator high-water counters reset at phase boundaries; no GPU polling",
                    "optimization_incremental": "optimization phase peak allocated minus allocated bytes at phase entry; includes lazily materialized parameter gradients and is not a render-working-set measurement",
                    "cuda_fragmentation_ratio": "peak inactive-split divided by peak reserved; diagnostic high-water ratio, not a simultaneous snapshot",
                    "cpu": ("Disabled on Windows" if platform_utils.WINDOWS else
                            "Linux RSS for the training process and its threads; excludes system filesystem page cache"),
                    "cpu_peak_method": (None if platform_utils.WINDOWS else
                                        "VmHWM plus VmSwap sampling every {} iterations".format(
                                            PROCESS_MEMORY_SAMPLE_INTERVAL)),
                },
                "cuda": self._cuda_payload(),
                "cpu": self._cpu_payload(process_memory),
            },
            "bounded_scheduler": (
                self._bounded_scheduler_payload()
                if int(self._bounded_scheduler_stats["iterations_recorded"]) > 0
                else {}
            ),
            "error": error_payload,
            "monitor_warnings": self._monitor_warnings,
        }

    def _cpu_payload(self, process_memory: dict[str, int]) -> dict[str, Any]:
        if platform_utils.WINDOWS:
            return {"available": False, "disabled_reason": "Linux CPU memory statistics disabled on Windows"}
        return {
            "peak_rss_bytes": int(self._peak_cpu_rss_bytes),
            "current_rss_bytes": int(process_memory.get("VmRSS", 0)),
            "current_swap_bytes": int(process_memory.get("VmSwap", 0)),
            "peak_sampled_swap_bytes": int(self._peak_sampled_swap_bytes),
            "peak_sampled_rss_plus_swap_bytes": int(self._peak_sampled_rss_plus_swap_bytes),
            "sample_count": int(self._process_memory_sample_count),
            "total_system_ram_bytes": _read_total_ram_bytes(),
        }

    @staticmethod
    def _format_gib(value: int | None) -> str:
        if value is None:
            return "n/a"
        return "{:.2f} GiB".format(int(value) / float(1024**3))

    def _format_log_lines(self, payload: dict[str, Any]) -> list[str]:
        cuda = payload["memory"]["cuda"]
        cpu = payload["memory"]["cpu"]
        optimization = cuda.get("phases", {}).get("optimization", {})
        structure = cuda.get("phases", {}).get("structure", {})
        pure_training_seconds = payload["timing"]["pure_training_seconds"]
        training_hours = (
            "n/a"
            if pure_training_seconds is None
            else "{:.2f}h".format(float(pure_training_seconds) / 3600.0)
        )
        line = (
            "[Resources] status={} | rectangle={} | structure={} | completed={}/{} | peak_N_GS={:.2f}M | "
            "CUDA allocated={} | allocator reserved={} | inactive split={} | "
            "optimization incremental={} | structure peak={} | "
            "CPU RSS={} | training={} | wall={:.2f}h"
        ).format(
            payload["status"],
            self.experiment["group_grid"],
            self.experiment["structure_update_mode"],
            self.completed_iteration,
            self.experiment["iterations_planned"],
            self.peak_num_gaussians / 1_000_000.0,
            self._format_gib(cuda.get("overall_peak_allocated_bytes")),
            self._format_gib(cuda.get("overall_peak_reserved_bytes")),
            self._format_gib(cuda.get("overall_peak_inactive_split_bytes")),
            self._format_gib(optimization.get("max_incremental_allocated_bytes")),
            self._format_gib(structure.get("peak_allocated_bytes")),
            self._format_gib(cpu.get("peak_rss_bytes")),
            training_hours,
            payload["timing"]["wall_seconds"] / 3600.0,
        )
        lines = [line]
        if optimization.get("working_set_peak_bytes") is not None:
            lines.append("[Resources] training working-set peak={} (includes gradients; excludes parameter/Adam storage)".format(
                self._format_gib(optimization["working_set_peak_bytes"])))
        rebuild_counts = payload.get("workload", {}).get("structure_rebuild_backend_counts", {})
        if rebuild_counts:
            lines.append(
                "[Resources] structure rebuild backends | configured={} | gpu={} | cpu={}".format(
                    self.experiment.get("structure_rebuild_backend", "auto"),
                    int(rebuild_counts.get("gpu", 0)),
                    int(rebuild_counts.get("cpu", 0)),
                )
            )
        bounded_scheduler = payload.get("bounded_scheduler", {})
        if bounded_scheduler:
            lines.append(
                "[Resources] rectangle workload | iterations={} | grid={} | leaves/iter={:.2f} | "
                "splits/iter={:.2f} | max estimated/actual payload={}/{} | forced leaves={}".format(
                    int(bounded_scheduler.get("iterations_recorded", 0)),
                    bounded_scheduler.get("final_base_group_grid"),
                    float(bounded_scheduler.get("average_raster_leaf_count_per_iteration", 0.0)),
                    float(bounded_scheduler.get("average_split_count_per_iteration", 0.0)),
                    self._format_gib(
                        bounded_scheduler.get("maximum_estimated_payload_bytes")
                    ),
                    self._format_gib(
                        bounded_scheduler.get("maximum_actual_payload_bytes")
                    ),
                    int(bounded_scheduler.get("forced_leaf_count", 0)),
                )
            )
        peak_context = cuda.get("overall_peak_allocated_context", {})
        if peak_context:
            lines.append(
                "[Resources] GPU peak context | phase={} | iteration={} | camera={} | size={}x{} | N_GS={}".format(
                    peak_context.get("phase", ""),
                    peak_context.get("iteration", 0),
                    peak_context.get("camera_name", ""),
                    peak_context.get("image_width", 0),
                    peak_context.get("image_height", 0),
                    peak_context.get("num_gaussians", 0),
                )
            )
        if payload["error"] is not None:
            error_lines = str(payload["error"]["message"]).splitlines()
            error_message = error_lines[0][:300] if error_lines else ""
            lines.append("[Resources] error={} | {}".format(payload["error"]["type"], error_message))
        return lines

    def finalize(self, status: str, error: BaseException | None = None) -> dict[str, Any] | None:
        if self._finalized:
            return None
        payload = self._build_payload(status, error)
        if self.model_path == "":
            raise ValueError("Cannot write resource summary without model_path.")
        summary_path = os.path.join(self.model_path, RESOURCE_SUMMARY_FILENAME)
        _atomic_write_json(summary_path, payload)
        self._finalized = True
        lines = self._format_log_lines(payload)
        for line in lines:
            print(line)
        try:
            append_log_lines(os.path.join(self.model_path, "log.txt"), lines)
        except OSError as log_error:
            print("[Resources] Could not append log.txt: {}".format(log_error))
        print("[Resources] summary saved to {}".format(summary_path))
        return payload
