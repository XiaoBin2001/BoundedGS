"""
Batch result aggregation helpers.

Post-processing writes one JSON summary per scene and split. This module reads
those summaries, turns them into CSV rows, optionally chooses the best repeat by
PSNR, and can rebuild total CSV files from already existing outputs.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Optional

from utils.batch_training.config import (
    build_output_name,
    build_scene_output_stem,
    build_scene_paths,
)
from utils.experiment_utils import load_json
from utils.batch_training.repeat_storage import read_selection


def _bytes_to_gib(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) / float(1024**3), 4)


def _spatial_execution_mode(experiment: dict[str, Any]) -> str:
    group_grid = int(experiment.get("group_grid", 1) or 1)
    projection_mode = str(experiment.get("projection_mode", "full") or "full")
    if projection_mode == "group_compact":
        return "full_boundedgs"
    if group_grid <= 1:
        return "full_frame"
    if projection_mode == "simple_split":
        return "simple_split"
    return "simple_boundedgs"


def load_resource_result_fields(output_dir: str) -> dict[str, Any]:
    """Flatten the resource fields needed by experiment-level comparison tables."""
    resource_path = os.path.join(output_dir, "resource_summary.json")
    payload = load_json(resource_path, default={})
    if not isinstance(payload, dict) or not payload:
        return {}

    experiment = payload.get("experiment", {})
    workload = payload.get("workload", {})
    timing = payload.get("timing", {})
    cuda = payload.get("memory", {}).get("cuda", {})
    optimization = cuda.get("phases", {}).get("optimization", {})
    structure = cuda.get("phases", {}).get("structure", {})
    allocator = cuda.get("allocator", {})
    bounded_scheduler = payload.get("bounded_scheduler", {})
    pure_training_seconds = timing.get("pure_training_seconds")
    return {
        "resource_status": payload.get("status"),
        "group_grid": experiment.get("group_grid"),
        "projection_mode": experiment.get("projection_mode"),
        "execution_mode": _spatial_execution_mode(experiment),
        "sh_degree": experiment.get("sh_degree"),
        "training_image_upsample_factor": experiment.get(
            "training_image_upsample_factor"
        ),
        "structure_update_mode": experiment.get("structure_update_mode"),
        "leaf_raster_budget_mib": experiment.get(
            "leaf_raster_budget_mib"
        ),
        "peak_num_gaussians": workload.get("peak_num_gaussians"),
        "final_num_gaussians": workload.get("final_num_gaussians"),
        "pure_training_minutes": (
            None if pure_training_seconds is None else round(float(pure_training_seconds) / 60.0, 4)
        ),
        "wall_minutes": (
            None
            if timing.get("wall_seconds") is None
            else round(float(timing["wall_seconds"]) / 60.0, 4)
        ),
        "peak_allocated_gib": _bytes_to_gib(cuda.get("overall_peak_allocated_bytes")),
        "optimization_working_set_peak_gib": _bytes_to_gib(optimization.get("working_set_peak_bytes")),
        "optimization_working_set_resident_gib": _bytes_to_gib(
            optimization.get("working_set_peak_context", {}).get(
                "long_term_resident_bytes",
                optimization.get("working_set_peak_context", {}).get("parameter_adam_resident_bytes"),
            )),
        "optimization_working_set_peak_iteration": optimization.get("working_set_peak_context", {}).get("iteration"),
        "optimization_incremental_peak_gib": _bytes_to_gib(
            optimization.get("max_incremental_allocated_bytes")
        ),
        "optimization_incremental_start_allocated_gib": _bytes_to_gib(
            optimization.get("max_incremental_start_allocated_bytes")
        ),
        "optimization_incremental_total_peak_allocated_gib": _bytes_to_gib(
            optimization.get("max_incremental_peak_allocated_bytes")
        ),
        "peak_reserved_gib": _bytes_to_gib(cuda.get("overall_peak_reserved_bytes")),
        "peak_inactive_split_gib": _bytes_to_gib(cuda.get("overall_peak_inactive_split_bytes")),
        "peak_inactive_split_fraction": (
            None
            if cuda.get("overall_peak_inactive_split_fraction_of_peak_reserved") is None
            else round(
                float(cuda["overall_peak_inactive_split_fraction_of_peak_reserved"]),
                6,
            )
        ),
        "structure_peak_allocated_gib": _bytes_to_gib(structure.get("peak_allocated_bytes")),
        "structure_peak_inactive_split_gib": _bytes_to_gib(
            structure.get("peak_inactive_split_bytes")
        ),
        "allocator_retries": allocator.get("num_alloc_retries"),
        "allocator_ooms": allocator.get("num_ooms"),
        "resource_monitor_overhead_seconds": timing.get(
            "resource_monitor_overhead_seconds"
        ),
        "bounded_average_leaves": bounded_scheduler.get(
            "average_raster_leaf_count_per_iteration"
        ),
        "bounded_average_splits": bounded_scheduler.get(
            "average_split_count_per_iteration"
        ),
        "forced_raster_leaves": bounded_scheduler.get("forced_leaf_count"),
        "initial_group_grid": bounded_scheduler.get("initial_base_group_grid"),
        "final_group_grid": bounded_scheduler.get("final_base_group_grid"),
    }


def load_result_summary(output_dir: str, split_name: str) -> Optional[dict[str, Any]]:
    """
        Load one `result_<split>.json` file if it exists and contains data.
    """
    summary_path = os.path.join(output_dir, "result_{}.json".format(split_name))
    if not os.path.exists(summary_path):
        return None
    payload = load_json(summary_path, default={})
    return payload if isinstance(payload, dict) and payload else None


def collect_scene_result_rows(
    config_path: str,
    scene: dict[str, Any],
    output_dir: str,
) -> dict[str, list[dict[str, Any]]]:
    """
        Convert one scene output directory into train/test CSV rows.
    """
    status = load_json(os.path.join(output_dir, "run_status.json"), default={})
    if status and (status.get("train") != "completed" or status.get("postprocess") != "completed"):
        return {"train": [], "test": []}
    params = load_json(os.path.join(output_dir, "training_parameters.json"), default={})
    common_row = {
        "config_name": Path(config_path).stem,
        "scene_name": str(scene["name"]),
        "output_name": os.path.basename(output_dir.rstrip(os.sep)),
        "repeat": status.get("repeat", 1),
        "seed": params.get("seed", 0),
        "record_path": os.path.abspath(output_dir),
        "model_path": os.path.abspath(output_dir),
    }
    common_row.update(load_resource_result_fields(output_dir))
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for split_name in ("train", "test"):
        summary = load_result_summary(output_dir, split_name)
        if summary is None or (status and split_name not in status.get("written_splits", [])):
            continue
        row = dict(common_row)
        row["split"] = split_name
        row.update(summary)
        rows_by_split[split_name].append(row)
    return rows_by_split


def write_result_total_csv(output_root: str, config_path: str, split_name: str, rows: list[dict[str, Any]], selection: str = "total") -> Optional[str]:
    """
        Write rows for one split into the config-level total CSV file.
    """
    csv_path = os.path.join(output_root, "{}_result_{}_{}.csv".format(Path(config_path).stem, split_name, selection))
    if not rows and not os.path.exists(csv_path):
        return None
    ordered_fieldnames = [
        "config_name",
        "split",
        "iteration",
        "scene_name",
        "output_name",
        "repeat",
        "seed",
        "record_path",
        "model_path",
        "resource_status",
        "group_grid",
        "projection_mode",
        "execution_mode",
        "sh_degree",
        "training_image_upsample_factor",
        "structure_update_mode",
        "leaf_raster_budget_mib",
        "PSNR",
        "SSIM",
        "LPIPS",
        "FPS",
        "NUM",
        "Training_time",
        "pure_training_minutes",
        "wall_minutes",
        "peak_num_gaussians",
        "final_num_gaussians",
        "peak_allocated_gib",
        "optimization_working_set_peak_gib",
        "optimization_working_set_resident_gib",
        "optimization_working_set_peak_iteration",
        "optimization_incremental_peak_gib",
        "optimization_incremental_start_allocated_gib",
        "optimization_incremental_total_peak_allocated_gib",
        "peak_reserved_gib",
        "peak_inactive_split_gib",
        "peak_inactive_split_fraction",
        "structure_peak_allocated_gib",
        "structure_peak_inactive_split_gib",
        "allocator_retries",
        "allocator_ooms",
        "terminal_probe_camera",
        "terminal_probe_peak_allocated_gib",
        "terminal_probe_peak_reserved_gib",
        "estimated_minimum_gib",
        "allocator_capacity_estimate_gib",
        "allocator_segment_upper_gib",
        "resource_monitor_overhead_seconds",
        "allocator_trace_events",
        "allocator_trace_truncated",
        "bounded_average_leaves",
        "bounded_average_splits",
        "forced_raster_leaves",
        "initial_group_grid",
        "final_group_grid",
        "group_grid_upgrade_count",
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            display = dict(row)
            for metric, decimals in (("PSNR", 3), ("SSIM", 5), ("LPIPS", 5), ("FPS", 1), ("Training_time", 2)):
                if display.get(metric) is not None:
                    display[metric] = format(float(display[metric]), ".{}f".format(decimals))
            writer.writerow(display)
    return csv_path


def should_select_best_repeat_by_psnr(config: dict[str, Any]) -> bool:
    # Keep the old JSON key as an alias for direct callers as well as loaded configs.
    enabled = config.get("cleanup_repeats", config.get("select_best_repeat_by_psnr", True))
    return bool(enabled) and int(config.get("repeat", 1)) > 1


def extend_result_rows(
    result_rows_by_split: dict[str, list[dict[str, Any]]],
    scene_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """
        Append newly collected rows into the split-indexed row accumulator.
    """
    for split_name, rows in scene_rows.items():
        result_rows_by_split[split_name].extend(rows)


def collect_existing_total_rows(config_path: str, config: dict[str, Any], selection: str = "total") -> dict[str, list[dict[str, Any]]]:
    """Totals locate every frozen record; Best rows locate the selected complete model."""
    rows = {"train": [], "test": []}
    for scene in config.get("scenes", []):
        plan = read_selection(config, scene)
        if plan:
            base = Path(plan["base_path"])
            runs = ([(plan["winner_repeat"], base)] if selection == "best" else
                    [(run["repeat"], base / plan["raw_directory"] / Path(run["original_path"]).name)
                     for run in plan["runs"]])
        elif selection == "best":
            continue
        else:
            runs = [(i + 1, Path(build_scene_paths(config, scene, i)[1])) for i in range(config.get("repeat", 1))]
        for repeat, directory in runs:
            found = collect_scene_result_rows(config_path, scene, str(directory))
            for split_rows in found.values():
                for row in split_rows:
                    row["repeat"] = repeat
                    if plan and selection == "total":
                        row["model_path"] = ""  # This path is a record copy, never a model.
            extend_result_rows(rows, found)
    return rows


def rebuild_result_total_csv(config_path: str, config: dict[str, Any]) -> None:
    """Rebuild all-repeat and selected-repeat tables from their actual locations."""
    for selection in ("total", "best"):
        rows_by_split = collect_existing_total_rows(config_path, config, selection)
        for split_name, rows in rows_by_split.items():
            csv_path = write_result_total_csv(config["output_root"], config_path, split_name, rows, selection)
            if csv_path is not None:
                print("Rebuilt {} {} CSV to {}".format(split_name, selection, csv_path))
