"""
Batch-training configuration helpers.

The batch runner stores shared defaults at the config level and per-scene
overrides inside each scene entry. These helpers merge those dictionaries,
resolve data/output paths, and convert the final payloads into command-line
arguments for `train.py` and `postprocess.py`.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

from utils.experiment_utils import load_json
from utils.platform_utils import runner_resource_limits


DEFAULT_CONFIG_PATH = "training_config.json"


def parse_args() -> argparse.Namespace:
    """
        Parse command-line options for the batch runner.
    """
    parser = argparse.ArgumentParser(description="Config-driven batch training")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data_root", help="Override the dataset root in the config")
    parser.add_argument("--output_root", help="Override the output root in the config")
    parser.add_argument("--gpu", "--gpu_id", dest="gpu_id", type=int, help="Override the physical GPU index")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--only-scene", "--only_scene", dest="only_scene", type=str, default="")
    parser.add_argument("--model_root", help="Override the fixed-model ablation source")
    parser.add_argument("--only-variant", "--only_variant", dest="only_variant")
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--max_views", type=int, help="Fixed-model verification view limit; 0 uses all views")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rebuild-summary", "--rebuild_total", dest="rebuild_total", action="store_true")
    modes.add_argument("--postprocess-only", action="store_true")
    modes.add_argument("--cleanup-repeats", action="store_true")
    return parser.parse_args()


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
        Deep-merge two dictionaries without mutating either input.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_batch_config(config_path: str) -> dict[str, Any]:
    """
        Load one batch JSON config, returning an empty dict for missing content.
    """
    config = load_json(config_path, default={})
    if config.get("resource_limits") is not None:
        config["resource_limits"] = runner_resource_limits(config["resource_limits"])
    old_key = "select_best_repeat_by_psnr"
    if old_key in config:
        if "cleanup_repeats" in config and config["cleanup_repeats"] != config[old_key]:
            raise ValueError("Conflicting cleanup_repeats / select_best_repeat_by_psnr values")
        config["cleanup_repeats"] = config.pop(old_key)
    return config


def validate_batch_config(config: dict[str, Any], scene_filter: set[str]) -> None:
    """Validate the entire requested queue before any training or output mutation."""
    if config.get("experiment_type") is not None:
        from boundedgs.experiments import validate_experiment
        validate_experiment(config, scene_filter, dry_run=True)
        return
    if any(key in config for key in ('model_root', 'only_variant', 'widths', 'max_views')):
        raise ValueError('Fixed-model options require experiment_type.')
    if not isinstance(config.get("scenes"), list) or not config["scenes"]:
        raise ValueError("Training config must contain a nonempty scenes list")
    if any(not isinstance(scene, dict) or not isinstance(scene.get("name"), str)
           or not scene["name"] for scene in config["scenes"]):
        raise ValueError("Each scene requires a name")
    if type(config.get("repeat", 1)) is not int or config.get("repeat", 1) < 1:
        raise ValueError("repeat must be a positive integer")
    for key in ("cleanup_repeats", "run_postprocess", "resume_training", "stop_on_error"):
        if key in config and type(config[key]) is not bool:
            raise ValueError("{} must be a JSON boolean".format(key))
    for key in ("output_root", "data_root"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError("{} must be a nonempty path".format(key))
    names = []
    for scene in filter_config_scenes(config, scene_filter):
        if not isinstance(scene.get("name"), str) or not scene["name"]:
            raise ValueError("Each scene requires a name")
        stem = build_scene_output_stem(config, scene)
        if stem in {".", ".."} or Path(stem).name != stem:
            raise ValueError("Scene output must be a single directory name: {}".format(stem))
        names.extend(build_output_name(config, scene, index) for index in range(config.get("repeat", 1)))
    if len(names) != len(set(names)):
        raise ValueError("Scenes have conflicting output paths")


def parse_scene_filter(only_scene: str) -> set[str]:
    """
        Convert a comma-separated scene filter into a set of scene names.
    """
    return {item.strip() for item in str(only_scene).split(",") if item.strip()}


def filter_config_scenes(config: dict[str, Any], scene_filter: set[str]) -> list[dict[str, Any]]:
    """
        Return only scenes requested by the filter and report unknown names early.
    """
    scenes = list(config.get("scenes", []))
    if not scene_filter:
        return scenes
    filtered_scenes = [scene for scene in scenes if str(scene.get("name", "")) in scene_filter]
    missing_scenes = sorted(scene_filter - {str(scene.get("name", "")) for scene in filtered_scenes})
    if missing_scenes:
        raise ValueError("Requested scenes not found in config: {}".format(", ".join(missing_scenes)))
    return filtered_scenes


def resolve_config_paths(config_path: str) -> list[str]:
    """
        Resolve either a single config file or all runnable configs in a folder.
    """
    resolved_path = Path(config_path)
    if resolved_path.is_file():
        return [str(resolved_path)]
    if resolved_path.is_dir():
        config_files: list[str] = []
        for path in sorted(resolved_path.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".json" or path.name.endswith(".schema.json"):
                continue
            payload = load_json(str(path), default={})
            if isinstance(payload, dict) and (isinstance(payload.get("scenes"), list) or "experiment_type" in payload):
                config_files.append(str(path))
        if not config_files:
            raise FileNotFoundError(
                "No runnable training or experiment configs found in directory: {}".format(config_path)
            )
        return config_files
    raise FileNotFoundError("Config path does not exist: {}".format(config_path))


def dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    """
        Convert a payload dictionary into argparse-style `--key value` tokens.
    """
    cli_args = []
    for key, value in payload.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            cli_args.extend([flag, str(value).lower()])
        elif value is None:
            continue
        elif isinstance(value, list):
            if value:
                cli_args.append(flag)
                cli_args.extend(str(item) for item in value)
        else:
            cli_args.extend([flag, str(value)])
    return cli_args


def build_scene_output_stem(config: dict[str, Any], scene: dict[str, Any]) -> str:
    """
        Build the base output name from scene name plus global and scene suffixes.
    """
    output_name = scene["name"]
    suffix_parts = [part for part in [config.get("output_suffix", ""), scene.get("output_suffix", "")] if part]
    if suffix_parts:
        output_name = "{}-{}".format(output_name, "-".join(suffix_parts))
    return output_name


def build_output_name(config: dict[str, Any], scene: dict[str, Any], run_index: int) -> str:
    """
        Add a repeat index to the output name when the config runs repeats.
    """
    output_name = build_scene_output_stem(config, scene)
    if int(config.get("repeat", 1)) > 1:
        output_name = "{}-{}".format(output_name, run_index + 1)
    return output_name


def build_scene_paths(config: dict[str, Any], scene: dict[str, Any], run_index: int) -> tuple[str, str]:
    """
        Resolve the input data folder and output folder for one scene run.
    """
    output_dir = os.path.join(config["output_root"], build_output_name(config, scene, run_index))
    data_dir = os.path.join(config["data_root"], scene["name"])
    return data_dir, output_dir


def build_train_payload(config: dict[str, Any], scene: dict[str, Any], data_dir: str, output_dir: str) -> dict[str, Any]:
    """
        Merge train arguments and inject source/model paths for one scene run.
    """
    train_payload = merge_dict(config.get("train_args", {}), scene.get("train_args", {}))
    train_payload["source_path"] = data_dir
    train_payload["model_path"] = output_dir
    train_payload["resume_training"] = bool(config.get("resume_training", False))
    return train_payload


def build_postprocess_payload(config: dict[str, Any], scene: dict[str, Any], data_dir: str, output_dir: str,
                              use_saved: bool = True) -> dict[str, Any]:
    """
        Build post-processing arguments from config and matching train settings.

        Dataset-related train options are copied unless post-processing explicitly
        overrides them, allowing every method to use one common evaluation renderer.
    """
    postprocess_payload = merge_dict(config.get("postprocess_args", {}), scene.get("postprocess_args", {}))
    explicit_postprocess_keys = set(postprocess_payload)
    train_payload = merge_dict(config.get("train_args", {}), scene.get("train_args", {}))
    saved_path = Path(output_dir) / "cfg_args"
    if use_saved and saved_path.is_file():
        from arguments import parse_cfg_args_namespace
        train_payload = vars(parse_cfg_args_namespace(saved_path.read_text(encoding="utf-8")))
    for key in (
        "sh_degree",
        "images",
        "resolution",
        "white_background",
        "data_device",
        "use_in_memory_jpeg_cache",
        "jpeg_prefetch_count",
        "training_image_upsample_factor",
        "eval",
        "debug",
        "antialiasing",
        "group_grid",
        "low_memory_lpips",
    ):
        if key in train_payload and key not in explicit_postprocess_keys:
            postprocess_payload[key] = train_payload[key]
    postprocess_payload.setdefault("source_path", train_payload.get("source_path", data_dir))
    postprocess_payload["model_path"] = output_dir
    return postprocess_payload


def build_python_command(python_executable: str, script_name: str, payload: dict[str, Any]) -> list[str]:
    """
        Build the final Python command list passed to subprocess.
    """
    return [python_executable or sys.executable, script_name, *dict_to_cli_args(payload)]
