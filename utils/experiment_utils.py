"""
Experiment utilities: read/write JSON, record training parameters, write logs, archive output folders, and back up batch configs.
"""
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


BEIJING_TIMEZONE = timezone(timedelta(hours=8))
TRAINING_PARAMETERS_FILENAME = "training_parameters.json"


def beijing_now_iso() -> str:
    """
        Create the Beijing-time timestamp used in experiment records.
    """
    return datetime.now(BEIJING_TIMEZONE).isoformat()


def namespace_to_dict(args: Any) -> dict[str, Any]:
    """
        Convert a parameter object into a plain dictionary that can be written to JSON.
    """
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None
    }


def save_json(path: str, payload: Any) -> None:
    """
        Save a JSON file with the shared format and create parent folders automatically.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)


def load_json(path: str, default: Optional[Any] = None) -> Any:
    """
        Read a JSON config or result file in one common way.
    """
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_training_parameters_payload(args: Any) -> dict[str, Any]:
    """
        Build the training-parameter record saved in one experiment folder and add the start time.
    """
    payload = namespace_to_dict(args)
    payload.pop("test_iterations", None)
    payload["seed"] = 0  # safe_state() uses the same fixed seed for every repeat.
    payload["started_at_beijing"] = beijing_now_iso()
    return payload


def write_training_parameters(model_path: str, args: Any) -> dict[str, Any]:
    """
        Write the parameters used by this run into the experiment folder for reproduction and post-processing.
    """
    payload = build_training_parameters_payload(args)
    save_json(os.path.join(model_path, TRAINING_PARAMETERS_FILENAME), payload)
    return payload


def finalize_training_parameters(model_path: str, elapsed_seconds: float) -> None:
    """
        After training ends, add the end time and pure training duration to the parameter record.
    """
    parameter_path = os.path.join(model_path, TRAINING_PARAMETERS_FILENAME)
    payload = load_json(parameter_path, default={})
    payload["finished_at_beijing"] = beijing_now_iso()
    payload["elapsed_seconds"] = float(elapsed_seconds)
    payload["elapsed_minutes"] = float(elapsed_seconds) / 60.0
    save_json(parameter_path, payload)


def append_log_lines(log_path: str, lines: list[str]) -> None:
    """
        Copy terminal log lines into a log file and add a timestamp to each non-empty line.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        for line in lines:
            if line.strip() == "":
                handle.write("\n")
            else:
                timestamp = datetime.now(BEIJING_TIMEZONE).strftime("%y/%m/%d %H:%M:%S")
                handle.write("{} [{}]\n".format(line, timestamp))


def archive_existing_output(model_path: str) -> Optional[str]:
    """
        When an output folder with the same name already exists, rename the old folder by its training timestamp to avoid overwriting results.
    """
    if not os.path.isdir(model_path):
        return None

    training_parameters_path = os.path.join(model_path, TRAINING_PARAMETERS_FILENAME)
    if not os.path.exists(training_parameters_path):
        return None

    training_parameters = load_json(training_parameters_path, default={})
    started_at = training_parameters.get("started_at_beijing")

    if started_at:
        try:
            timestamp = datetime.fromisoformat(started_at).strftime("%y_%m_%d_%H_%M_%S")
        except ValueError:
            timestamp = datetime.now(BEIJING_TIMEZONE).strftime("%y_%m_%d_%H_%M_%S")
    else:
        timestamp = datetime.now(BEIJING_TIMEZONE).strftime("%y_%m_%d_%H_%M_%S")

    parent_dir = os.path.dirname(model_path)
    base_name = os.path.basename(model_path.rstrip(os.sep))
    archived_path = os.path.join(parent_dir, f"{base_name}_{timestamp}")
    suffix = 1
    while os.path.exists(archived_path):
        archived_path = os.path.join(parent_dir, f"{base_name}_{timestamp}_{suffix}")
        suffix += 1

    os.rename(model_path, archived_path)
    return archived_path


def count_total_runs(config: dict[str, Any]) -> int:
    """
        Check whether the current config contains batch training, which decides whether to back up the batch config.
    """
    repeat = int(config.get("repeat", 1))
    scenes = config.get("scenes", [])
    return max(repeat, 1) * max(len(scenes), 0)


def backup_batch_training_config(config: dict[str, Any], config_path: str) -> Optional[str]:
    """
        When one run contains multiple training jobs, back up the batch config into the output root.
    """
    if count_total_runs(config) <= 1:
        return None
    if not config_path or not os.path.exists(config_path):
        return None

    output_root = config.get("output_root", "output")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(BEIJING_TIMEZONE).strftime("%y_%m_%d_%H_%M_%S")
    output_suffix = config.get("output_suffix", "")
    prefix = f"{output_suffix}-" if output_suffix else ""
    backup_path = os.path.join(output_root, f"{prefix}training_config_{timestamp}.json")
    shutil.copy2(config_path, backup_path)
    return backup_path
