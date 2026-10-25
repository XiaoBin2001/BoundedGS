"""Config-driven training, postprocessing, summary and repeat cleanup."""
from __future__ import annotations

from pathlib import Path
import sys
import time

from utils.batch_training.config import (
    build_postprocess_payload, build_python_command, build_scene_paths, build_train_payload,
    filter_config_scenes, load_batch_config, parse_args, parse_scene_filter, resolve_config_paths,
    validate_batch_config,
)
from utils.batch_training.gpu import release_gpu_lock, select_scene_gpu
from utils.batch_training.processes import install_signal_handlers, run_command
from utils.batch_training.repeat_storage import (
    IncompleteRepeats, atomic_json, cleanup_scene, existing_model_runs, read_selection, reject_symlinks,
)
from utils.batch_training.results import rebuild_result_total_csv, should_select_best_repeat_by_psnr
from utils.experiment_utils import backup_batch_training_config, load_json
from utils.resource_limits import check_resource_support, resolve_resource_limits


def _apply_overrides(config, overrides):
    config.update(overrides or {})
    if config.get('experiment_type') is not None and 'model_root' in config:
        key = 'source_root' if config['experiment_type'] == 'components' else 'model_path'
        config[key] = config.pop('model_root')


def _model_tasks(config: dict, scenes: list[dict], postprocess_only: bool) -> list[tuple]:
    tasks = []
    for scene in scenes:
        plan = read_selection(config, scene)
        if postprocess_only or (plan and config.get("resume_training", False)):
            runs = existing_model_runs(config, scene)
        elif plan:
            raise ValueError("Already selected repeat group; use a new output root/suffix or resume_training")
        else:
            runs = [(i + 1, Path(build_scene_paths(config, scene, i)[1]).absolute())
                    for i in range(config.get("repeat", 1))]
        for repeat, directory in runs:
            reject_symlinks(directory)
            data_dir = str((Path(config["data_root"]) / scene["name"]).absolute())
            tasks.append((repeat, scene, data_dir, str(directory)))
    return sorted(tasks, key=lambda task: task[0])


def _run_stage(config: dict, command: list[str], gpu, output: Path, state: dict, stage: str, dry_run: bool) -> int:
    started_ns = time.time_ns()
    if not dry_run:
        state[stage] = "running"
        if stage == "train":
            state["postprocess"] = "pending"
        if stage == "postprocess":
            state["written_splits"] = []
        atomic_json(output / "run_status.json", state)
    code = 1
    try:
        code = run_command(command, gpu_id=gpu, dry_run=dry_run,
                           resource_limits=config.get("resource_limits"),
                           resource_path=str(output / (stage + "_resources.json")))
        return code
    finally:
        if not dry_run:
            state[stage] = "completed" if code == 0 else "failed"
            state[stage + "_exit_code"] = code
            if stage == "postprocess" and code == 0:
                state["written_splits"] = [split for split in ("train", "test")
                                            if (output / ("result_" + split + ".json")).is_file()
                                            and (output / ("result_" + split + ".json")).stat().st_mtime_ns >= started_ns]
            atomic_json(output / "run_status.json", state)


def run_single_config(config_path: str, dry_run: bool, scene_filter: set[str], rebuild_total: bool,
                      overrides: dict | None = None, postprocess_only: bool = False,
                      cleanup_repeats: bool = False) -> int:
    config = load_batch_config(config_path)
    _apply_overrides(config, overrides)
    validate_batch_config(config, scene_filter)
    if config.get('experiment_type') is not None:
        if postprocess_only or cleanup_repeats:
            raise ValueError('Fixed-model experiments support full execution or --rebuild-summary.')
        from boundedgs.experiments import run_config
        return run_config(config, scene_filter, dry_run, rebuild_total)
    scenes = filter_config_scenes(config, scene_filter)
    if rebuild_total:
        # Validate manifests even in preview, without writing tables.
        from utils.batch_training.results import collect_existing_total_rows
        collect_existing_total_rows(config_path, dict(config, scenes=scenes))
        if dry_run:
            print("Dry run: would rebuild total/best CSV for {}".format(config_path))
        else:
            rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
        return 0
    if cleanup_repeats:
        incomplete = False
        for scene in scenes:
            try:
                cleanup_scene(config, scene, dry_run=dry_run)
            except IncompleteRepeats as exc:
                print("Repeat cleanup skipped: {}".format(exc))
                incomplete = True
        if not dry_run and not incomplete:
            rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
        return int(incomplete)

    tasks = _model_tasks(config, scenes, postprocess_only)
    resource_limits = config.get("resource_limits")
    if resource_limits is not None:
        (resolve_resource_limits if dry_run else check_resource_support)(resource_limits)
    if not dry_run and not postprocess_only:
        Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
        backup = backup_batch_training_config(config, config_path)
        if backup is not None:
            print("Backed up batch training config to {}".format(backup))
    first_failure = 0
    for repeat, scene, data_dir, output_dir in tasks:
        lock = None
        output = Path(output_dir)
        state = (load_json(str(output / "run_status.json"), default={}) if postprocess_only else {})
        state.update(repeat=repeat, seed=0, scene=scene["name"], model_path=output_dir)
        if postprocess_only:
            state.setdefault("train", "completed")
        try:
            gpu, lock = select_scene_gpu(config, scene["name"], config.get("gpu_id"), dry_run)
            python = config.get("python_executable", sys.executable)
            if not postprocess_only:
                payload = build_train_payload(config, scene, data_dir, output_dir)
                code = _run_stage(config, build_python_command(python, "train.py", payload), gpu,
                                  output, state, "train", dry_run)
                if code:
                    first_failure = first_failure or code
                    if config.get("stop_on_error", True):
                        return code
                    continue
            if not postprocess_only and not config.get("run_postprocess", True):
                if not dry_run:
                    state["postprocess"] = "skipped"
                    atomic_json(output / "run_status.json", state)
                continue
            payload = build_postprocess_payload(config, scene, data_dir, output_dir,
                                                use_saved=postprocess_only or not dry_run or config.get("resume_training", False))
            state["required_splits"] = (["train"] if payload.get("include_train", False) else []) + (
                [] if payload.get("skip_test", False) else ["test"])
            code = _run_stage(config, build_python_command(python, "postprocess.py", payload), gpu,
                              output, state, "postprocess", dry_run)
            if code:
                first_failure = first_failure or code
                if config.get("stop_on_error", True):
                    return code
        finally:
            release_gpu_lock(lock)
    if not dry_run:
        if not postprocess_only and should_select_best_repeat_by_psnr(config):
            for scene in scenes:
                try:
                    cleanup_scene(config, scene)
                except IncompleteRepeats as exc:
                    print("Repeat cleanup skipped: {}".format(exc))
        rebuild_result_total_csv(config_path, config)
    elif not postprocess_only and should_select_best_repeat_by_psnr(config):
        print("After successful complete runs: verify all records, select finite Test PSNR, archive non-best outputs.")
    return first_failure


def main() -> int:
    install_signal_handlers()
    args = parse_args()
    try:
        scene_filter = parse_scene_filter(args.only_scene)
        paths = resolve_config_paths(args.config)
        overrides = {key: getattr(args, key) for key in ("data_root", "output_root", "gpu_id", "model_root", "only_variant", "widths", "max_views")
                     if getattr(args, key, None) is not None}
        # Every config is checked before backups, GPU locks or child processes.
        seen_outputs = set()
        for path in paths:
            config = load_batch_config(path)
            _apply_overrides(config, overrides)
            try:
                validate_batch_config(config, scene_filter)
                if config.get('experiment_type') is not None:
                    if args.postprocess_only or args.cleanup_repeats:
                        raise ValueError('Fixed-model experiments support full execution or --rebuild-summary.')
                    from boundedgs.experiments import validate_experiment
                    jobs = validate_experiment(config, scene_filter, args.dry_run, args.rebuild_total)
                    for _, _, _, output, _ in jobs:
                        if str(output) in seen_outputs:
                            raise ValueError('Multiple configs target the same output: ' + str(output))
                        seen_outputs.add(str(output))
                    limits = config.get('resource_limits')
                    if limits is not None and not args.rebuild_total:
                        (resolve_resource_limits if args.dry_run else check_resource_support)(limits)
                    continue
                if not args.rebuild_total and not args.cleanup_repeats:
                    tasks = _model_tasks(config, filter_config_scenes(config, scene_filter), args.postprocess_only)
                    for _, _, _, output in tasks:
                        if output in seen_outputs:
                            raise ValueError("Multiple configs target the same output: {}".format(output))
                        seen_outputs.add(output)
                    limits = config.get("resource_limits")
                    if limits is not None:
                        (resolve_resource_limits if args.dry_run else check_resource_support)(limits)
            except (ValueError, OSError) as exc:
                raise ValueError("{}: {}".format(path, exc)) from exc
        first_failure = 0
        for path in paths:
            print("Processing config {}".format(path))
            code = run_single_config(path, args.dry_run, scene_filter, args.rebuild_total, overrides,
                                     args.postprocess_only, args.cleanup_repeats)
            if code:
                first_failure = first_failure or code
                if load_batch_config(path).get("stop_on_error", True):
                    return code
        return first_failure
    except (ValueError, OSError) as exc:
        print("Batch workflow error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
