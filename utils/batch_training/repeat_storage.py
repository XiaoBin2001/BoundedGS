"""Verified, recoverable repeat selection for BoundedGS native outputs."""
from __future__ import annotations

from argparse import Namespace
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

from arguments import parse_cfg_args_namespace
from utils.batch_training.config import build_output_name, build_scene_output_stem
from utils.experiment_utils import load_json

MARKER = "repeat_selection.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD_FILES = {
    "cfg_args", "training_parameters.json", "log.txt", "run_status.json",
    "resource_summary.json", "train_resources.json", "postprocess_resources.json",
    "result_train.json", "result_test.json", "opacity_report.txt",
}


class IncompleteRepeats(ValueError):
    """A group is not eligible; its original outputs must remain intact."""


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def reject_symlinks(path: Path) -> None:
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError("Repeat operations do not follow symlinks: {}".format(item))


def _digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def _keep_record(name: str) -> bool:
    if name in RECORD_FILES or re.fullmatch(r"(train|test)/ours_\d+/per_view.json", name):
        return True
    if (name in {"input.ply", "cameras.json"}
            or re.fullmatch(r"chkpnt\d+\.pth", name)
            or re.fullmatch(r"point_cloud/iteration_\d+/point_cloud.ply", name)
            or re.fullmatch(r"(train|test)/ours_\d+/(renders|gt)/[^/]+\.png", name)):
        return False
    raise ValueError("Unknown output file purpose; repeat cleanup stopped: {}".format(name))


def inventory(directory: Path) -> dict:
    reject_symlinks(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Repeat output contains a symlink: {}".format(path))
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            files[name] = dict(_digest(path), keep=_keep_record(name))
        elif not path.is_dir():
            raise ValueError("Unsupported output entry: {}".format(path))
    return files


def _verify_files(directory: Path, files: dict, records_only: bool = False, allow_missing: bool = False) -> None:
    expected = {name: info for name, info in files.items() if not records_only or info["keep"]}
    reject_symlinks(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Unexpected symlink: {}".format(path))
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
        elif not path.is_dir():
            raise ValueError("Unsupported output entry: {}".format(path))
    if (actual - set(expected)) or (not allow_missing and actual != set(expected)):
        raise ValueError("File inventory changed: {}".format(directory))
    for name, info in expected.items():
        if allow_missing and name not in actual:
            continue
        if _digest(directory / name) != {key: info[key] for key in ("size", "sha256")}:
            raise ValueError("Size/SHA256 verification failed: {}".format(directory / name))


def _copy_files(source: Path, destination: Path, files: dict, records_only: bool = False) -> None:
    """Retry individual atomic copies after interruption, rejecting unknown staging files."""
    reject_symlinks(destination)
    expected = {name: info for name, info in files.items() if not records_only or info["keep"]}
    allowed = set(expected) | {name + ".copying" for name in expected}
    for path in destination.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.relative_to(destination).as_posix() not in allowed):
            raise ValueError("Unknown staging entry: {}".format(path))
    destination.mkdir(parents=True, exist_ok=True)
    if not records_only:
        for path in source.rglob("*"):
            if path.is_dir():
                (destination / path.relative_to(source)).mkdir(parents=True, exist_ok=True)
    for name, info in expected.items():
        target = destination / name
        digest = {key: info[key] for key in ("size", "sha256")}
        temporary = target.with_name(target.name + ".copying")
        if target.exists():
            if _digest(target) != digest:
                raise ValueError("Existing staging copy differs: {}".format(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, temporary)
            if _digest(temporary) != digest:
                raise ValueError("Copy verification failed: {}".format(temporary))
            temporary.replace(target)
        if temporary.exists():
            temporary.unlink()
    _verify_files(destination, files, records_only)


def _paths(config: dict, scene: dict) -> tuple[Path, Path, list[Path]]:
    root = Path(config["output_root"]).absolute()
    base = root / build_scene_output_stem(config, scene)
    stage = root / ("." + base.name + ".repeat_cleanup")
    originals = [root / build_output_name(config, scene, i) for i in range(config.get("repeat", 1))]
    for path in (base, stage, *originals):
        reject_symlinks(path)
    return base, stage, originals


def read_selection(config: dict, scene: dict, require_complete: bool = True) -> dict | None:
    base, stage, originals = _paths(config, scene)
    locations = [base / MARKER, *(p / MARKER for p in originals), stage / MARKER]
    for path in locations:
        reject_symlinks(path)
    found = [(path, load_json(str(path))) for path in locations if path.is_file()]
    if not found:
        if stage.exists() or (config.get("repeat", 1) > 1 and base.exists()):
            raise ValueError("Unidentified repeat target/staging directory: {}".format(base))
        return None
    plan = found[0][1]
    if (plan.get("version") != 1 or plan.get("base_path") != str(base)
            or plan.get("expected_repeats") != len(originals)
            or [r["original_path"] for r in plan["runs"]] != list(map(str, originals))
            or any(p.get("transaction") != plan.get("transaction") for _, p in found)):
        raise ValueError("Repeat manifest identity conflicts with requested group: {}".format(base))
    cleaning = PROJECT_ROOT / "output" / "cleaning"
    for run in plan["runs"]:
        if run["archive_path"] is not None and not Path(run["archive_path"]).is_relative_to(cleaning):
            raise ValueError("Archive is outside the project cleaning directory")
    if require_complete:
        if plan["status"] != "complete":
            raise ValueError("Interrupted repeat cleanup; run --cleanup-repeats first: {}".format(base))
        if any(path.exists() for path in originals):
            raise ValueError("New repeat outputs conflict with an already selected group: {}".format(base))
        for run in plan["runs"]:
            if run["archive_path"]:
                archive = Path(run["archive_path"])
                reject_symlinks(archive)
                if not archive.is_dir():
                    raise ValueError("Repeat archive is missing: {}".format(archive))
        _verify_records(base / plan["raw_directory"], plan)
        winner = plan["runs"][plan["winner_repeat"] - 1]
        if not (base / winner["model_file"]).is_file():
            raise ValueError("Selected complete model is missing: {}".format(base))
        args = vars(parse_cfg_args_namespace((base / "cfg_args").read_text(encoding="utf-8")))
        if Path(args["model_path"]).absolute() != base:
            raise ValueError("Selected model configuration path is inconsistent")
    return plan


def validate_model(directory: Path, final: bool = False) -> dict:
    """Only native complete outputs are model inputs, never lightweight record copies."""
    reject_symlinks(directory)
    if any(part.startswith("raw_repeat-") for part in directory.parts):
        raise ValueError("Lightweight repeat records do not contain a loadable model: {}".format(directory))
    params = load_json(str(directory / "training_parameters.json"), default={})
    cfg = directory / "cfg_args"
    if not params or not cfg.is_file():
        raise IncompleteRepeats("Missing effective training configuration: {}".format(directory))
    args = vars(parse_cfg_args_namespace(cfg.read_text(encoding="utf-8")))
    if Path(args.get("model_path", "")).absolute() != directory.absolute():
        raise ValueError("Saved model path does not match output directory: {}".format(directory))
    models = list(directory.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not models or (final and not (directory / "point_cloud" / ("iteration_{}".format(params["iterations"])) / "point_cloud.ply").is_file()):
        raise IncompleteRepeats("Complete model is missing: {}".format(directory))
    return params


def _new_plan(config: dict, scene: dict) -> dict:
    base, stage, originals = _paths(config, scene)
    if base.exists() or stage.exists():
        raise FileExistsError("Repeat target or staging directory already exists: {}".format(base))
    runs = []
    for number, source in enumerate(originals, 1):
        params = validate_model(source, final=True)
        status = load_json(str(source / "run_status.json"), default={})
        if (status.get("train") != "completed" or status.get("postprocess") != "completed"
                or "test" not in status.get("required_splits", [])
                or not set(status.get("required_splits", [])).issubset(status.get("written_splits", []))
                or status.get("repeat") != number or status.get("model_path") != str(source)):
            raise IncompleteRepeats("Missing successful matching train/postprocess status: {}".format(source))
        summary = load_json(str(source / "result_test.json"), default={})
        psnr = summary.get("PSNR")
        if (type(psnr) not in (int, float) or not math.isfinite(psnr)
                or summary.get("iteration") != params["iterations"]):
            raise IncompleteRepeats("Requires unrounded finite final Test PSNR: {}".format(source))
        for split in status.get("required_splits", ["test"]):
            result = load_json(str(source / ("result_" + split + ".json")), default={})
            if result.get("iteration") != params["iterations"]:
                raise IncompleteRepeats("Missing required final {} results: {}".format(split, source))
        runs.append(dict(repeat=number, seed=params.get("seed", 0), original_path=str(source),
                         test_psnr=psnr, model_file="point_cloud/iteration_{}/point_cloud.ply".format(params["iterations"]),
                         files=inventory(source), archive_path=None, archived=False))
    winner = max(runs, key=lambda r: (r["test_psnr"], -r["repeat"]))
    transaction = uuid.uuid4().hex
    archive_root = PROJECT_ROOT / "output" / "cleaning" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_repeats_" + transaction[:8])
    root = base.parent
    relative_root = (root.relative_to(PROJECT_ROOT) if root.is_relative_to(PROJECT_ROOT)
                     else Path("external") / hashlib.sha256(str(root).encode()).hexdigest()[:12])
    for run in runs:
        if run != winner:
            target = archive_root / relative_root / Path(run["original_path"]).name
            reject_symlinks(target)
            if target.exists():
                raise FileExistsError(target)
            run["archive_path"] = str(target)
    return dict(version=1, transaction=transaction, status="copying", base_path=str(base),
                expected_repeats=len(runs), winner_repeat=winner["repeat"],
                raw_directory="raw_repeat-{}_data".format(len(runs)), runs=runs)


def _verify_records(directory: Path, plan: dict) -> None:
    expected = {Path(run["original_path"]).name for run in plan["runs"]}
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
        raise ValueError("Incomplete repeat record directory: {}".format(directory))
    for run in plan["runs"]:
        _verify_files(directory / Path(run["original_path"]).name, run["files"], records_only=True)


def _relocate_best(base: Path, old: Path) -> None:
    cfg = base / "cfg_args"
    args = vars(parse_cfg_args_namespace(cfg.read_text(encoding="utf-8")))
    if Path(args["model_path"]).absolute() not in (old, base):
        raise ValueError("Unexpected saved winner model path")
    args["model_path"] = str(base)
    # An explicitly saved checkpoint path may also refer to the old output.
    for key in ("start_checkpoint_dir", "start_checkpoint_file"):
        value = args.get(key)
        if value and Path(value).absolute().is_relative_to(old):
            args[key] = str(base / Path(value).absolute().relative_to(old))
    temporary = cfg.with_name("cfg_args.tmp")
    temporary.write_text(str(Namespace(**args)), encoding="utf-8")
    temporary.replace(cfg)
    for filename in ("training_parameters.json", "run_status.json"):
        payload = load_json(str(base / filename))
        payload["model_path"] = str(base)
        if filename == "training_parameters.json":
            for key in ("start_checkpoint_dir", "start_checkpoint_file"):
                if key in args:
                    payload[key] = args[key]
        atomic_json(base / filename, payload)


def _archive_run(run: dict, transaction: str) -> None:
    source, target = Path(run["original_path"]), Path(run["archive_path"])
    reject_symlinks(target)
    if target.exists():
        _verify_files(target, run["files"])
        if source.exists():
            # Only a journaled cross-device copy may have both locations.
            if not run.get("cross_device"):
                raise FileExistsError("Both repeat source and archive exist: {}".format(source))
            _verify_files(source, run["files"], allow_missing=True)
            shutil.rmtree(source)
        return
    _verify_files(source, run["files"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not run.get("cross_device"):
        try:
            source.rename(target)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # Caller persists this transition before retrying the cross-device move.
            raise
    partial = target.with_name("." + target.name + ".copying-" + transaction)
    _copy_files(source, partial, run["files"])
    _verify_files(source, run["files"])
    partial.rename(target)
    _verify_files(target, run["files"])
    shutil.rmtree(source)  # Verified full archive exists: finish the cross-device move.


def cleanup_scene(config: dict, scene: dict, dry_run: bool = False) -> dict | None:
    if config.get("repeat", 1) <= 1:
        return None
    base, stage, originals = _paths(config, scene)
    plan = read_selection(config, scene, require_complete=False)
    if plan is None:
        plan = _new_plan(config, scene)
    if plan["status"] == "complete":
        read_selection(config, scene)
        for run in plan["runs"]:
            if run["archive_path"]:
                _verify_files(Path(run["archive_path"]), run["files"])
        print("Repeat cleanup already complete: {}".format(base))
        return plan
    if dry_run:
        print("Repeat cleanup preview: winner={} -> {}; records={}".format(
            plan["winner_repeat"], base, plan["raw_directory"]))
        for run in plan["runs"]:
            if run["archive_path"]:
                print("  archive {} -> {}".format(run["original_path"], run["archive_path"]))
        return plan
    winner = originals[plan["winner_repeat"] - 1]
    if not base.exists():
        if not stage.exists():
            stage.mkdir()
            atomic_json(stage / MARKER, plan)
        record_target = winner / plan["raw_directory"]
        if not record_target.exists():
            for run in plan["runs"]:
                source = Path(run["original_path"])
                _verify_files(source, run["files"])
                _copy_files(source, stage / plan["raw_directory"] / source.name, run["files"], records_only=True)
            _verify_records(stage / plan["raw_directory"], plan)
            (stage / plan["raw_directory"]).rename(record_target)
        _verify_records(record_target, plan)
        # The copy may have taken a long time. Recheck the winner's original
        # files too; the only new entries owned by this transaction are these.
        winner_run = plan["runs"][plan["winner_repeat"] - 1]
        for name, info in winner_run["files"].items():
            if _digest(winner / name) != {key: info[key] for key in ("size", "sha256")}:
                raise ValueError("Winner source changed during copying: {}".format(winner / name))
        for path in winner.rglob("*"):
            relative = path.relative_to(winner)
            if path.is_symlink():
                raise ValueError("Unexpected winner symlink: {}".format(path))
            if (path.is_file() and relative.parts[0] != plan["raw_directory"]
                    and relative.as_posix() not in {*winner_run["files"], MARKER, MARKER + ".tmp"}):
                raise ValueError("Unexpected winner file: {}".format(path))
        # Losers must still match before even renaming the winner.
        for run in plan["runs"]:
            if run["repeat"] != plan["winner_repeat"]:
                _verify_files(Path(run["original_path"]), run["files"])
        plan["status"] = "archiving"
        atomic_json(winner / MARKER, plan)
        winner.rename(base)
    _verify_records(base / plan["raw_directory"], plan)
    _relocate_best(base, winner)
    for run in plan["runs"]:
        if run["archive_path"]:
            try:
                _archive_run(run, plan["transaction"])
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                run["cross_device"] = True
                atomic_json(base / MARKER, plan)
                _archive_run(run, plan["transaction"])
            run["archived"] = True
            atomic_json(base / MARKER, plan)
    if stage.exists():
        if {p.name for p in stage.iterdir()} - {MARKER, MARKER + ".tmp"}:
            raise ValueError("Unknown staging contents; refusing to finish: {}".format(stage))
        for path in stage.iterdir():
            path.unlink()
        stage.rmdir()
    plan["status"] = "complete"
    atomic_json(base / MARKER, plan)
    print("Selected repeat {} at {}; other complete outputs moved to output/cleaning".format(plan["winner_repeat"], base))
    return plan


def existing_model_runs(config: dict, scene: dict) -> list[tuple[int, Path]]:
    """Resolve postprocessing inputs without cleanup or training side effects."""
    base, _, originals = _paths(config, scene)
    plan = read_selection(config, scene)
    runs = [(plan["winner_repeat"], base)] if plan else list(enumerate(originals, 1))
    for _, directory in runs:
        validate_model(directory)
        status = load_json(str(directory / "run_status.json"), default={})
        if status and status.get("train") != "completed":
            raise IncompleteRepeats("Training did not complete: {}".format(directory))
    return runs
