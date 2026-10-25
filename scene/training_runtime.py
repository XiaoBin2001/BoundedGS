"""
Runtime helpers used by the training script.

This module contains code that is not tied to one training method: output folder
setup, checkpoint I/O, argument parsing, shared loop state, camera sampling, and
periodic evaluation reports.
"""
import os
import random
import re
import time
import uuid
from argparse import ArgumentParser, Namespace
from typing import Any, Optional, TypedDict

import torch
import numpy as np
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.general_utils import get_expon_lr_func
from utils.image_prefetch import iter_prefetched_camera_images
from utils.evaluation.lpips_metric import compute_lpips
from utils.experiment_utils import append_log_lines, archive_existing_output, write_training_parameters
from utils.loss_utils import psnr
from utils.loss_utils import ssim
from utils.evaluation.bounded_evaluation import evaluate_bounded_inference_view
from scene.training_context import TrainingContext
from scene.methods.adaptive_leaves import capture_cached_leaf_plans, restore_cached_leaf_plans


class TrainingLoopState(TypedDict):
    """
        State reused by many training-loop stages.

        Keeping these values in one typed dictionary makes call sites shorter
        while still showing which runtime objects are expected.
    """
    background: torch.Tensor
    log_path: str
    depth_l1_weight: Any
    report_lpips_model: Optional[LPIPS]
    viewpoint_stack: list[Any]
    testing_iterations: list[int]
    saving_iterations: list[int]
    checkpoint_iterations: list[int]
    ply_save_seconds: float


def synchronized_timestamp() -> float:
    """
        Return a timer value after pending CUDA work has finished.

        CUDA kernels are asynchronous, so synchronization is needed before timing
        sections that include GPU work.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _ensure_model_path(args: Any) -> None:
    """
        Fill `args.model_path` with a unique output folder when it is empty.
    """
    if args.model_path:
        return
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    args.model_path = os.path.join("./output/", unique_str[0:10])


def _build_training_background(dataset: Any) -> torch.Tensor:
    """
        Build the fixed white or black CUDA background tensor for training renders.
    """
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    return torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def _build_lpips_model_preserving_rng() -> LPIPS:
    """
        Create LPIPS without advancing the training RNG streams.
    """
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model = LPIPS("vgg").to("cuda").eval()
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.random.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return model


def prepare_output_and_logger(args: Any, runtime_args: Optional[Any] = None) -> None:
    """
        Prepare the experiment folder before training starts.

        Existing output is archived unless this run resumes in place. The current
        arguments are then written so the run can be reproduced later.
    """
    _ensure_model_path(args)

    should_archive_existing = True
    if runtime_args is not None:
        should_archive_existing = not is_inplace_checkpoint_resume(runtime_args, args.model_path)
    if should_archive_existing:
        archived_path = archive_existing_output(args.model_path)
        if archived_path is not None:
            print("Archived existing output to {}".format(archived_path))

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    if not should_archive_existing:
        print("[checkpoint] Keeping the original cfg_args and training_parameters.json.")
    elif runtime_args is not None:
        write_training_parameters(args.model_path, runtime_args)
        cfg_args = Namespace(**vars(runtime_args))
        cfg_args.source_path = args.source_path
        cfg_args.model_path = args.model_path
    else:
        cfg_args = Namespace(**vars(args))
    if should_archive_existing:
        with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(cfg_args))


def build_training_loop_state(
    context: TrainingContext,
    report_lpips_test: bool,
    report_lpips_train: bool,
) -> TrainingLoopState:
    """
        Build shared loop state after scene creation.

        This attaches scene-related objects into `context.runtime_state`, creates
        the background tensor, prepares evaluation settings, and initializes the
        shuffled training-camera pool.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene before building loop state.")

    dataset = context.dataset
    opt = context.opt
    runtime_args = context.runtime_args
    background = _build_training_background(dataset)
    runtime_state = context.runtime_state
    runtime_state["scene"] = scene
    runtime_state["pipe"] = context.pipe
    needs_lpips_model = (
        (report_lpips_test and bool(scene.getTestCameras()))
        or (report_lpips_train and bool(scene.getTrainCameras()))
    )

    return {
        "background": background,
        "log_path": os.path.join(scene.model_path, "log.txt"),
        "depth_l1_weight": get_expon_lr_func(
            opt.depth_l1_weight_init,
            opt.depth_l1_weight_final,
            max_steps=opt.iterations,
        ),
        "report_lpips_model": _build_lpips_model_preserving_rng() if needs_lpips_model else None,
        "viewpoint_stack": scene.getTrainCameras().copy(),
        "testing_iterations": list(runtime_args.test_iterations),
        "saving_iterations": list(runtime_args.save_iterations),
        "checkpoint_iterations": list(runtime_args.checkpoint_iterations),
        "ply_save_seconds": 0.0,
    }


def sample_training_viewpoint(scene: Scene, viewpoint_stack: list[Any]) -> Any:
    """
        Draw one random training camera without replacement.

        When the local pool becomes empty, it is refilled from the scene so each
        pass sees all training cameras in a random order.
    """
    if not viewpoint_stack:
        viewpoint_stack.extend(scene.getTrainCameras().copy())
    random_index = torch.randint(0, len(viewpoint_stack), (1,)).item()
    return viewpoint_stack.pop(random_index)


def latest_training_checkpoint_path(model_path: str) -> str:
    """Find the highest numbered checkpoint saved in this output folder."""
    checkpoints = []
    if os.path.isdir(model_path):
        for entry in os.scandir(model_path):
            match = re.fullmatch(r"chkpnt(\d+)\.pth", entry.name)
            if match and entry.is_file():
                checkpoints.append((int(match.group(1)), entry.path))
    if not checkpoints:
        raise FileNotFoundError("No numbered training checkpoint found in {}.".format(model_path))
    return max(checkpoints)[1]


def _merge_saved_resume_arguments(args: Namespace) -> Namespace:
    if not is_inplace_checkpoint_resume(args):
        return args

    from arguments import parse_cfg_args_namespace

    model_path = str(getattr(args, "model_path", "") or "").strip()
    if model_path == "":
        raise ValueError("resume_training requires model_path.")
    cfg_args_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_args_path):
        raise FileNotFoundError("Resume config not found: {}".format(cfg_args_path))
    checkpoint_path = resolve_start_checkpoint_path(model_path, args)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Resume checkpoint not found: {}".format(checkpoint_path))

    with open(cfg_args_path, "r", encoding="utf-8") as handle:
        saved_args = parse_cfg_args_namespace(handle.read())
    saved_model_path = str(getattr(saved_args, "model_path", "") or "").strip()
    if saved_model_path and os.path.realpath(saved_model_path) != os.path.realpath(model_path):
        raise ValueError(
            "Saved model_path {} does not match resume output {}.".format(
                saved_model_path,
                model_path,
            )
        )

    merged = vars(args).copy()
    for key, value in vars(saved_args).items():
        if key in merged:
            merged[key] = value
    merged["model_path"] = model_path
    for key in ("resume_training", "start_checkpoint_dir", "start_checkpoint_file"):
        merged[key] = getattr(args, key)
    print("[checkpoint] Using original training arguments from {}.".format(cfg_args_path))
    return Namespace(**merged)


def parse_training_arguments(argv: list[str]) -> tuple[Any, Any, Any, Namespace]:
    """
        Build the training CLI parser, register all option groups, and parse argv.
    """
    from arguments import ModelParams, OptimizationParams, PipelineParams, parse_bool_arg

    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000 * (i + 1) for i in range(1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000000 * (i + 1) for i in range(1000)])
    parser.add_argument("--quiet", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--progress_bar_width", type=int, default=90)
    parser.add_argument("--report_lpips_test", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--report_lpips_train", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--low_memory_lpips", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--start_checkpoint_dir", type=str, default="")
    parser.add_argument("--start_checkpoint_file", type=str, default="")
    parser.add_argument("--resume_training", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--save_final_model", default=True, nargs="?", const=True, type=parse_bool_arg)
    args = _merge_saved_resume_arguments(parser.parse_args(argv))
    return model_params, optimization_params, pipeline_params, args


def resolve_start_checkpoint_path(model_path: str, runtime_args: Any) -> Optional[str]:
    """
        Return the checkpoint path to load, or None when resume is disabled.

        Automatic resume uses the highest numbered checkpoint in model_path.
        An explicit checkpoint can also be selected by directory and filename.
    """
    if bool(getattr(runtime_args, "resume_training", False)):
        return latest_training_checkpoint_path(model_path)

    checkpoint_file = str(getattr(runtime_args, "start_checkpoint_file", "") or "").strip()
    if checkpoint_file == "":
        return None
    checkpoint_dir = str(getattr(runtime_args, "start_checkpoint_dir", "") or "").strip()
    if checkpoint_dir == "":
        checkpoint_dir = model_path
    return os.path.realpath(os.path.join(checkpoint_dir, checkpoint_file))


_CHECKPOINT_RUNTIME_FIELDS = (
    "opacity_min",
    "gns_lr_scaled",
    "gns_finished",
    "reg_prune_finished",
    "reg_prune_target_budget",
    "reg_prune_budget_reference_count",
)


def _camera_names(cameras: list[Any]) -> list[str]:
    return [str(camera.image_name) for camera in cameras]


def _capture_iteration_state(
    runtime_state: dict[str, Any],
    loop_state: TrainingLoopState | None,
    pending_viewpoints: list[Any] | None,
) -> dict[str, Any]:
    method_state = {
        key: runtime_state.get(key)
        for key in _CHECKPOINT_RUNTIME_FIELDS
    }
    numpy_rng_state = np.random.get_state()
    state: dict[str, Any] = {
        "method_state": method_state,
        "cached_leaf_plans": capture_cached_leaf_plans(runtime_state),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "bit_generator": str(numpy_rng_state[0]),
            "state": numpy_rng_state[1].tolist(),
            "position": int(numpy_rng_state[2]),
            "has_gauss": int(numpy_rng_state[3]),
            "cached_gaussian": float(numpy_rng_state[4]),
        },
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if loop_state is not None:
        state["viewpoint_stack"] = _camera_names(loop_state["viewpoint_stack"])
    if pending_viewpoints is not None:
        state["pending_viewpoints"] = _camera_names(pending_viewpoints)
    return state


def _resolve_checkpoint_cameras(scene: Scene, image_names: list[str], state_name: str) -> list[Any]:
    cameras_by_name = {str(camera.image_name): camera for camera in scene.getTrainCameras()}
    if len(cameras_by_name) != len(scene.getTrainCameras()):
        raise ValueError("Training camera image names must be unique for checkpoint restore.")
    missing_names = [name for name in image_names if str(name) not in cameras_by_name]
    if missing_names:
        raise ValueError(
            "Checkpoint {} contains unknown cameras: {}".format(
                state_name,
                ", ".join(str(name) for name in missing_names[:5]),
            )
        )
    return [cameras_by_name[str(name)] for name in image_names]


def restore_iteration_state(
    checkpoint_payload: dict[str, Any],
    scene: Scene,
    runtime_state: dict[str, Any],
    loop_state: TrainingLoopState,
    opt: Any = None,
) -> list[Any]:
    state = checkpoint_payload.get("iteration_state")
    if not isinstance(state, dict):
        return []

    if opt is not None:
        for key, value in checkpoint_payload.get("optimization_state", {}).items():
            if key in {"gns_opacity_reg", "reg_prune_until_iter"}:
                setattr(opt, key, value)
    restore_cached_leaf_plans(runtime_state, scene.getTrainCameras(), state.get("cached_leaf_plans", {}))

    method_state = state.get("method_state")
    if isinstance(method_state, dict):
        for key in _CHECKPOINT_RUNTIME_FIELDS:
            if key in method_state:
                runtime_state[key] = method_state[key]
    viewpoint_names = state.get("viewpoint_stack")
    if isinstance(viewpoint_names, list):
        loop_state["viewpoint_stack"] = _resolve_checkpoint_cameras(
            scene,
            viewpoint_names,
            "viewpoint_stack",
        )
    pending_names = state.get("pending_viewpoints")
    pending_viewpoints = (
        _resolve_checkpoint_cameras(scene, pending_names, "pending_viewpoints")
        if isinstance(pending_names, list)
        else []
    )

    python_rng_state = state.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)
    numpy_rng_state = state.get("numpy_rng_state")
    if isinstance(numpy_rng_state, dict):
        np.random.set_state(
            (
                str(numpy_rng_state["bit_generator"]),
                np.asarray(numpy_rng_state["state"], dtype=np.uint32),
                int(numpy_rng_state["position"]),
                int(numpy_rng_state["has_gauss"]),
                float(numpy_rng_state["cached_gaussian"]),
            )
        )
    torch_rng_state = state.get("torch_rng_state")
    if isinstance(torch_rng_state, torch.Tensor):
        torch.random.set_rng_state(torch_rng_state.cpu())
    cuda_rng_state = state.get("cuda_rng_state")
    if torch.cuda.is_available() and isinstance(cuda_rng_state, list) and cuda_rng_state:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return pending_viewpoints


def save_training_checkpoint(
    checkpoint_path: str,
    gaussians: GaussianModel3DGS,
    iteration: int,
    runtime_state: dict[str, Any] | None = None,
    loop_state: TrainingLoopState | None = None,
    pending_viewpoints: list[Any] | None = None,
    opt: Any = None,
    resource_state: dict[str, Any] | None = None,
) -> None:
    """
        Atomically save a complete end-of-iteration training checkpoint.
    """
    runtime_state = runtime_state or {}
    payload = {
        "model_params": gaussians.capture(),
        "iteration": int(iteration),
        "iteration_state": _capture_iteration_state(runtime_state, loop_state, pending_viewpoints),
        "optimization_state": {
            key: getattr(opt, key)
            for key in ("gns_opacity_reg", "reg_prune_until_iter")
        } if opt is not None else {},
        "resource_state": resource_state,
    }
    temporary_path = checkpoint_path + ".tmp"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_training_checkpoint(
    checkpoint_path: str,
) -> tuple[Any, int, dict[str, Any]]:
    """
        Load the current checkpoint format, including auxiliary resume state.
    """
    # Trusted training checkpoints include NumPy and RNG state beyond tensors.
    payload = torch.load(checkpoint_path, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint payload format: {}".format(type(payload).__name__))
    model_params = payload.get("model_params")
    if not isinstance(model_params, tuple) or len(model_params) != 15:
        raise ValueError("Checkpoint requires the current 15-field Gaussian model state.")
    scaling = model_params[5]
    if not isinstance(scaling, torch.Tensor) or scaling.ndim != 2 or scaling.shape[1] != 3:
        raise ValueError("Checkpoint requires three-dimensional Gaussian scales.")
    return model_params, int(payload["iteration"]), payload


def is_inplace_checkpoint_resume(runtime_args: Any, model_path: str | None = None) -> bool:
    """
        Check whether resume should reuse the current output folder.

        Resolve directory aliases and absolute filenames before deciding whether
        archiving the output folder would remove the checkpoint source.
    """
    if bool(getattr(runtime_args, "resume_training", False)):
        return True
    model_path = str(model_path or getattr(runtime_args, "model_path", "") or "").strip()
    if not model_path:
        return False
    checkpoint_path = resolve_start_checkpoint_path(model_path, runtime_args)
    if checkpoint_path is None:
        return False
    output_path = os.path.realpath(model_path)
    return os.path.commonpath((output_path, os.path.realpath(checkpoint_path))) == output_path


def _evaluate_report_view(
    viewpoint: Any,
    scene: Scene,
    render_func: Any,
    render_args: tuple[Any, ...],
    l1_loss_fn: Any,
    lpips_model: Optional[LPIPS],
    report_lpips: bool,
    group_grid: int,
    low_memory_lpips: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
        Render one validation view and compute all requested image metrics.
    """
    if int(group_grid) > 1:
        bounded_metrics = evaluate_bounded_inference_view(
            viewpoint,
            scene.gaussians,
            render_args[0],
            render_args[1],
            int(group_grid),
            lpips_model,
            report_lpips,
            low_memory_lpips=low_memory_lpips,
        )
        return (
            bounded_metrics["l1"],
            bounded_metrics["psnr"],
            bounded_metrics["ssim"],
            bounded_metrics["lpips"],
        )

    image = render_func(viewpoint, scene.gaussians, *render_args)["render"]
    image = torch.clamp(image, 0.0, 1.0)
    gt_image = torch.clamp(viewpoint.original_image_cuda(), 0.0, 1.0)
    image_batch = image.unsqueeze(0)
    gt_image_batch = gt_image.unsqueeze(0)
    l1_value = l1_loss_fn(image, gt_image).mean().double()
    psnr_value = psnr(image_batch, gt_image_batch).mean().double()
    ssim_value = ssim(image_batch, gt_image_batch).mean().double()
    lpips_value = None
    if lpips_model is not None and report_lpips:
        lpips_value = compute_lpips(
            lpips_model,
            image_batch,
            gt_image_batch,
            low_memory_lpips,
        ).mean().double()
    return l1_value, psnr_value, ssim_value, lpips_value


def training_report(
    iteration: int,
    l1_loss_fn: Any,
    scene: Scene,
    render_func: Any,
    render_args: tuple[Any, ...],
    lpips_model: Optional[LPIPS],
    report_lpips_test: bool,
    report_lpips_train: bool,
    log_path: Optional[str],
    is_first_report_iteration: bool,
    group_grid: int = 1,
    jpeg_prefetch_count: int = 1,
    low_memory_lpips: bool = False,
) -> None:
    """
        Evaluate train and test cameras and write a compact report.

        The function averages metrics over each split, prints them through tqdm,
        and mirrors the same lines into the experiment log file.
    """
    any_split_reports_lpips = report_lpips_test or report_lpips_train
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras(), "report_lpips": report_lpips_test},
        {"name": "train", "cameras": scene.getTrainCameras(), "report_lpips": report_lpips_train},
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0 if config["report_lpips"] and lpips_model is not None else None
            for viewpoint in iter_prefetched_camera_images(
                config["cameras"],
                prefetch_count=jpeg_prefetch_count,
            ):
                l1_value, psnr_value, ssim_value, lpips_value = _evaluate_report_view(
                    viewpoint,
                    scene,
                    render_func,
                    render_args,
                    l1_loss_fn,
                    lpips_model,
                    config["report_lpips"],
                    group_grid,
                    low_memory_lpips,
                )
                viewpoint.release_image_cache()
                l1_test += l1_value
                psnr_test += psnr_value
                ssim_test += ssim_value
                if lpips_test is not None and lpips_value is not None:
                    lpips_test += lpips_value
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            ssim_test /= len(config["cameras"])
            if lpips_test is not None:
                lpips_test /= len(config["cameras"])
            split_label = "Test " if config["name"] == "test" else "Train"
            message = "[{}] {}: [L1 {:.5f}] [PSNR {:.3f}] [SSIM {:.5f}]".format(
                iteration,
                split_label,
                float(l1_test),
                float(psnr_test),
                float(ssim_test),
            )
            if config["report_lpips"] and lpips_test is not None:
                message += " [LPIPS {:.5f}]".format(float(lpips_test))
            elif any_split_reports_lpips:
                message += " [LPIPS -------]"
            message += " [N_GS {}]".format(scene.gaussians.get_xyz.shape[0])
            log_lines = [message]
            if config["name"] == "train":
                log_lines.append("")
                tqdm.write(message)
            else:
                terminal_message = message if is_first_report_iteration else "\n" + message
                tqdm.write(terminal_message)
            if log_path is not None:
                append_log_lines(log_path, log_lines)
