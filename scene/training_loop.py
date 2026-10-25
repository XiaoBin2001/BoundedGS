"""
Small stages called from the main training loop.

These helpers keep the loop in `train.py` readable. They update progress output,
run scheduled evaluation and saving, adjust learning-rate/SH schedules, and
perform optimizer steps according to the active training method.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.loss_utils import l1_loss
from utils.image_prefetch import resolve_jpeg_prefetch_count
from scene.training_context import TrainingContext
from scene.training_runtime import (
    TrainingLoopState,
    save_training_checkpoint,
    synchronized_timestamp,
    training_report,
)


def _resolve_dense_optimizer_type(opt: Any, model_label: str = "3DGS") -> str:
    """
        Validate and return the configured optimizer type for dense Gaussian tensors.
    """
    configured_optimizer_type = opt.optimizer_type
    if configured_optimizer_type not in {"default", "ours_adam"}:
        raise ValueError(
            "{} currently supports only optimizer_type in {{default, ours_adam}}; got {}.".format(
                str(model_label).upper(),
                configured_optimizer_type,
            )
        )
    return configured_optimizer_type


def update_training_progress_bar(
    progress_bar: Any,
    iteration: int,
    ema_loss_for_log: float,
    loss_value: float,
    num_gaussians: int,
) -> float:
    """
        Update the tqdm postfix with smoothed loss and current Gaussian count.
    """
    ema_loss_for_log = 0.4 * loss_value + 0.6 * ema_loss_for_log
    if iteration % 10 == 0:
        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.4f}", "N_GS": f"{num_gaussians}"})
        progress_bar.update(10)
    return ema_loss_for_log


def run_log_and_save_stage(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
) -> float:
    """
        Run scheduled evaluation and model saving.

        Return reporting time; accumulate PLY time separately in loop_state.
        The caller subtracts both from pure training time.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene during log/save stage.")

    runtime_args = context.runtime_args
    dataset = context.dataset
    pipe = context.pipe
    report_time_seconds = 0.0
    if iteration in loop_state["testing_iterations"]:
        report_start = synchronized_timestamp()
        training_report(
            iteration,
            l1_loss,
            scene,
            render,
            (pipe, loop_state["background"], 1.0, False, True, None),
            loop_state["report_lpips_model"],
            bool(getattr(runtime_args, "report_lpips_test", False)),
            bool(getattr(runtime_args, "report_lpips_train", False)),
            loop_state["log_path"],
            iteration == min(loop_state["testing_iterations"]),
            1,
            resolve_jpeg_prefetch_count(dataset),
            bool(getattr(runtime_args, "low_memory_lpips", False)),
        )
        report_time_seconds += synchronized_timestamp() - report_start
    if iteration in loop_state["saving_iterations"]:
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        save_start = synchronized_timestamp()
        try:
            scene.save(iteration)
        finally:
            loop_state["ply_save_seconds"] = (
                loop_state.get("ply_save_seconds", 0.0)
                + synchronized_timestamp() - save_start
            )
    return report_time_seconds


def run_checkpoint_stage(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
    pending_viewpoints: list[Any],
    resource_state: dict[str, Any] | None = None,
) -> None:
    """
        Save checkpoints after every operation for this iteration has completed.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene during checkpoint stage.")

    if iteration in loop_state["checkpoint_iterations"]:
        checkpoint_path = os.path.join(scene.model_path, "chkpnt{}.pth".format(iteration))
        print("\n[ITER {}] Saving checkpoint {}".format(iteration, checkpoint_path))
        save_training_checkpoint(
            checkpoint_path,
            gaussians,
            iteration,
            context.runtime_state,
            loop_state,
            pending_viewpoints,
            context.opt,
            resource_state,
        )


def _validate_dense_optimizer_type(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    model_label: str = "3DGS",
) -> None:
    """
        Ensure the model optimizer type matches the current config.
    """
    configured_optimizer_type = _resolve_dense_optimizer_type(context.opt, model_label)
    if str(getattr(gaussians, "optimizer_type", "default")).lower() != configured_optimizer_type:
        raise ValueError(
            "GaussianModel{} optimizer_type={} does not match current config {}.".format(
                str(model_label).upper(),
                getattr(gaussians, "optimizer_type", "default"),
                configured_optimizer_type,
            )
        )


def update_3dgs_training_schedule(context: TrainingContext, gaussians: GaussianModel3DGS, iteration: int) -> None:
    """
        Update learning rate and SH degree before rendering the current iteration.
    """
    _validate_dense_optimizer_type(context, gaussians, "3dgs")
    gaussians.update_learning_rate(int(iteration))
    sh_step_interval = 1000
    if iteration % sh_step_interval == 0:
        gaussians.oneupSHdegree()


def _step_dense_optimizer(
    gaussians: GaussianModel3DGS,
    optimizer_type: str,
    step_size: int,
    iteration: int,
) -> None:
    """
        Step either the custom GaussianAdam optimizer or the default Adam optimizer.
    """
    if optimizer_type == "ours_adam":
        gaussians.optimizer.step(step_size, iteration)
    else:
        gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


def run_3dgs_parameter_update_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
) -> None:
    """
        Apply optimizer updates for one iteration.

        Dense Gaussian tensors follow the selected method schedule.
    """
    if iteration >= int(context.opt.iterations):
        return

    optimizer_type = _resolve_dense_optimizer_type(context.opt, "3dgs")
    _step_dense_optimizer(gaussians, optimizer_type, len(gaussians.get_xyz), iteration)
