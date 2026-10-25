"""
Pruning stage for methods that remove low-value Gaussians.

GNS uses a regularized pruning window: it raises opacity learning rate, applies
threshold pruning during the window, then enforces the final budget at the end.
"""
import os
from typing import Any

import torch
from tqdm import tqdm

from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.experiment_utils import append_log_lines
from scene.training_context import TrainingContext


def resolve_regularized_prune_target_budget(
    gaussians: GaussianModel3DGS,
    opt: Any,
    state: dict[str, Any],
) -> int | None:
    """
        Resolve and cache the final Gaussian count used by GNS pruning.

        `num` mode uses a fixed count. `rate` mode converts a keep percentage
        into a rounded count based on the model size at the start of pruning.
    """
    cached_budget = state.get("reg_prune_target_budget")
    if cached_budget is not None:
        return int(cached_budget)

    budget_mode = str(opt.final_budget_mode).lower()
    if budget_mode == "off":
        return None
    if budget_mode == "num":
        target_budget = int(opt.final_budget)
    elif budget_mode == "rate":
        reference_count = state.get("reg_prune_budget_reference_count")
        if reference_count is None:
            reference_count = int(gaussians.get_opacity.shape[0])
            state["reg_prune_budget_reference_count"] = reference_count
        final_rate = min(max(float(opt.final_rate), 0.0), 1.0)
        target_budget = int(round((int(reference_count) * final_rate) / 10000.0) * 10000.0)
        target_budget = max(target_budget, 1)
    else:
        raise ValueError("Unsupported final_budget_mode: {}".format(opt.final_budget_mode))

    target_budget = min(max(int(target_budget), 1), int(gaussians.get_opacity.shape[0]))
    state["reg_prune_target_budget"] = target_budget
    return target_budget


def apply_gns_threshold_prune(
    gaussians: GaussianModel3DGS,
    min_opacity: float,
    final_budget: int | None = None,
    structure_update_mode: str = "single_pass",
) -> None:
    """
        Remove Gaussians below the opacity threshold without crossing the budget.

        If threshold pruning would remove too many Gaussians, the function falls
        back to exact budget pruning.
    """
    opacity = gaussians.get_opacity
    if opacity.numel() == 0:
        return
    opacity_flat = opacity.detach().flatten()
    prune_mask = (opacity_flat < float(min_opacity)).flatten()
    if final_budget is not None:
        current_count = int(opacity_flat.shape[0])
        target_budget = min(max(int(final_budget), 1), current_count)
        if current_count - int(prune_mask.sum().item()) < target_budget:
            apply_gns_final_prune(
                gaussians,
                target_budget,
                structure_update_mode=structure_update_mode,
            )
            return
    if prune_mask.any():
        gaussians.prune_points(prune_mask, structure_update_mode=structure_update_mode)


def apply_gns_final_prune(
    gaussians: GaussianModel3DGS,
    final_budget: int,
    structure_update_mode: str = "single_pass",
) -> None:
    """
        Keep the top-opacity Gaussians until the exact final budget is reached.
    """
    importance = gaussians.get_opacity.detach().squeeze()
    if importance.numel() <= int(final_budget):
        return
    randomized_importance = importance + torch.rand_like(importance) * 1e-6
    keep_indices = torch.topk(randomized_importance, int(final_budget), largest=True, sorted=False).indices
    prune_mask = torch.ones_like(importance, dtype=torch.bool)
    prune_mask[keep_indices] = False
    gaussians.prune_points(prune_mask, structure_update_mode=structure_update_mode)


def finish_gns_pruning(
    iteration: int,
    gaussians: GaussianModel3DGS,
    opt: Any,
    state: dict[str, Any],
) -> None:
    """
        Mark GNS pruning as complete and restore the opacity learning rate.

        The completion message is written both to tqdm and to the experiment log
        so batch runs can inspect when pruning stopped.
    """
    if state.get("gns_lr_scaled", False):
        gaussians.update_opacity_lr(1.0 / float(opt.gns_opacity_lr_scale))
        state["gns_lr_scaled"] = False
    state["gns_finished"] = True
    state["reg_prune_finished"] = True
    if iteration < int(opt.reg_prune_until_iter):
        setattr(opt, "reg_prune_until_iter", int(iteration))
    message = "[GNS] Regularized pruning finished at iter {}".format(iteration)
    tqdm.write("\n" + message)
    scene = state.get("scene")
    if scene is not None:
        append_log_lines(os.path.join(scene.model_path, "log.txt"), [message, ""])


def apply_gns_pruning(
    iteration: int,
    gaussians: GaussianModel3DGS,
    opt: Any,
    state: dict[str, Any],
    structure_update_mode: str = "single_pass",
) -> None:
    """
        Run GNS pruning logic for the current iteration.

        The first pruning iteration scales the opacity learning rate. During the
        pruning window it performs periodic threshold pruning, and on the final
        iteration it enforces the exact target budget.
    """
    if bool(state.get("gns_finished", False)):
        return

    reg_start = int(opt.reg_prune_from_iter)
    reg_end = int(opt.reg_prune_until_iter)
    if int(iteration) == reg_start and not state["gns_lr_scaled"]:
        resolve_regularized_prune_target_budget(gaussians, opt, state)
        gaussians.update_opacity_lr(float(opt.gns_opacity_lr_scale))
        state["gns_lr_scaled"] = True

    if reg_start <= int(iteration) < reg_end:
        prune_interval = 100
        if int(iteration) % prune_interval == 0:
            final_budget = resolve_regularized_prune_target_budget(gaussians, opt, state)
            apply_gns_threshold_prune(
                gaussians,
                float(opt.min_opacity),
                final_budget=final_budget,
                structure_update_mode=structure_update_mode,
            )
            if final_budget is not None and int(gaussians.get_xyz.shape[0]) <= int(final_budget):
                finish_gns_pruning(iteration, gaussians, opt, state)
        return

    if int(iteration) == reg_end:
        final_budget = resolve_regularized_prune_target_budget(gaussians, opt, state)
        if final_budget is not None:
            apply_gns_final_prune(
                gaussians,
                final_budget,
                structure_update_mode=structure_update_mode,
            )
        finish_gns_pruning(iteration, gaussians, opt, state)


def run_3dgs_pruning_method(context: TrainingContext, gaussians: GaussianModel3DGS, iteration: int) -> None:
    """
        Dispatch GNS pruning after densification.
    """
    opt = context.opt
    state = context.runtime_state
    structure_update_mode = str(context.method_config.get("structure_update_mode", "single_pass"))

    if context.method_config["use_gns_pruning"]:
        apply_gns_pruning(
            iteration,
            gaussians,
            opt,
            state,
            structure_update_mode=structure_update_mode,
        )
