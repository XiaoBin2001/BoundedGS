"""
ABS-gradient statistics and budgeted long-axis densification.

The training loop calls this module after backpropagation and parameter updates.
It collects mean absolute screen-space gradients, then runs budgeted long-axis
splitting at the configured densification intervals.
"""
from __future__ import annotations

from typing import Any

import torch

from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from scene.training_context import TrainingContext


def _compute_current_budget(iteration: int, opt: Any) -> int:
    """
        Compute the active Gaussian budget for the current densification step.

        Early iterations use a square-root warmup toward the configured budget,
        which avoids adding too many Gaussians before the model has useful
        gradients.
    """
    budget = int(opt.budget)
    if budget > 0:
        max_budget = budget
    else:
        final_budget = int(opt.final_budget)
        budget_multiplier = float(opt.budget_multiplier)
        max_budget = max(int(final_budget * budget_multiplier), final_budget)
    densify_start = int(opt.densify_from_iter)
    densify_end = int(opt.densify_until_iter) - int(opt.budget_warmup_until_offset)
    if densify_end <= densify_start:
        return max_budget
    progress = (iteration - densify_start) / float(densify_end - densify_start)
    progress = min(max(progress, 0.0), 1.0)
    if progress >= 1.0:
        return max_budget
    return max(int(progress ** 0.5 * max_budget), 1)


def _collect_3dgs_densification_stats(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    render_state: dict[str, Any],
) -> torch.Tensor | None:
    """
        Collect absolute screen-space gradient statistics.

        Both full and compact projection paths supply absolute gradients.
    """
    visibility_filter = render_state["visibility_filter"]
    point_indices = render_state.get("viewspace_point_indices")
    direct_grad_values = render_state.get("densification_grad_values")
    if direct_grad_values is not None:
        targets = torch.nonzero(visibility_filter, as_tuple=False).squeeze(1)
        gaussians._accumulate_densification_stats(
            targets, direct_grad_values.index_select(0, targets))
    else:
        gaussians.add_densification_stats_abs(
            render_state["viewspace_point_tensor"], visibility_filter,
            point_indices=point_indices)
    return None


def _run_3dgs_improvedgs_budget_densification(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
) -> None:
    """
        Run ImprovedGS budgeted densification on densification intervals.

        The Gaussian model samples candidates from absolute screen-space
        gradients under the active budget.
    """
    opt = context.opt
    if iteration % int(opt.densification_interval) != 0:
        return
    gaussians.densify_and_prune_improved(
        float(opt.min_opacity),
        _compute_current_budget(iteration, opt),
        opt,
        iteration,
        float(context.scene.cameras_extent),
        str(context.method_config.get("structure_update_mode", "single_pass")),
    )


def run_3dgs_densification_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    render_state: dict[str, Any],
) -> None:
    """
        Collect ABS statistics and run the scheduled long-axis split.

        The caller already produced `render_state` during optimization. This
        function consumes that state to update accumulators and model structure.
    """
    opt = context.opt
    use_memory_lifecycle_trim = bool(
        context.method_config.get("use_memory_lifecycle_trim", True)
    )
    densify_from_iter = int(opt.densify_from_iter)
    densify_until_iter = int(opt.densify_until_iter)
    if not (densify_from_iter < iteration < densify_until_iter):
        if use_memory_lifecycle_trim:
            render_state.clear()
        return

    _collect_3dgs_densification_stats(context, gaussians, render_state)
    if use_memory_lifecycle_trim:
        render_state.clear()

    _run_3dgs_improvedgs_budget_densification(context, gaussians, iteration)
