"""
Regularization helpers used during loss construction.

The optimization stage computes the image loss first, then calls this module to
add the GNS opacity penalty during its regularized pruning window.
"""
import torch

from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from scene.methods.pruning_methods import resolve_regularized_prune_target_budget
from scene.training_context import TrainingContext


def compute_gns_pre_l1_regularization(gaussians: GaussianModel3DGS) -> torch.Tensor:
    """
        Compute GNS pre-activation opacity regularization.

        GNS regularizes raw opacity logits instead of activated opacity values,
        which makes the pruning pressure act before the sigmoid.
    """
    if gaussians._opacity.numel() == 0:
        return gaussians._opacity.sum() * 0.0
    return torch.mean(gaussians._opacity.flatten())


def apply_gns_opacity_regularization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    loss: torch.Tensor,
) -> torch.Tensor:
    """
        Add GNS opacity regularization during the pruning window.

        The strength is adjusted every few iterations by comparing the current
        budget boundary opacity with a simple target schedule.
    """
    opt = context.opt
    state = context.runtime_state
    if bool(state.get("gns_finished", False)):
        return loss

    reg_start = int(opt.reg_prune_from_iter)
    reg_end = int(opt.reg_prune_until_iter)
    if not (reg_start <= int(iteration) < reg_end):
        return loss

    opacity_flat = gaussians.get_opacity.flatten()
    final_budget = resolve_regularized_prune_target_budget(gaussians, opt, state)
    if final_budget is not None and opacity_flat.numel() > final_budget:
        kth_index = opacity_flat.shape[0] - int(final_budget) + 1
        adjustment_interval = 10
        tail_iterations = 100
        if state["opacity_min"] is None:
            state["opacity_min"] = torch.kthvalue(opacity_flat, kth_index).values.item()
        elif iteration % adjustment_interval == 0:
            remaining_ratio = 1.0 - (
                iteration - reg_start
            ) / max(float(reg_end - reg_start - tail_iterations), 1.0)
            opacity_goal = max(remaining_ratio * float(state["opacity_min"]), 0.0)
            current_value = torch.kthvalue(opacity_flat, kth_index).values.item()
            if current_value < opacity_goal * 0.9:
                opt.gns_opacity_reg *= 0.95
            elif current_value > opacity_goal * 1.1:
                opt.gns_opacity_reg *= 1.05
    return loss + float(opt.gns_opacity_reg) * compute_gns_pre_l1_regularization(gaussians)


def apply_regularization_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    loss: torch.Tensor,
) -> torch.Tensor:
    """
        Apply the optional GNS opacity penalty.
    """
    if context.method_config["use_gns_pruning"]:
        return apply_gns_opacity_regularization(context, gaussians, iteration, loss)
    return loss
