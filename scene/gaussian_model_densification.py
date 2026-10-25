"""
GaussianModel structure-changing operations.

These methods modify the number of Gaussians or reset their trainable tensors.
They also keep optimizer state aligned with the new tensors, so training can
continue after clone, split, or prune operations.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from utils.general_utils import build_rotation, inverse_sigmoid


_MULTINOMIAL_MAX_CATEGORIES = 1 << 24


def _sample_compact_weighted_indices(
    scores: torch.Tensor,
    filter_mask: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """
        Sample positive-score entries that also pass the densification filter.

        CUDA multinomial rejects inputs longer than 2^24 even when almost all
        entries have zero weight, so sampling operates on compact candidates.
    """
    score_values = scores.detach().reshape(-1).float().clamp_min(0)
    flat_filter = filter_mask.reshape(-1)
    if score_values.numel() > flat_filter.numel():
        raise ValueError("Densification scores cannot outnumber the Gaussian filter.")

    eligible_mask = torch.logical_and(
        flat_filter[: score_values.numel()],
        score_values > 0,
    )
    candidate_indices = torch.nonzero(eligible_mask, as_tuple=False).squeeze(1)
    selected_count = min(max(int(sample_count), 0), int(candidate_indices.numel()))
    if selected_count == 0:
        return candidate_indices[:0]
    if selected_count == candidate_indices.numel():
        return candidate_indices

    candidate_weights = score_values[candidate_indices]
    if candidate_weights.numel() <= _MULTINOMIAL_MAX_CATEGORIES:
        local_indices = torch.multinomial(
            candidate_weights,
            selected_count,
            replacement=False,
        )
    else:
        exponential_keys = torch.empty_like(candidate_weights).exponential_()
        exponential_keys.div_(candidate_weights)
        local_indices = torch.topk(
            exponential_keys,
            selected_count,
            largest=False,
            sorted=False,
        ).indices
    return candidate_indices[local_indices]


class GaussianModelDensificationMixin:
    """
        Mixin that adds structure updates to GaussianModel.
    """

    def _new_densification_stats(self, count: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """One FP16 score accumulator and an exact integer observation count."""
        device = self.get_xyz.device if device is None else device
        return (
            torch.zeros((int(count), 1), dtype=torch.float16, device=device),
            torch.zeros((int(count), 1), dtype=torch.int16, device=device),
        )

    def _bind_densification_stats(self, accumulator: torch.Tensor, denom: torch.Tensor) -> None:
        # Both checkpoint field names share the same ABS accumulator allocation.
        self.xyz_gradient_accum = accumulator
        self.xyz_gradient_accum_abs = accumulator
        self.denom = denom

    def _reset_densification_stats(self) -> None:
        self._bind_densification_stats(*self._new_densification_stats(len(self.get_xyz)))
        self._densification_count_bound = 0

    def _restore_densification_stats(self, accumulator, denom) -> None:
        source = accumulator
        if source.dtype != torch.float16 or denom.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError("Expected FP16 ABS statistics and integer observation counts.")
        expected = (len(self.get_xyz), 1)
        if tuple(source.shape) != expected or tuple(denom.shape) != expected:
            raise ValueError("Checkpoint densification statistics must match the Gaussian count.")
        if not bool((denom >= 0).all()):
            raise ValueError("Checkpoint densification counts must be finite nonnegative integers.")
        bound = int(denom.max().item()) if denom.numel() else 0
        dtype = torch.int16 if bound <= torch.iinfo(torch.int16).max else torch.int32
        if bound > torch.iinfo(torch.int32).max:
            dtype = torch.int64
        self._bind_densification_stats(
            source.to(device=self.get_xyz.device, dtype=torch.float16),
            denom.to(device=self.get_xyz.device, dtype=dtype),
        )
        self._densification_count_bound = bound

    def _accumulate_densification_stats(self, targets, grad_values) -> None:
        """Compute in FP32, round stored scores to FP16, and never wrap counts."""
        # One collection per camera, with unique targets. The host-side upper bound
        # avoids a device reduction/synchronization on every training iteration.
        bound = self._densification_count_bound + 1
        if bound > torch.iinfo(self.denom.dtype).max:
            dtype = torch.int32 if self.denom.dtype == torch.int16 else torch.int64
            if bound > torch.iinfo(dtype).max:
                raise OverflowError("Densification observation count exceeds int64.")
            self.denom = self.denom.to(dtype)
        self._densification_count_bound = bound
        values = self.xyz_gradient_accum[targets].float() + grad_values.float()
        self.xyz_gradient_accum[targets] = values.to(torch.float16)
        self.denom[targets] += 1


    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        reset_params: bool = True,
    ) -> None:
        """
            Append newly created Gaussians to all parameter tensors.

            The optimizer receives the new tensors first, then the model fields
            are rebound to the optimizer-owned parameters. Gradient accumulators
            are reset because the Gaussian count has changed.
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self._reset_densification_stats()


    def _commit_densification_plan(
        self,
        old_remove_mask: torch.Tensor,
        additions: list[dict[str, torch.Tensor]],
        reset_stats: bool,
    ) -> str | None:
        """Apply one logical clone/split/prune plan with a single storage rebuild."""
        has_removals = bool(old_remove_mask.any().item())
        has_additions = any(int(extension["xyz"].shape[0]) > 0 for extension in additions)
        if not has_removals and not has_additions:
            if reset_stats:
                self.xyz_gradient_accum.zero_()
                self.denom.zero_()
                self._densification_count_bound = 0
            return None
        return self.rebuild_optimizer_structure(
            ~old_remove_mask,
            extension_dicts=additions,
            reset_stats=reset_stats,
        )

    def _append_and_prune_split_multi_pass(
        self,
        parent_mask: torch.Tensor,
        additions: dict[str, torch.Tensor],
    ) -> None:
        """Append split children, then prune their parents as a separate legacy operation."""
        addition_count = int(additions["xyz"].shape[0])
        self.densification_postfix(
            additions["xyz"],
            additions["f_dc"],
            additions["f_rest"],
            additions["opacity"],
            additions["scaling"],
            additions["rotation"],
        )
        prune_mask = torch.cat(
            (
                parent_mask,
                torch.zeros((addition_count,), dtype=torch.bool, device=parent_mask.device),
            ),
            dim=0,
        )
        self.prune_points_multi_pass(prune_mask)


    def _densification_gradient_targets(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
        point_indices: torch.Tensor | None,
        gradient_slice: slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_indices is None:
            return update_filter, viewspace_point_tensor.grad[update_filter, gradient_slice]
        compact_filter = update_filter.index_select(0, point_indices)
        return point_indices[compact_filter], viewspace_point_tensor.grad[compact_filter, gradient_slice]


    def add_densification_stats_abs(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
        point_indices: torch.Tensor | None = None,
    ) -> None:
        """
            Collect `abs(dx)` and `abs(dy)` written by CUDA backward.
            The mean reduction uses the shared score sum and observation count.
        """
        targets, gradients = self._densification_gradient_targets(
            viewspace_point_tensor,
            update_filter,
            point_indices,
            slice(2, 4),
        )
        grad_values = torch.norm(gradients, dim=-1, keepdim=True)
        self._accumulate_densification_stats(targets, grad_values)

    def densify_and_prune_improved(
        self,
        min_opacity: float,
        budget: int,
        opt: Any,
        iteration: int,
        scene_extent: float,
        structure_update_mode: str = "single_pass",
    ) -> None:
        """
            Select candidate Gaussians with absolute screen-space gradients, sample them under the active budget, then build long-axis Split children. In late training, fall back to gradient scores and relax the threshold while the budget is not full.
        """
        grad_values = self.xyz_gradient_accum_abs.float() / self.denom.float()
        grad_values[grad_values.isnan()] = 0.0
        scores = grad_values.squeeze(-1)

        min_grad = float(opt.densify_grad_threshold)
        late_densify_iter = max(
            int(opt.densify_until_iter) - 100,
            int(opt.densify_from_iter),
        )
        if self.get_opacity.shape[0] < budget and iteration > late_densify_iter:
            min_grad = min_grad / 1.5

        grad_qualifiers = torch.where(torch.norm(grad_values, dim=-1) >= min_grad, True, False)
        total_candidates = int(grad_qualifiers.sum().item())
        current_points = int(self.get_xyz.shape[0])
        current_budget = min(int(budget), total_candidates + current_points)
        split_budget = current_budget - current_points
        split_parent_mask = torch.zeros((current_points,), dtype=torch.bool, device=self.get_xyz.device)
        split_additions = None
        if split_budget > 0:
            split_parent_mask, split_additions, _ = self._build_long_axis_split(
                scores, split_budget, grad_qualifiers, opt.split_distance,
                opt.opacity_reduction)

        if iteration < late_densify_iter:
            old_opacity_prune_mask = (self.get_opacity < float(min_opacity)).squeeze()
        else:
            old_opacity_prune_mask = torch.zeros_like(split_parent_mask)
        additions = [] if split_additions is None else [split_additions]
        if str(structure_update_mode).strip().lower() == "multi_pass":
            if split_additions is not None:
                self._append_and_prune_split_multi_pass(split_parent_mask, split_additions)
            if iteration < late_densify_iter:
                opacity_prune_mask = (self.get_opacity < float(min_opacity)).squeeze()
                if bool(opacity_prune_mask.any().item()):
                    self.prune_points_multi_pass(opacity_prune_mask)
            return
        old_remove_mask = torch.logical_or(split_parent_mask, old_opacity_prune_mask)
        self._commit_densification_plan(
            old_remove_mask,
            additions,
            reset_stats=True,
        )

    def _build_long_axis_split(
        self,
        scores: torch.Tensor,
        budget: int,
        filter_mask: torch.Tensor,
        split_distance: float,
        opacity_reduction: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, int]:
        """
            Build long-axis split children and return their old-parent removal mask without mutating storage.
        """
        empty_parent_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
        if budget <= 0 or scores.numel() == 0 or not torch.any(filter_mask):
            return empty_parent_mask, None, 0

        selected_indices = _sample_compact_weighted_indices(
            scores,
            filter_mask,
            budget,
        )
        selected_count = int(selected_indices.numel())
        if selected_count == 0:
            return empty_parent_mask, None, 0

        selected_pts_mask = torch.zeros_like(filter_mask, dtype=torch.bool)
        selected_pts_mask[selected_indices] = True

        stds = self.get_scaling[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        axis_mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)
        axis_offsets = stds * axis_mask * 3.0 * float(split_distance)
        axis_offsets = torch.cat([axis_offsets, -axis_offsets], dim=0)

        rotation_mats = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        parent_xyz = self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_xyz = torch.bmm(rotation_mats, axis_offsets.unsqueeze(-1)).squeeze(-1) + parent_xyz

        split_distance_sq = float(split_distance) * float(split_distance)
        rate_w = max(1.0 - float(split_distance), 1e-6)
        rate_h = math.sqrt(max(1.0 - split_distance_sq, 1e-6))
        new_scales = stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h
        new_scaling = self.scaling_inverse_activation(new_scales)
        new_opacity = inverse_sigmoid(self.get_opacity[selected_pts_mask] * float(opacity_reduction)).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)

        additions = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        return selected_pts_mask, additions, selected_count
