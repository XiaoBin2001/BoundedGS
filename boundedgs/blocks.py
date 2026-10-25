"""Appendix C fixed-block experiments; the caller performs the final Adam step.

Blocks use area-weighted L1. Independent updates reproject from the parameters
updated by the previous block; accumulated updates merge only within one view.
These optional protocols do not participate in the default training loop.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from diff_gaussian_rasterization_boundedgs import _C
from gaussian_renderer.staged import (
    accumulate_gaussians_rect_backward,
    accumulate_indexed_raw_projection_backward,
    build_group_rectangles,
    project_gaussians_staged,
    rasterize_gaussians_rect,
)
from scene.methods.optimization_methods import _raw_gaussian_parameters, _compute_reconstruction_loss
from utils.compact_gradients import CompactGradientAccumulator, can_use_compact_gradients
from utils.loss_utils import l1_loss


class BlockGradientAccumulator(CompactGradientAccumulator):
    """View-local sparse union for the Appendix C.1 accumulated baseline."""

    @torch.no_grad()
    def prepare(self):
        if self.index_map is not None:
            return
        n = self.parameters[0].shape[0]
        device = self.parameters[0].device
        selected = torch.zeros(n, dtype=torch.bool, device=device)
        selected[self.candidates.long()] = True
        # Preserve earlier blocks in this view, including rows absent this block.
        for i, parameter in enumerate(self.parameters):
            if i == 3 or parameter.grad is None:
                continue
            if not parameter.grad.is_sparse:
                raise RuntimeError("Compact accumulation requires sparse pending gradients")
            selected[parameter.grad.coalesce().indices()[0]] = True
        indices = selected.nonzero(as_tuple=True)[0]
        del selected
        self.index_map = torch.full((n,), -1, dtype=torch.int32, device=device)
        self.index_map[indices] = torch.arange(indices.numel(), dtype=torch.int32, device=device)
        for i, parameter in enumerate(self.parameters):
            if i == 3:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                continue
            values = torch.zeros((indices.numel(), *parameter.shape[1:]),
                                 dtype=parameter.dtype, device=device)
            if parameter.grad is not None:
                previous = parameter.grad.coalesce()
                values.index_copy_(0, self.index_map[previous.indices()[0]].long(), previous.values())
                del previous
            gradient = torch.sparse_coo_tensor(indices.unsqueeze(0), values,
                size=parameter.shape, is_coalesced=True)
            gradient._boundedgs_index_map = self.index_map
            parameter.grad = gradient


def candidate_ids_for_group(camera, raw, pipe, rect):
    """Use the production raw candidate kernel with just this rectangle's boundaries."""
    y, x, height, width = rect
    device = raw[0].device
    indices, offsets = _C.build_gaussian_group_candidates_raw(
        raw[0], raw[3], raw[4], 1.0,
        camera.world_view_transform.contiguous(),
        camera.full_proj_transform.contiguous(),
        math.tan(float(camera.FoVx) * 0.5),
        math.tan(float(camera.FoVy) * 0.5),
        int(camera.image_height), int(camera.image_width),
        torch.tensor([y, y + height], dtype=torch.int32, device=device),
        torch.tensor([x, x + width], dtype=torch.int32, device=device),
        bool(pipe.debug),
    )
    return indices, offsets


def run_compact_blocks(context, gaussians, loop_state, iteration, camera, grid,
                       independent):
    if float(context.opt.lambda_dssim) != 0.0 or bool(context.opt.random_background):
        raise ValueError("This protocol requires pure L1 and fixed background")
    if float(context.method_config.get("leaf_raster_budget_mib", 0)) != 0.0:
        raise ValueError("No additional Raster Leaf subdivision is allowed")
    gt_mode = getattr(camera, "training_gt_mode", None)
    if gt_mode not in {"leaf", "leaf_prefetch"}:
        raise ValueError("Fixed-block experiments require leaf or leaf_prefetch GT")
    raw = _raw_gaussian_parameters(gaussians)
    if not can_use_compact_gradients(raw, gaussians.optimizer):
        raise RuntimeError("Compact gradients require GaussianAdam and cleared pending gradients")
    background = loop_state["background"]
    render_depth = False
    background_is_zero = (
        not bool(context.dataset.white_background)
        and not bool(context.opt.random_background)
    )
    group_rects = build_group_rectangles(camera.image_height, camera.image_width, grid)
    total_pixels = int(camera.image_height) * int(camera.image_width)
    reported_losses = []
    stats = {
        "base_group_grid": grid, "projection_group_count": len(group_rects),
        "projection_indexed_group_count": 0,
        "projection_candidate_count_max": 0, "projection_candidate_count_sum": 0,
        "projection_input_count_max": 0, "projection_input_count_sum": 0,
        "raster_leaf_count": 0, "split_count": 0, "count_call_count": 0,
        "forced_leaf_count": 0, "adaptive_budget_mib": 0.0,
        "maximum_actual_payload_bytes": 0,
        "optimizer_steps_inside_render": len(group_rects) - 1 if independent else 0,
        "optimizer_steps_at_worker_boundary": 1,
        "block_update_semantics": "independent" if independent else "accumulated",
    }
    context.runtime_state["active_bounded_execution_stats"] = stats

    prefetch_context = camera.leaf_gt_prefetch() if gt_mode == "leaf_prefetch" else nullcontext()
    with prefetch_context as prefetcher:
        if prefetcher is not None:
            prefetcher.submit(group_rects[0])
        for group_index, rect in enumerate(group_rects):
            y, x, height, width = rect
            candidate_indices, offsets = candidate_ids_for_group(camera, raw, context.pipe, rect)
            assert offsets.numel() == 2
            candidate_count = int(candidate_indices.numel())
            stats["projection_candidate_count_max"] = max(stats["projection_candidate_count_max"], candidate_count)
            stats["projection_candidate_count_sum"] += candidate_count
            del offsets
            projection = project_gaussians_staged(
                camera, gaussians, context.pipe,
                track_gradients=True, render_depth=render_depth,
                raw_inputs=raw, preselected_indices=candidate_indices,
            )
            if not projection.indexed_raw or projection.candidate_indices is None:
                raise RuntimeError("Indexed raw projection was not used")
            projected_count = int(projection.compact_radii.numel())
            stats["projection_input_count_max"] = max(stats["projection_input_count_max"], projected_count)
            stats["projection_input_count_sum"] += projected_count
            stats["projection_indexed_group_count"] += 1

            raster_state = rasterize_gaussians_rect(
                projection, background, rect,
                render_depth=render_depth, background_is_zero=background_is_zero,
            )
            if prefetcher is not None and group_index + 1 < len(group_rects):
                prefetcher.submit(group_rects[group_index + 1])
            payload = (
                raster_state.binning_buffer.numel() + raster_state.image_buffer.numel()
                + raster_state.sample_buffer.numel()
                + raster_state.image.numel() * raster_state.image.element_size()
                + raster_state.depth.numel() * raster_state.depth.element_size()
            )
            stats["maximum_actual_payload_bytes"] = max(stats["maximum_actual_payload_bytes"], int(payload))
            stats["raster_leaf_count"] += 1
            raster_state.image.requires_grad_(True)
            image = raster_state.image
            mask = camera.alpha_mask_crop_cuda(y, x, height, width)
            if mask is not None:
                image = image * mask
            gt = camera.original_image_crop_cuda(y, x, height, width)
            loss = _compute_reconstruction_loss(image, gt, 0.0)
            weighted_loss = loss * (float(height * width) / float(total_pixels))
            weighted_loss.backward()
            reported_losses.append(weighted_loss.detach())
            accumulate_gaussians_rect_backward(
                projection, background, rect, raster_state, raster_state.image.grad,
                None, background_is_zero=background_is_zero,
            )
            del image, gt, loss, weighted_loss, mask, raster_state

            accumulator_type = CompactGradientAccumulator if independent else BlockGradientAccumulator
            compact_gradients = accumulator_type(raw, candidate_indices)
            accumulate_indexed_raw_projection_backward(
                projection, raw, compact_gradients=compact_gradients,
            )
            del projection, candidate_indices, compact_gradients
            if independent and group_index + 1 < len(group_rects):
                gaussians.optimizer.step(int(raw[0].shape[0]), iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
        if prefetcher is not None:
            stats["gt_prefetch"] = dict(prefetcher.stats)

    return {"loss": torch.stack(reported_losses).sum(),
            "render_state": {"bounded_execution_stats": stats}}


class _OriginalStyleRect(torch.autograd.Function):
    """Bind full projection and rectangle raster backward to one autograd edge.

    The existing rectangle CUDA primitives are used because the Original CUDA entry
    point has no rectangle argument.  No crop-camera transform, candidate selection,
    DirectGrad, or separate Python autograd.backward call is used.
    """

    @staticmethod
    def forward(ctx, means3d, means2d, dc, sh, opacity, scales, rotations,
                background, camera, rect, antialiasing):
        y, x, height, width = rect
        tanfovx = math.tan(float(camera.FoVx) / 2)
        tanfovy = math.tan(float(camera.FoVy) / 2)
        radii, geometry = _C.project_gaussians(
            means3d, opacity, scales, rotations, 1.0,
            camera.world_view_transform, camera.full_proj_transform,
            tanfovx, tanfovy, int(camera.image_height), int(camera.image_width),
            dc, sh, int(camera._experiment_sh_degree), camera.camera_center,
            False, bool(antialiasing), False,
        )
        raster = _C.rasterize_gaussians_rect(
            background, radii, geometry,
            int(camera.image_height), int(camera.image_width),
            int(y), int(x), int(height), int(width),
            False, False, False,
        )
        ctx.num_rendered = int(raster[0])
        ctx.num_buckets = int(raster[1])
        ctx.rect = rect
        ctx.tanfovx = tanfovx
        ctx.tanfovy = tanfovy
        ctx.degree = int(camera._experiment_sh_degree)
        ctx.image_height = int(camera.image_height)
        ctx.image_width = int(camera.image_width)
        ctx.antialiasing = bool(antialiasing)
        ctx.save_for_backward(
            means3d, dc, sh, opacity, scales, rotations, background,
            camera.world_view_transform, camera.full_proj_transform,
            camera.camera_center, radii, geometry,
            raster[2], raster[3], raster[4], raster[5], raster[6],
        )
        return raster[2]

    @staticmethod
    def backward(ctx, grad_image):
        (means3d, dc, sh, opacity, scales, rotations, background,
         viewmatrix, projmatrix, campos, radii, geometry,
         image, depth, binning, image_buffer, sample_buffer) = ctx.saved_tensors
        y, x, height, width = ctx.rect
        n = int(means3d.shape[0])
        opts = dict(device=means3d.device, dtype=torch.float32)
        projected = (
            torch.zeros((n, 4), **opts),
            torch.zeros((n, 4), **opts),
            torch.zeros((n, 1), **opts),
            torch.zeros((n, 3), **opts),
            torch.empty(0, **opts),
        )
        if ctx.num_rendered:
            _C.rasterize_gaussians_rect_backward(
                background, radii, geometry,
                ctx.image_height, ctx.image_width,
                int(y), int(x), int(height), int(width),
                image, depth, grad_image.contiguous(), torch.empty(0, **opts),
                ctx.num_rendered, binning, image_buffer,
                ctx.num_buckets, sample_buffer,
                *projected, False, False,
            )
        gradients = _C.project_gaussians_backward(
            means3d, radii, opacity, scales, rotations, 1.0,
            viewmatrix, projmatrix, ctx.tanfovx, ctx.tanfovy,
            ctx.image_height, ctx.image_width,
            dc, sh, ctx.degree, campos, geometry,
            *projected, ctx.antialiasing, False,
        )
        return (gradients[0], projected[0], gradients[1], gradients[2],
                gradients[3], gradients[4], gradients[5],
                None, None, None, None)


def run_original_blocks(context, gaussians, loop_state, iteration, camera, grid):
    if float(context.opt.lambda_dssim) != 0.0 or bool(context.opt.random_background):
        raise ValueError("This protocol requires pure L1 and fixed background")
    background = loop_state["background"]
    camera._experiment_sh_degree = int(gaussians.active_sh_degree)
    n = int(gaussians.get_xyz.shape[0])
    total_pixels = int(camera.image_height) * int(camera.image_width)
    group_rects = build_group_rectangles(camera.image_height, camera.image_width, grid)
    reported_losses = []
    for group_index, rect in enumerate(group_rects):
        y, x, height, width = rect
        means2d = torch.zeros((n, 4), dtype=torch.float32,
                              device=gaussians.get_xyz.device, requires_grad=True)
        image = _OriginalStyleRect.apply(
            gaussians.get_xyz, means2d,
            gaussians.get_features_dc, gaussians.get_features_rest,
            gaussians.get_opacity, gaussians.get_scaling, gaussians.get_rotation,
            background, camera, rect, bool(context.pipe.antialiasing),
        )
        mask = camera.alpha_mask_crop_cuda(y, x, height, width)
        if mask is not None:
            image = image * mask
        gt = camera.original_image_crop_cuda(y, x, height, width)
        area_weight = float(height * width) / float(total_pixels)
        local_loss = l1_loss(image, gt)
        backward_loss = local_loss * area_weight
        backward_loss.backward()
        reported_losses.append((local_loss.detach() * area_weight))
        del backward_loss, local_loss, gt, image, mask, means2d
        if group_index + 1 < len(group_rects):
            gaussians.optimizer.step(n, iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)
    return {
        "loss": torch.stack(reported_losses).sum(),
        "render_state": {"bounded_execution_stats": {
            "base_group_grid": grid,
            "projection_group_count": len(group_rects),
            "projection_candidate_count_max": n,
            "projection_candidate_count_sum": len(group_rects) * n,
            "raster_leaf_count": len(group_rects),
            "optimizer_steps_inside_render": len(group_rects) - 1,
            "optimizer_steps_at_worker_boundary": 1,
            "loss_weighting": "area",
            "block_update_semantics": "independent",
        }},
    }
