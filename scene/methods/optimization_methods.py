"""
Optimization stage for one training iteration.

This module turns one sampled camera into gradients. It renders the view, masks
invalid pixels if needed, computes image reconstruction loss, adds optional
regularization, and calls backward so later stages can update parameters and
Gaussian structure.
"""
from types import SimpleNamespace
from typing import Any

import torch
from fused_ssim import fused_ssim

from gaussian_renderer import render
from gaussian_renderer.staged import (
    accumulate_indexed_raw_projection_backward,
    accumulate_gaussians_rect_backward,
    bisect_tile_aligned_rect,
    build_raw_group_gaussian_candidates,
    build_group_rectangles,
    count_gaussian_tile_intersections_rect,
    estimate_rect_raster_payload,
    finish_staged_projection_backward,
    project_gaussians_staged,
    rasterize_gaussians_rect,
)
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.loss_utils import l1_loss
from scene.methods.adaptive_leaves import get_cached_leaf_plan
from scene.methods.regularization_methods import apply_regularization_method
from utils.compact_gradients import CompactGradientAccumulator, can_use_compact_gradients
from scene.training_context import TrainingContext
from scene.training_runtime import TrainingLoopState


_MIB = 1024**2


def _iter_raster_leaves(context, camera, projection, group_rects, budget_bytes, stats, render_depth):
    for rect in group_rects:
        if budget_bytes <= 0:
            yield rect, 0, None
            continue
        plan = get_cached_leaf_plan(context, camera, rect, budget_bytes, render_depth)
        yield from plan.leaves(
            lambda rect: count_gaussian_tile_intersections_rect(projection, rect, render_depth=render_depth),
            lambda count, rect: estimate_rect_raster_payload(count, rect, render_depth=render_depth),
            bisect_tile_aligned_rect, budget_bytes, stats,
        )


def _run_prefetched_leaves(context, camera, projection, group_rects, budget_bytes, stats, render_leaf, render_depth):
    """Discover one Leaf ahead only after the current Raster consumes its count scan.

    count_rect overwrites GeometryState point_offsets/tiles_touched. Advancing the
    iterator before the current forward would corrupt the precounted Raster path.
    The backward uses projected attributes, not these scan scratch fields.
    """
    iterator = iter(_iter_raster_leaves(context, camera, projection, group_rects, budget_bytes, stats, render_depth))
    current = next(iterator, None)
    if current is None:
        return
    with camera.leaf_gt_prefetch() as prefetcher:
        prefetcher.submit(current[0])
        while current is not None:
            following = None

            def after_raster():
                nonlocal following
                following = next(iterator, None)
                if following is not None:
                    prefetcher.submit(following[0])

            render_leaf(*current, after_raster=after_raster)
            current = following
        summary = stats.setdefault('gt_prefetch', dict(requests=0, upload_bytes=0,
                                                      max_pending=0, cpu_wait_seconds=0.0))
        for key in ('requests', 'upload_bytes', 'cpu_wait_seconds'):
            summary[key] += prefetcher.stats[key]
        summary['max_pending'] = max(summary['max_pending'], prefetcher.stats['max_pending'])


def _enable_debug_if_needed(context: TrainingContext, iteration: int) -> None:
    """
        Enable renderer debug mode one iteration before the requested debug step.

        The rasterizer uses the flag during the next render call, so the switch
        is flipped just before the target iteration is rendered.
    """
    debug_from = int(getattr(context.runtime_args, "debug_from", -1))
    if (iteration - 1) == debug_from:
        context.pipe.debug = True


def _select_training_background(context: TrainingContext, loop_state: TrainingLoopState) -> torch.Tensor:
    """
        Choose the background color for this render.

        Random background is a training augmentation. Otherwise the dataset's
        fixed white or black background is reused from loop state.
    """
    return torch.rand((3), device="cuda") if context.opt.random_background else loop_state["background"]


def _compute_reconstruction_loss(image: torch.Tensor, gt_image: torch.Tensor, lambda_dssim: float) -> torch.Tensor:
    """
        Combine pixel L1 loss and fused SSIM into the standard 3DGS image loss.
    """
    l1_value = l1_loss(image, gt_image)
    if float(lambda_dssim) == 0.0:
        return l1_value
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    return (1.0 - float(lambda_dssim)) * l1_value + float(lambda_dssim) * (1.0 - ssim_value)


def _compute_non_render_loss(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
) -> torch.Tensor:
    """
        Build regularizers that must be evaluated once per camera, not once per rectangle.
    """
    loss = torch.zeros((), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    return apply_regularization_method(context, gaussians, iteration, loss)


def _resolve_depth_supervision(
    loop_state: TrainingLoopState,
    iteration: int,
    viewpoint_cam: Any,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    depth_weight = float(loop_state["depth_l1_weight"](iteration))
    if depth_weight <= 0.0 or not viewpoint_cam.depth_reliable:
        return 0.0, None, None
    return depth_weight, viewpoint_cam.invdepthmap_cuda(), viewpoint_cam.depth_mask_cuda()


def _add_depth_loss(
    loss: torch.Tensor,
    inv_depth: torch.Tensor,
    mono_invdepth: torch.Tensor | None,
    depth_mask: torch.Tensor | None,
    depth_weight: float,
) -> torch.Tensor:
    if depth_weight <= 0.0 or mono_invdepth is None or depth_mask is None:
        return loss
    depth_l1_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
    return loss + depth_weight * depth_l1_pure


def _build_bounded_render_state(
    viewpoint_cam: Any,
    bg: torch.Tensor,
    projection: Any,
    visibility_filter: torch.Tensor | None = None,
) -> dict[str, Any]:
    return {
        "bg": bg,
        "viewpoint_camera": viewpoint_cam,
        "viewspace_point_tensor": projection.viewspace_points,
        "viewspace_point_indices": projection.candidate_indices,
        "visibility_filter": projection.radii > 0 if visibility_filter is None else visibility_filter,
    }


def _run_independent_split_optimization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
    viewpoint_cam: Any,
    bg: torch.Tensor,
    group_grid: int,
) -> dict[str, Any]:
    """
        Treat every fixed crop as an independent full-projection training sample.

        Parameter gradients and view-level densification statistics are accumulated
        across all crops, while the optimizer and topology stages still run once for
        the original camera.
    """
    image_height = int(viewpoint_cam.image_height)
    image_width = int(viewpoint_cam.image_width)
    group_rects = build_group_rectangles(image_height, image_width, group_grid)
    point_count = int(gaussians.get_xyz.shape[0])
    full_visibility = torch.zeros(
        (point_count,),
        dtype=torch.bool,
        device=gaussians.get_xyz.device,
    )
    gradient_slice = _densification_gradient_slice(context, iteration)
    densification_screen_gradients = None
    if gradient_slice is not None:
        densification_screen_gradients = torch.zeros(
            (point_count, 2),
            dtype=torch.float32,
            device=gaussians.get_xyz.device,
        )

    depth_weight, mono_invdepth, depth_mask = _resolve_depth_supervision(
        loop_state,
        iteration,
        viewpoint_cam,
    )
    render_depth = depth_weight > 0.0
    total_pixels = image_height * image_width
    detached_losses: list[torch.Tensor] = []
    bounded_execution_stats: dict[str, Any] = {
        "base_group_grid": int(group_grid),
        "projection_group_count": len(group_rects),
        "projection_candidate_count_max": point_count,
        "projection_candidate_count_sum": point_count * len(group_rects),
        "projection_input_count_max": point_count,
        "projection_input_count_sum": point_count * len(group_rects),
        "projection_indexed_group_count": 0,
        "raster_leaf_count": 0,
        "split_count": 0,
        "count_call_count": 0,
        "forced_leaf_count": 0,
        "maximum_estimated_payload_bytes": 0,
        "maximum_actual_payload_bytes": 0,
    }
    context.runtime_state["active_bounded_execution_stats"] = bounded_execution_stats

    for rect in group_rects:
        projection = project_gaussians_staged(
            viewpoint_cam,
            gaussians,
            context.pipe,
            track_gradients=True,
            render_depth=render_depth,
        )
        rect_y, rect_x, rect_height, rect_width = rect
        raster_state = rasterize_gaussians_rect(
            projection,
            bg,
            rect,
            render_depth=render_depth,
        )
        raster_state.image.requires_grad_(True)
        if depth_weight > 0.0:
            raster_state.depth.requires_grad_(True)

        image = raster_state.image
        alpha_mask = viewpoint_cam.alpha_mask_crop_cuda(rect_y, rect_x, rect_height, rect_width)
        if alpha_mask is not None:
            image = image * alpha_mask
        gt_image = viewpoint_cam.original_image_crop_cuda(rect_y, rect_x, rect_height, rect_width)
        rect_loss = _compute_reconstruction_loss(image, gt_image, float(context.opt.lambda_dssim))
        if depth_weight > 0.0:
            rect_mono_invdepth = mono_invdepth[
                :,
                rect_y : rect_y + rect_height,
                rect_x : rect_x + rect_width,
            ]
            rect_depth_mask = depth_mask[
                :,
                rect_y : rect_y + rect_height,
                rect_x : rect_x + rect_width,
            ]
            rect_loss = _add_depth_loss(
                rect_loss,
                raster_state.depth,
                rect_mono_invdepth,
                rect_depth_mask,
                depth_weight,
            )
        rect_weight = float(rect_height * rect_width) / float(total_pixels)
        weighted_loss = rect_loss * rect_weight
        weighted_loss.backward()

        grad_depth = raster_state.depth.grad if depth_weight > 0.0 else None
        accumulate_gaussians_rect_backward(
            projection,
            bg,
            rect,
            raster_state,
            raster_state.image.grad,
            grad_depth,
        )
        actual_payload_bytes = (
            raster_state.binning_buffer.numel()
            + raster_state.image_buffer.numel()
            + raster_state.sample_buffer.numel()
            + raster_state.image.numel() * raster_state.image.element_size()
            + raster_state.depth.numel() * raster_state.depth.element_size()
        )
        bounded_execution_stats["raster_leaf_count"] += 1
        bounded_execution_stats["maximum_actual_payload_bytes"] = max(
            bounded_execution_stats["maximum_actual_payload_bytes"],
            int(actual_payload_bytes),
        )
        with torch.no_grad():
            full_visibility.logical_or_(projection.radii > 0)
            if densification_screen_gradients is not None:
                densification_screen_gradients.add_(
                    projection.projected_gradients[0][:, gradient_slice]
                )
        detached_losses.append(weighted_loss.detach())
        del image, gt_image, rect_loss, weighted_loss, raster_state, grad_depth
        if alpha_mask is not None:
            del alpha_mask
        if depth_weight > 0.0:
            del rect_mono_invdepth, rect_depth_mask

        # Raster backward has already reduced this crop into projected gradients.
        # Release all leaf-local state before the full projection backward allocates
        # its parameter-sized temporary gradients.
        finish_staged_projection_backward(projection)
        del projection

    non_render_loss = _compute_non_render_loss(
        context,
        gaussians,
        iteration,
    )
    if non_render_loss.requires_grad:
        non_render_loss.backward()

    densification_grad_values = None
    if densification_screen_gradients is not None:
        densification_grad_values = torch.linalg.vector_norm(
            densification_screen_gradients,
            dim=-1,
            keepdim=True,
        )
    projection_summary = SimpleNamespace(
        viewspace_points=torch.empty(
            (0, 4),
            dtype=gaussians.get_xyz.dtype,
            device=gaussians.get_xyz.device,
        ),
        candidate_indices=None,
    )
    render_state = _build_bounded_render_state(
        viewpoint_cam,
        bg,
        projection_summary,
        full_visibility,
    )
    render_state["densification_grad_values"] = densification_grad_values
    render_state["bounded_execution_stats"] = bounded_execution_stats
    reported_loss = torch.stack(detached_losses).sum() + non_render_loss.detach()
    return {
        "loss": reported_loss,
        "render_state": render_state,
    }


def _run_shared_projection_bounded_optimization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
    viewpoint_cam: Any,
    bg: torch.Tensor,
    group_grid: int,
) -> dict[str, Any]:
    """Project the full Gaussian set once, then stream bounded Raster Leaves."""
    depth_weight, mono_invdepth, depth_mask = _resolve_depth_supervision(
        loop_state,
        iteration,
        viewpoint_cam,
    )
    render_depth = depth_weight > 0.0
    projection = project_gaussians_staged(
        viewpoint_cam,
        gaussians,
        context.pipe,
        track_gradients=True,
        render_depth=render_depth,
    )
    group_rects = build_group_rectangles(
        projection.image_height,
        projection.image_width,
        group_grid,
    )
    total_pixels = projection.image_height * projection.image_width
    adaptive_budget_mib = float(
        context.method_config.get("leaf_raster_budget_mib", 0.0)
    )
    adaptive_budget_bytes = max(int(adaptive_budget_mib * _MIB), 0)
    point_count = int(projection.compact_radii.numel())
    background_is_zero = (
        not bool(context.dataset.white_background)
        and not bool(context.opt.random_background)
    )
    detached_losses: list[torch.Tensor] = []
    bounded_execution_stats: dict[str, Any] = {
        "base_group_grid": int(group_grid),
        "projection_group_count": 1,
        "projection_candidate_count_min": point_count,
        "projection_candidate_count_max": point_count,
        "projection_candidate_count_sum": point_count,
        "projection_input_count_max": point_count,
        "projection_input_count_sum": point_count,
        "projection_indexed_group_count": 0,
        "raster_leaf_count": 0,
        "split_count": 0,
        "count_call_count": 0,
        "forced_leaf_count": 0,
        "adaptive_budget_mib": adaptive_budget_mib,
        "maximum_estimated_payload_bytes": 0,
        "maximum_actual_payload_bytes": 0,
    }
    context.runtime_state["active_bounded_execution_stats"] = bounded_execution_stats

    def render_leaf(
        leaf_rect: tuple[int, int, int, int],
        estimated_payload_bytes: int = 0,
        precounted_num_rendered: int | None = None,
        after_raster=None,
    ) -> None:
        rect_y, rect_x, rect_height, rect_width = leaf_rect
        raster_state = rasterize_gaussians_rect(
            projection,
            bg,
            leaf_rect,
            precounted_num_rendered=precounted_num_rendered,
            render_depth=render_depth,
            background_is_zero=background_is_zero,
        )
        if after_raster is not None:
            after_raster()
        actual_payload_bytes = (
            raster_state.binning_buffer.numel()
            + raster_state.image_buffer.numel()
            + raster_state.sample_buffer.numel()
            + raster_state.image.numel() * raster_state.image.element_size()
            + raster_state.depth.numel() * raster_state.depth.element_size()
        )
        bounded_execution_stats["raster_leaf_count"] += 1
        bounded_execution_stats["maximum_estimated_payload_bytes"] = max(
            bounded_execution_stats["maximum_estimated_payload_bytes"],
            int(estimated_payload_bytes),
        )
        bounded_execution_stats["maximum_actual_payload_bytes"] = max(
            bounded_execution_stats["maximum_actual_payload_bytes"],
            int(actual_payload_bytes),
        )

        raster_state.image.requires_grad_(True)
        if depth_weight > 0.0:
            raster_state.depth.requires_grad_(True)
        image = raster_state.image
        alpha_mask = viewpoint_cam.alpha_mask_crop_cuda(
            rect_y,
            rect_x,
            rect_height,
            rect_width,
        )
        if alpha_mask is not None:
            image = image * alpha_mask
        gt_image = viewpoint_cam.original_image_crop_cuda(
            rect_y,
            rect_x,
            rect_height,
            rect_width,
        )
        rect_loss = _compute_reconstruction_loss(
            image,
            gt_image,
            float(context.opt.lambda_dssim),
        )
        if depth_weight > 0.0:
            rect_loss = _add_depth_loss(
                rect_loss,
                raster_state.depth,
                mono_invdepth[
                    :,
                    rect_y : rect_y + rect_height,
                    rect_x : rect_x + rect_width,
                ],
                depth_mask[
                    :,
                    rect_y : rect_y + rect_height,
                    rect_x : rect_x + rect_width,
                ],
                depth_weight,
            )
        weighted_loss = rect_loss * (
            float(rect_height * rect_width) / float(total_pixels)
        )
        weighted_loss.backward()
        accumulate_gaussians_rect_backward(
            projection,
            bg,
            leaf_rect,
            raster_state,
            raster_state.image.grad,
            raster_state.depth.grad if depth_weight > 0.0 else None,
            background_is_zero=background_is_zero,
        )
        detached_losses.append(weighted_loss.detach())

    if getattr(viewpoint_cam, 'training_gt_mode', 'leaf') == 'leaf_prefetch':
        _run_prefetched_leaves(context, viewpoint_cam, projection, group_rects, adaptive_budget_bytes,
                               bounded_execution_stats, render_leaf, render_depth)
        group_rects = []
    for leaf_rect, payload, count in _iter_raster_leaves(
        context, viewpoint_cam, projection, group_rects, adaptive_budget_bytes, bounded_execution_stats, render_depth,
    ):
        render_leaf(leaf_rect, payload, count)

    non_render_loss = _compute_non_render_loss(
        context,
        gaussians,
        iteration,
    )
    if non_render_loss.requires_grad:
        non_render_loss.backward()
    finish_staged_projection_backward(projection)
    render_state = _build_bounded_render_state(viewpoint_cam, bg, projection)
    render_state["bounded_execution_stats"] = bounded_execution_stats
    reported_loss = torch.stack(detached_losses).sum() + non_render_loss.detach()
    del projection
    return {
        "loss": reported_loss,
        "render_state": render_state,
    }


def _raw_gaussian_parameters(gaussians: GaussianModel3DGS) -> tuple[torch.Tensor, ...]:
    return (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    )


def _densification_gradient_slice(
    context: TrainingContext,
    iteration: int,
) -> slice | None:
    if not (
        int(context.opt.densify_from_iter)
        < int(iteration)
        < int(context.opt.densify_until_iter)
    ):
        return None
    return slice(2, 4)




def _run_group_compact_direct_optimization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
    viewpoint_cam: Any,
    bg: torch.Tensor,
    group_grid: int,
) -> dict[str, Any]:
    """Run indexed projection and bounded leaf-local rasterization for one view."""
    image_height = int(viewpoint_cam.image_height)
    image_width = int(viewpoint_cam.image_width)
    group_rects = build_group_rectangles(image_height, image_width, group_grid)
    raw_parameters = _raw_gaussian_parameters(gaussians)
    group_candidates = build_raw_group_gaussian_candidates(
        viewpoint_cam,
        raw_parameters,
        context.pipe,
        group_grid,
    )
    compact_gradients = None
    if (context.method_config.get("use_compact_parameter_gradients", False)
            and can_use_compact_gradients(raw_parameters, gaussians.optimizer)):
        compact_gradients = CompactGradientAccumulator(raw_parameters, group_candidates.indices)

    adaptive_budget_mib = float(
        context.method_config.get("leaf_raster_budget_mib", 0.0)
    )
    adaptive_budget_bytes = max(int(adaptive_budget_mib * _MIB), 0)
    point_count = int(raw_parameters[0].shape[0])
    full_visibility = torch.zeros(
        (point_count,),
        dtype=torch.bool,
        device=raw_parameters[0].device,
    )
    gradient_slice = _densification_gradient_slice(context, iteration)
    densification_screen_gradients = None
    if gradient_slice is not None:
        densification_screen_gradients = torch.zeros(
            (point_count, 2),
            dtype=torch.float32,
            device=raw_parameters[0].device,
        )

    total_pixels = image_height * image_width
    depth_weight, mono_invdepth, depth_mask = _resolve_depth_supervision(
        loop_state,
        iteration,
        viewpoint_cam,
    )
    render_depth = depth_weight > 0.0
    background_is_zero = (
        not bool(context.dataset.white_background)
        and not bool(context.opt.random_background)
    )
    detached_losses: list[torch.Tensor] = []
    bounded_execution_stats: dict[str, Any] = {
        "base_group_grid": int(group_grid),
        "projection_group_count": len(group_rects),
        "projection_candidate_count_min": min(
            (end - start for start, end in group_candidates.ranges),
            default=0,
        ),
        "projection_candidate_count_max": max(
            (end - start for start, end in group_candidates.ranges),
            default=0,
        ),
        "projection_candidate_count_sum": sum(
            end - start for start, end in group_candidates.ranges
        ),
        "projection_input_count_max": 0,
        "projection_input_count_sum": 0,
        "projection_indexed_group_count": 0,
        "raster_leaf_count": 0,
        "split_count": 0,
        "count_call_count": 0,
        "forced_leaf_count": 0,
        "adaptive_budget_mib": adaptive_budget_mib,
        "maximum_estimated_payload_bytes": 0,
        "maximum_actual_payload_bytes": 0,
    }
    context.runtime_state["active_bounded_execution_stats"] = bounded_execution_stats

    for group_index, rect in enumerate(group_rects):
        projection = project_gaussians_staged(
            viewpoint_cam,
            gaussians,
            context.pipe,
            track_gradients=True,
            render_depth=render_depth,
            raw_inputs=raw_parameters,
            preselected_indices=group_candidates.for_group(group_index),
        )
        if not projection.indexed_raw or projection.candidate_indices is None:
            raise RuntimeError("Compact Group training requires indexed raw projection.")

        projected_count = int(projection.compact_radii.numel())
        bounded_execution_stats["projection_input_count_max"] = max(
            bounded_execution_stats["projection_input_count_max"],
            projected_count,
        )
        bounded_execution_stats["projection_input_count_sum"] += projected_count
        bounded_execution_stats["projection_indexed_group_count"] += 1

        def render_projected_leaf(
            leaf_rect: tuple[int, int, int, int],
            estimated_payload_bytes: int = 0,
            precounted_num_rendered: int | None = None,
            after_raster=None,
        ) -> None:
            rect_y, rect_x, rect_height, rect_width = leaf_rect
            raster_state = rasterize_gaussians_rect(
                projection,
                bg,
                leaf_rect,
                precounted_num_rendered=precounted_num_rendered,
                render_depth=render_depth,
                background_is_zero=background_is_zero,
            )
            if after_raster is not None:
                after_raster()
            actual_payload_bytes = (
                raster_state.binning_buffer.numel()
                + raster_state.image_buffer.numel()
                + raster_state.sample_buffer.numel()
                + raster_state.image.numel() * raster_state.image.element_size()
                + raster_state.depth.numel() * raster_state.depth.element_size()
            )
            bounded_execution_stats["raster_leaf_count"] += 1
            bounded_execution_stats["maximum_estimated_payload_bytes"] = max(
                bounded_execution_stats["maximum_estimated_payload_bytes"],
                int(estimated_payload_bytes),
            )
            bounded_execution_stats["maximum_actual_payload_bytes"] = max(
                bounded_execution_stats["maximum_actual_payload_bytes"],
                int(actual_payload_bytes),
            )

            raster_state.image.requires_grad_(True)
            if depth_weight > 0.0:
                raster_state.depth.requires_grad_(True)
            image = raster_state.image
            alpha_mask = viewpoint_cam.alpha_mask_crop_cuda(
                rect_y,
                rect_x,
                rect_height,
                rect_width,
            )
            if alpha_mask is not None:
                image = image * alpha_mask
            gt_image = viewpoint_cam.original_image_crop_cuda(
                rect_y,
                rect_x,
                rect_height,
                rect_width,
            )
            rect_loss = _compute_reconstruction_loss(
                image,
                gt_image,
                float(context.opt.lambda_dssim),
            )
            if depth_weight > 0.0:
                rect_loss = _add_depth_loss(
                    rect_loss,
                    raster_state.depth,
                    mono_invdepth[
                        :,
                        rect_y : rect_y + rect_height,
                        rect_x : rect_x + rect_width,
                    ],
                    depth_mask[
                        :,
                        rect_y : rect_y + rect_height,
                        rect_x : rect_x + rect_width,
                    ],
                    depth_weight,
                )
            weighted_loss = rect_loss * (
                float(rect_height * rect_width) / float(total_pixels)
            )
            weighted_loss.backward()
            grad_image = raster_state.image.grad
            grad_depth = (
                raster_state.depth.grad
                if depth_weight > 0.0
                else None
            )
            detached_losses.append(weighted_loss.detach())

            accumulate_gaussians_rect_backward(
                projection,
                bg,
                leaf_rect,
                raster_state,
                grad_image,
                grad_depth,
                background_is_zero=background_is_zero,
            )

        if getattr(viewpoint_cam, 'training_gt_mode', 'leaf') == 'leaf_prefetch':
            _run_prefetched_leaves(context, viewpoint_cam, projection, [rect], adaptive_budget_bytes,
                                   bounded_execution_stats, render_projected_leaf, render_depth)
        elif adaptive_budget_bytes > 0:
            for leaf_rect, payload, count in _iter_raster_leaves(
                context, viewpoint_cam, projection, [rect], adaptive_budget_bytes, bounded_execution_stats, render_depth,
            ):
                render_projected_leaf(leaf_rect, payload, count)
        else:
            render_projected_leaf(rect)

        candidate_indices = projection.candidate_indices
        with torch.no_grad():
            full_visibility[candidate_indices] |= projection.compact_radii > 0
            if densification_screen_gradients is not None:
                densification_screen_gradients.index_add_(
                    0,
                    candidate_indices,
                    projection.projected_gradients[0][:, gradient_slice],
                )

        accumulate_indexed_raw_projection_backward(
            projection,
            raw_parameters,
            compact_gradients=compact_gradients,
        )
        # Release the completed Group before allocating the next projection.
        # Global parameter/densification gradients have already been accumulated.
        del projection, candidate_indices, render_projected_leaf

    del group_candidates

    non_render_loss = _compute_non_render_loss(
        context,
        gaussians,
        iteration,
    )
    if non_render_loss.requires_grad:
        non_render_loss.backward()

    densification_grad_values = None
    if densification_screen_gradients is not None:
        densification_grad_values = torch.linalg.vector_norm(
            densification_screen_gradients,
            dim=-1,
            keepdim=True,
        )
    projection_summary = SimpleNamespace(
        viewspace_points=torch.empty(
            (0, 4),
            dtype=raw_parameters[0].dtype,
            device=raw_parameters[0].device,
        ),
        candidate_indices=None,
    )
    render_state = _build_bounded_render_state(
        viewpoint_cam,
        bg,
        projection_summary,
        full_visibility,
    )
    render_state["densification_grad_values"] = densification_grad_values
    render_state["bounded_execution_stats"] = bounded_execution_stats
    return {
        "loss": torch.stack(detached_losses).sum() + non_render_loss.detach(),
        "render_state": render_state,
    }




def run_3dgs_optimization_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
    viewpoint_cam: Any,
) -> dict[str, Any]:
    """
        Render one camera and create gradients for this iteration.

        The returned `render_state` keeps only the render outputs needed by later
        stages, such as visibility and view-space gradient tensors.
    """
    pipe = context.pipe
    _enable_debug_if_needed(context, iteration)

    bg = _select_training_background(context, loop_state)
    group_grid = int(context.method_config["group_grid"])
    projection_mode = str(context.method_config.get("projection_mode", "full"))
    if group_grid > 1 or projection_mode == "group_compact":
        if projection_mode == "group_compact":
            return _run_group_compact_direct_optimization(
                context,
                gaussians,
                loop_state,
                iteration,
                viewpoint_cam,
                bg,
                group_grid,
            )
        if projection_mode == "simple_split":
            return _run_independent_split_optimization(
                context,
                gaussians,
                loop_state,
                iteration,
                viewpoint_cam,
                bg,
                group_grid,
            )
        return _run_shared_projection_bounded_optimization(
            context,
            gaussians,
            loop_state,
            iteration,
            viewpoint_cam,
            bg,
            group_grid,
        )

    depth_weight, mono_invdepth, depth_mask = _resolve_depth_supervision(
        loop_state,
        iteration,
        viewpoint_cam,
    )
    render_depth = depth_weight > 0.0
    render_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg,
        track_gradients=True,
        render_depth=render_depth,
    )
    image = render_pkg["render"]

    # Match the loss to valid image pixels only when the dataset provides an alpha mask.
    alpha_mask = viewpoint_cam.alpha_mask_cuda()
    if alpha_mask is not None:
        image = image * alpha_mask

    gt_image = viewpoint_cam.original_image_cuda()
    loss = _compute_reconstruction_loss(
        image,
        gt_image,
        float(context.opt.lambda_dssim),
    )

    # Add monocular inverse-depth supervision when the view marks its depth map as reliable.
    if depth_weight > 0.0:
        inv_depth = render_pkg["depth"]
        depth_l1_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
        loss = loss + depth_weight * depth_l1_pure

    loss = apply_regularization_method(context, gaussians, iteration, loss)
    loss.backward()
    return {
        "loss": loss,
        "render_state": {
            "bg": bg,
            "viewpoint_camera": viewpoint_cam,
            "viewspace_point_tensor": render_pkg["viewspace_points"],
            "visibility_filter": render_pkg["visibility_filter"],
        },
    }
