"""
Memory-bounded rectangle rendering and metric evaluation.
"""
from __future__ import annotations

from typing import Any

import torch

from gaussian_renderer.staged import (
    build_group_rectangles,
    project_gaussians_staged,
    rasterize_gaussians_rect,
)
from utils.evaluation.lpips_metric import compute_lpips
from utils.loss_utils import ssim


@torch.inference_mode()
def render_bounded_inference_image(
    view: Any,
    gaussians: Any,
    pipeline: Any,
    background: torch.Tensor,
    group_grid: int,
) -> torch.Tensor:
    """
    Rasterize an inference image rectangle by rectangle.
    """
    projection = project_gaussians_staged(
        view,
        gaussians,
        pipeline,
        track_gradients=False,
    )
    rendering = torch.empty(
        (3, projection.image_height, projection.image_width),
        dtype=projection.means3d.dtype,
        device=projection.means3d.device,
    )
    for rect in build_group_rectangles(projection.image_height, projection.image_width, int(group_grid)):
        rect_y, rect_x, rect_height, rect_width = rect
        raster_state = rasterize_gaussians_rect(projection, background, rect)
        rendering[
            :,
            rect_y : rect_y + rect_height,
            rect_x : rect_x + rect_width,
        ].copy_(raster_state.image)
        del raster_state
    del projection
    return rendering


@torch.inference_mode()
def evaluate_bounded_inference_view(
    view: Any,
    gaussians: Any,
    pipeline: Any,
    background: torch.Tensor,
    group_grid: int,
    lpips_model: Any | None,
    report_lpips: bool,
    return_rendering: bool = False,
    low_memory_lpips: bool = False,
) -> dict[str, torch.Tensor | None]:
    """
    Evaluate full-view L1/PSNR, approximate local SSIM, and optional global LPIPS.
    """
    rendering = torch.clamp(
        render_bounded_inference_image(
            view,
            gaussians,
            pipeline,
            background,
            group_grid,
        ),
        0.0,
        1.0,
    )
    total_pixels = int(view.image_height) * int(view.image_width)
    absolute_error = torch.zeros((), dtype=torch.float64, device=rendering.device)
    squared_error = torch.zeros((), dtype=torch.float64, device=rendering.device)
    squared_error_channels = torch.zeros((3,), dtype=torch.float64, device=rendering.device)
    weighted_ssim = torch.zeros((), dtype=torch.float64, device=rendering.device)

    for rect in build_group_rectangles(int(view.image_height), int(view.image_width), int(group_grid)):
        rect_y, rect_x, rect_height, rect_width = rect
        rendered_crop = rendering[
            :,
            rect_y : rect_y + rect_height,
            rect_x : rect_x + rect_width,
        ]
        gt_crop = torch.clamp(
            view.original_image_crop_cuda(rect_y, rect_x, rect_height, rect_width),
            0.0,
            1.0,
        )
        difference = rendered_crop - gt_crop
        absolute_error += difference.abs().sum(dtype=torch.float64)
        squared_error += difference.square().sum(dtype=torch.float64)
        squared_error_channels += difference.square().sum(dim=(1, 2), dtype=torch.float64)
        rect_weight = float(rect_height * rect_width) / float(total_pixels)
        rendered_batch = rendered_crop.unsqueeze(0)
        gt_batch = gt_crop.unsqueeze(0)
        weighted_ssim += ssim(rendered_batch, gt_batch).double() * rect_weight
        del difference, rendered_batch, gt_batch, rendered_crop, gt_crop

    element_count = float(total_pixels * 3)
    l1_value = absolute_error / element_count
    mse_value = squared_error / element_count
    psnr_value = 20.0 * torch.log10(1.0 / torch.sqrt(mse_value))
    channel_mse = squared_error_channels / float(total_pixels)
    psnr_channel_mean = (20.0 * torch.log10(1.0 / torch.sqrt(channel_mse))).mean()
    lpips_value = None
    if report_lpips and lpips_model is not None:
        gt_image = torch.clamp(view.original_image_cuda(), 0.0, 1.0)
        lpips_value = compute_lpips(
            lpips_model,
            rendering.unsqueeze(0),
            gt_image.unsqueeze(0),
            low_memory_lpips,
        ).mean().double()
        del gt_image
    return {
        "rendering": rendering if return_rendering else None,
        "l1": l1_value,
        "psnr": psnr_value,
        "psnr_channel_mean": psnr_channel_mean,
        "ssim": weighted_ssim,
        "lpips": lpips_value,
    }
