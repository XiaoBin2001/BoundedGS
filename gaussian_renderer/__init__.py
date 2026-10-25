"""
Renderer wrapper around the CUDA Gaussian rasterizer.

This module hides the low-level rasterizer argument layout from training code.
The main path renders normal 3DGS images and keeps gradients when training needs
them.
"""
from __future__ import annotations

import torch
import math
from diff_gaussian_rasterization_boundedgs import GaussianRasterizationSettings, GaussianRasterizer
from arguments import GroupParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel as GaussianModel3DGS

def _build_screenspace_points(pc: GaussianModel3DGS, track_gradients: bool) -> torch.Tensor:
    """
        Create the 2D placeholder tensor passed to the rasterizer.

        The CUDA backward pass writes gradients into this tensor. Channels 0-1
        are the standard screen-space gradients used by 3DGS; channels 2-3 store
        absolute gradients used by ABSGS and ImprovedGS.
    """
    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=track_gradients,
        device=pc.get_xyz.device,
    )
    if track_gradients:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
    return screenspace_points


def _build_3dgs_raster_settings(
    viewpoint_camera: Camera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float,
    inference_only: bool,
    render_depth: bool,
) -> GaussianRasterizationSettings:
    """
        Build rasterizer settings for the main 3DGS render call.

        These settings choose image size, camera transforms, active SH degree,
        and debug flags.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        inference_only=inference_only,
        render_depth=bool(render_depth) and not inference_only,
    )

def render_3dgs(
    viewpoint_camera: Camera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    track_gradients: bool = True,
    inference_only: bool = False,
    render_depth: bool = True,
) -> dict[str, torch.Tensor]:
    """
        Render one camera view with the standard 3DGS parameter tensors.

        The function always passes DC color, higher-order SH, scale, rotation,
        and opacity explicitly. During training it returns gradients and
        visibility data for densification; during inference it returns only the
        rendered image to reduce work.
    """
    if inference_only:
        track_gradients = False

    screenspace_points = _build_screenspace_points(pc, track_gradients)

    raster_settings = _build_3dgs_raster_settings(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier,
        inference_only,
        render_depth,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, depth_image = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        dc=pc.get_features_dc,
        shs=pc.get_features_rest,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )

    if inference_only:
        return {
            "render": rendered_image,
        }

    render_pkg = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth_image,
    }
    return render_pkg


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    track_gradients: bool = True,
    inference_only: bool = False,
    render_depth: bool = True,
) -> dict[str, torch.Tensor]:
    """
        Common render entry point used by training and evaluation code.

        It currently forwards directly to `render_3dgs`, which keeps call sites
        stable if more render variants are added later.
    """
    return render_3dgs(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        track_gradients=track_gradients,
        inference_only=inference_only,
        render_depth=render_depth,
    )
