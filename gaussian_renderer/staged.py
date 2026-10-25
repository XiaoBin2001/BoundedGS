"""
Staged rectangle rendering helpers for high-resolution training.

The staged path projects either the full Gaussian set or a preselected indexed
candidate set, rasterizes one tile-aligned rectangle at a time, and finally maps
accumulated gradients back to the full parameter tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from diff_gaussian_rasterization_boundedgs import _C


TILE_SIZE = 16


@dataclass
class StagedProjection:
    candidate_indices: torch.Tensor | None
    means3d: torch.Tensor
    dc: torch.Tensor
    sh: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    radii: torch.Tensor
    compact_radii: torch.Tensor
    geometry: torch.Tensor
    viewspace_points: torch.Tensor
    projected_gradients: list[torch.Tensor] | None
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    tanfovx: float
    tanfovy: float
    image_height: int
    image_width: int
    sh_degree: int
    scale_modifier: float
    antialiasing: bool
    debug: bool
    indexed_raw: bool = False


@dataclass
class RectRasterState:
    num_rendered: int
    num_buckets: int
    image: torch.Tensor
    depth: torch.Tensor
    binning_buffer: torch.Tensor
    image_buffer: torch.Tensor
    sample_buffer: torch.Tensor


@dataclass(frozen=True)
class RectRasterEstimate:
    num_rendered: int
    binning_bytes: int
    image_bytes: int
    sample_bytes: int
    output_bytes: int

    @property
    def payload_bytes(self) -> int:
        return (
            self.binning_bytes
            + self.image_bytes
            + self.sample_bytes
            + self.output_bytes
        )

@dataclass
class GroupGaussianCandidates:
    indices: torch.Tensor
    ranges: list[tuple[int, int]]

    def for_group(self, group_index: int) -> torch.Tensor:
        start, end = self.ranges[int(group_index)]
        return self.indices.narrow(0, start, end - start)


def _require_leaf_raster_api() -> None:
    required_names = (
        "count_gaussian_tile_intersections_rect",
        "estimate_rect_raster_buffer_sizes",
        "rasterize_gaussians_rect_precounted",
    )
    missing_names = [name for name in required_names if not hasattr(_C, name)]
    if missing_names:
        raise RuntimeError(
            "The installed diff_gaussian_rasterization_boundedgs extension does not contain the Leaf raster API "
            "({}). Reinstall the extension in the active environment.".format(
                ", ".join(missing_names)
            )
        )


def _require_staged_raster_api() -> None:
    required_names = (
        "project_gaussians",
        "rasterize_gaussians_rect",
        "rasterize_gaussians_rect_backward",
        "project_gaussians_backward",
    )
    missing_names = [name for name in required_names if not hasattr(_C, name)]
    if missing_names:
        raise RuntimeError(
            "The installed diff_gaussian_rasterization_boundedgs extension does not contain the staged raster API "
            "({}). Reinstall submodules/diff-gaussian-rasterization in the active environment.".format(
                ", ".join(missing_names)
            )
        )


def _tile_boundaries(size: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("group_grid must be positive.")
    if parts > 1 and size < parts * TILE_SIZE:
        raise ValueError(
            "Image dimension {} is too small for group_grid={} with {}-pixel raster tiles.".format(
                size,
                parts,
                TILE_SIZE,
            )
        )
    boundaries = [0]
    for index in range(1, parts):
        aligned = int(round((size * index / parts) / TILE_SIZE)) * TILE_SIZE
        aligned = max(boundaries[-1] + TILE_SIZE, min(size - TILE_SIZE, aligned))
        boundaries.append(aligned)
    boundaries.append(size)
    return boundaries


def build_group_rectangles(image_height: int, image_width: int, group_grid: int) -> list[tuple[int, int, int, int]]:
    """
    Return tile-aligned `(y, x, height, width)` rectangles in row-major order.
    """
    grid = int(group_grid)
    y_boundaries = _tile_boundaries(int(image_height), grid)
    x_boundaries = _tile_boundaries(int(image_width), grid)
    return [
        (
            y_boundaries[row],
            x_boundaries[column],
            y_boundaries[row + 1] - y_boundaries[row],
            x_boundaries[column + 1] - x_boundaries[column],
        )
        for row in range(grid)
        for column in range(grid)
    ]


def bisect_tile_aligned_rect(
    rect: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Split a rectangle along its longer tile dimension without moving its outer boundary."""
    rect_y, rect_x, rect_height, rect_width = (int(value) for value in rect)
    tile_rows = (rect_height + TILE_SIZE - 1) // TILE_SIZE
    tile_columns = (rect_width + TILE_SIZE - 1) // TILE_SIZE
    if tile_columns >= tile_rows and tile_columns > 1:
        first_width = (tile_columns // 2) * TILE_SIZE
        return (
            (rect_y, rect_x, rect_height, first_width),
            (rect_y, rect_x + first_width, rect_height, rect_width - first_width),
        )
    if tile_rows > 1:
        first_height = (tile_rows // 2) * TILE_SIZE
        return (
            (rect_y, rect_x, first_height, rect_width),
            (rect_y + first_height, rect_x, rect_height - first_height, rect_width),
        )
    return None



def build_raw_group_gaussian_candidates(
    viewpoint_camera: Any,
    raw_inputs: tuple[torch.Tensor, ...],
    pipe: Any,
    group_grid: int,
    *,
    scaling_modifier: float = 1.0,
) -> GroupGaussianCandidates:
    """Build one conservative uint32 global-ID table from raw Gaussian parameters."""
    grid = int(group_grid)
    if grid < 1:
        raise ValueError("group_grid must be positive.")
    if not hasattr(_C, "build_gaussian_group_candidates_raw"):
        raise RuntimeError(
            "The installed diff_gaussian_rasterization_boundedgs extension does not contain "
            "build_gaussian_group_candidates_raw. Reinstall the extension in the active environment."
        )

    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
    device = raw_inputs[0].device
    indices, offsets = _C.build_gaussian_group_candidates_raw(
        raw_inputs[0],
        raw_inputs[3],
        raw_inputs[4],
        float(scaling_modifier),
        viewpoint_camera.world_view_transform.contiguous(),
        viewpoint_camera.full_proj_transform.contiguous(),
        math.tan(float(viewpoint_camera.FoVx) * 0.5),
        math.tan(float(viewpoint_camera.FoVy) * 0.5),
        image_height,
        image_width,
        torch.tensor(_tile_boundaries(image_height, grid), dtype=torch.int32, device=device),
        torch.tensor(_tile_boundaries(image_width, grid), dtype=torch.int32, device=device),
        bool(pipe.debug),
    )
    host_offsets = [int(value) for value in offsets.cpu().tolist()]
    ranges = list(zip(host_offsets[:-1], host_offsets[1:]))
    return GroupGaussianCandidates(indices=indices, ranges=ranges)


def project_gaussians_staged(
    viewpoint_camera: Any,
    gaussians: Any,
    pipe: Any,
    *,
    scaling_modifier: float = 1.0,
    track_gradients: bool,
    render_depth: bool = True,
    raw_inputs: tuple[torch.Tensor, ...] | None = None,
    preselected_indices: torch.Tensor | None = None,
) -> StagedProjection:
    """
    Project full-view activated parameters or a preselected raw candidate set.
    """
    _require_staged_raster_api()
    raw_input_mode = raw_inputs is not None
    if preselected_indices is not None and not raw_input_mode:
        raise ValueError("Preselected Gaussian indices are only supported for raw-input projection.")
    if raw_input_mode:
        if preselected_indices is None:
            raise ValueError("Raw-input staged projection requires preselected candidate indices.")
        source_inputs = raw_inputs
    else:
        source_inputs = (
            gaussians.get_xyz,
            gaussians.get_features_dc,
            gaussians.get_features_rest,
            gaussians.get_opacity,
            gaussians.get_scaling,
            gaussians.get_rotation,
        )
    means3d, dc, sh, opacities, scales, rotations = source_inputs
    if not track_gradients and not raw_input_mode:
        means3d = means3d.detach()
        dc = dc.detach()
        sh = sh.detach()
        opacities = opacities.detach()
        scales = scales.detach()
        rotations = rotations.detach()
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
    tanfovx = math.tan(float(viewpoint_camera.FoVx) * 0.5)
    tanfovy = math.tan(float(viewpoint_camera.FoVy) * 0.5)
    scale_modifier = float(scaling_modifier)

    source_inputs = (means3d, dc, sh, opacities, scales, rotations)
    candidate_indices = preselected_indices
    indexed_raw = raw_input_mode
    if indexed_raw:
        candidate_indices = candidate_indices.to(
            dtype=torch.int32,
            device=means3d.device,
        ).contiguous()
        required_names = (
            "project_gaussians_indexed_raw",
            "project_gaussians_indexed_raw_backward",
        )
        missing_names = [name for name in required_names if not hasattr(_C, name)]
        if missing_names:
            raise RuntimeError(
                "The installed diff_gaussian_rasterization_boundedgs extension does not contain "
                "the indexed raw projection API ({}). Reinstall the extension in the "
                "active environment.".format(", ".join(missing_names))
            )

    if indexed_raw:
        compact_radii, geometry = _C.project_gaussians_indexed_raw(
            candidate_indices,
            means3d,
            opacities,
            scales,
            rotations,
            scale_modifier,
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            tanfovx,
            tanfovy,
            image_height,
            image_width,
            dc,
            sh,
            int(gaussians.active_sh_degree),
            viewpoint_camera.camera_center,
            False,
            bool(pipe.antialiasing),
            bool(pipe.debug),
        )
    else:
        projection_args = (
            source_inputs[0],
            source_inputs[3],
            source_inputs[4],
            source_inputs[5],
            scale_modifier,
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            tanfovx,
            tanfovy,
            image_height,
            image_width,
            source_inputs[1],
            source_inputs[2],
            int(gaussians.active_sh_degree),
            viewpoint_camera.camera_center,
            False,
            bool(pipe.antialiasing),
        )
        compact_radii, geometry = _C.project_gaussians(
            *projection_args,
            bool(pipe.debug),
        )
    radii = compact_radii

    if raw_input_mode:
        viewspace_points = torch.empty(
            (0, 4),
            dtype=means3d.dtype,
            device=means3d.device,
        )
    else:
        viewspace_points = torch.zeros(
            (compact_radii.numel(), 4),
            dtype=means3d.dtype,
            device=means3d.device,
            requires_grad=bool(track_gradients),
        )
    projected_gradients = None
    if track_gradients:
        if viewspace_points.requires_grad:
            viewspace_points.retain_grad()
        projected_gradients = [
            torch.zeros((compact_radii.numel(), 4), dtype=torch.float32, device=means3d.device),
            torch.zeros((compact_radii.numel(), 4), dtype=torch.float32, device=means3d.device),
            torch.zeros((compact_radii.numel(), 1), dtype=torch.float32, device=means3d.device),
            torch.zeros((compact_radii.numel(), 3), dtype=torch.float32, device=means3d.device),
            (
                torch.zeros((compact_radii.numel(), 1), dtype=torch.float32, device=means3d.device)
                if render_depth
                else torch.empty(0, dtype=torch.float32, device=means3d.device)
            ),
        ]

    return StagedProjection(
        candidate_indices=candidate_indices,
        means3d=source_inputs[0],
        dc=source_inputs[1],
        sh=source_inputs[2],
        opacities=source_inputs[3],
        scales=source_inputs[4],
        rotations=source_inputs[5],
        radii=radii,
        compact_radii=compact_radii,
        geometry=geometry,
        viewspace_points=viewspace_points,
        projected_gradients=projected_gradients,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        image_height=image_height,
        image_width=image_width,
        sh_degree=int(gaussians.active_sh_degree),
        scale_modifier=scale_modifier,
        antialiasing=bool(pipe.antialiasing),
        debug=bool(pipe.debug),
        indexed_raw=indexed_raw,
    )


def count_gaussian_tile_intersections_rect(
    projection: StagedProjection,
    rect: tuple[int, int, int, int],
    *,
    render_depth: bool = True,
) -> RectRasterEstimate:
    """Count one rectangle's exact Gaussian--tile references without expanding or sorting them."""
    _require_leaf_raster_api()
    rect_y, rect_x, rect_height, rect_width = (int(value) for value in rect)
    if projection.compact_radii.numel() == 0:
        num_rendered = 0
    else:
        num_rendered = int(
            _C.count_gaussian_tile_intersections_rect(
                projection.compact_radii,
                projection.geometry,
                projection.image_height,
                projection.image_width,
                rect_y,
                rect_x,
                rect_height,
                rect_width,
                projection.debug,
            )
        )
    return estimate_rect_raster_payload(num_rendered, rect, render_depth=render_depth)


def estimate_rect_raster_payload(num_rendered: int, rect, *, render_depth: bool = True) -> RectRasterEstimate:
    """Estimate payload from a count without changing projection scan buffers."""
    _, _, rect_height, rect_width = rect
    sizes = _C.estimate_rect_raster_buffer_sizes(
        num_rendered,
        rect_height,
        rect_width,
        bool(render_depth),
    )
    return RectRasterEstimate(
        num_rendered=num_rendered,
        binning_bytes=int(sizes[1]),
        image_bytes=int(sizes[2]),
        sample_bytes=int(sizes[3]),
        output_bytes=int(sizes[4]),
    )


def rasterize_gaussians_rect(
    projection: StagedProjection,
    background: torch.Tensor,
    rect: tuple[int, int, int, int],
    *,
    precounted_num_rendered: int | None = None,
    render_depth: bool = True,
    background_is_zero: bool = False,
) -> RectRasterState:
    """
    Rasterize one rectangle using previously computed detailed geometry.
    """
    rect_y, rect_x, rect_height, rect_width = (int(value) for value in rect)
    if projection.compact_radii.numel() == 0:
        return RectRasterState(
            num_rendered=0,
            num_buckets=0,
            image=background[:, None, None].expand(-1, rect_height, rect_width).clone(),
            depth=(
                torch.zeros(
                    (1, rect_height, rect_width),
                    dtype=torch.float32,
                    device=projection.means3d.device,
                )
                if render_depth
                else torch.empty(0, dtype=torch.float32, device=projection.means3d.device)
            ),
            binning_buffer=torch.empty(0, dtype=torch.uint8, device=projection.means3d.device),
            image_buffer=torch.empty(0, dtype=torch.uint8, device=projection.means3d.device),
            sample_buffer=torch.empty(0, dtype=torch.uint8, device=projection.means3d.device),
        )
    if precounted_num_rendered is not None:
        _require_leaf_raster_api()
        outputs = _C.rasterize_gaussians_rect_precounted(
            background,
            projection.compact_radii,
            projection.geometry,
            projection.image_height,
            projection.image_width,
            rect_y,
            rect_x,
            rect_height,
            rect_width,
            int(precounted_num_rendered),
            bool(render_depth),
            bool(background_is_zero),
            projection.debug,
        )
    else:
        outputs = _C.rasterize_gaussians_rect(
            background,
            projection.compact_radii,
            projection.geometry,
            projection.image_height,
            projection.image_width,
            rect_y,
            rect_x,
            rect_height,
            rect_width,
            bool(render_depth),
            bool(background_is_zero),
            projection.debug,
        )
    return RectRasterState(
        num_rendered=int(outputs[0]),
        num_buckets=int(outputs[1]),
        image=outputs[2],
        depth=outputs[3],
        binning_buffer=outputs[4],
        image_buffer=outputs[5],
        sample_buffer=outputs[6],
    )


def accumulate_gaussians_rect_backward(
    projection: StagedProjection,
    background: torch.Tensor,
    rect: tuple[int, int, int, int],
    raster_state: RectRasterState,
    grad_image: torch.Tensor,
    grad_depth: torch.Tensor | None = None,
    background_is_zero: bool = False,
) -> None:
    """
    Add one rectangle's rasterization gradients to the detailed projected gradients.
    """
    if projection.projected_gradients is None:
        raise ValueError("The staged projection was created without gradient tracking.")
    if raster_state.num_rendered == 0:
        return
    if grad_depth is None:
        grad_depth = torch.empty(
            0,
            dtype=raster_state.depth.dtype,
            device=raster_state.depth.device,
        )
    rect_y, rect_x, rect_height, rect_width = (int(value) for value in rect)
    _C.rasterize_gaussians_rect_backward(
        background,
        projection.compact_radii,
        projection.geometry,
        projection.image_height,
        projection.image_width,
        rect_y,
        rect_x,
        rect_height,
        rect_width,
        raster_state.image,
        raster_state.depth,
        grad_image,
        grad_depth,
        raster_state.num_rendered,
        raster_state.binning_buffer,
        raster_state.image_buffer,
        raster_state.num_buckets,
        raster_state.sample_buffer,
        *projection.projected_gradients,
        bool(background_is_zero),
        projection.debug,
    )


def compute_staged_projection_backward(
    projection: StagedProjection,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Convert accumulated screen-space gradients into detailed-input gradients.
    """
    if projection.projected_gradients is None:
        raise ValueError("The staged projection was created without gradient tracking.")
    if projection.indexed_raw:
        raise ValueError("Indexed raw projections use accumulate_indexed_raw_projection_backward().")
    detailed_inputs = (
        projection.means3d,
        projection.dc,
        projection.sh,
        projection.opacities,
        projection.scales,
        projection.rotations,
    )
    parameter_gradients = _C.project_gaussians_backward(
        detailed_inputs[0],
        projection.compact_radii,
        detailed_inputs[3],
        detailed_inputs[4],
        detailed_inputs[5],
        projection.scale_modifier,
        projection.viewmatrix,
        projection.projmatrix,
        projection.tanfovx,
        projection.tanfovy,
        projection.image_height,
        projection.image_width,
        detailed_inputs[1],
        detailed_inputs[2],
        projection.sh_degree,
        projection.campos,
        projection.geometry,
        *projection.projected_gradients,
        projection.antialiasing,
        projection.debug,
    )
    if projection.viewspace_points.numel() == projection.projected_gradients[0].numel():
        projection.viewspace_points.grad = projection.projected_gradients[0]
    return detailed_inputs, parameter_gradients


def accumulate_indexed_raw_projection_backward(
    projection: StagedProjection,
    raw_parameters: tuple[torch.Tensor, ...],
    *, compact_gradients=None,
) -> None:
    """Backpropagate indexed projection gradients directly into full raw `.grad` tensors."""
    if not projection.indexed_raw or projection.candidate_indices is None:
        raise ValueError("The staged projection is not an indexed raw projection.")
    if projection.projected_gradients is None:
        raise ValueError("The staged projection was created without gradient tracking.")
    if compact_gradients is not None:
        gradient_buffers = compact_gradients.buffers()
        backward = _C.project_gaussians_indexed_raw_backward_compact
        prefix = (compact_gradients.index_map,)
    else:
        for raw_parameter in raw_parameters:
            if raw_parameter.grad is None:
                raw_parameter.grad = torch.zeros_like(raw_parameter)
            elif raw_parameter.grad.is_sparse:
                raw_parameter.grad = raw_parameter.grad.to_dense()
        gradient_buffers = tuple(p.grad for p in raw_parameters)
        backward = _C.project_gaussians_indexed_raw_backward
        prefix = ()
    backward(
        *prefix,
        projection.candidate_indices,
        raw_parameters[0],
        raw_parameters[3],
        raw_parameters[4],
        raw_parameters[5],
        projection.scale_modifier,
        projection.viewmatrix,
        projection.projmatrix,
        projection.tanfovx,
        projection.tanfovy,
        projection.image_height,
        projection.image_width,
        raw_parameters[1],
        raw_parameters[2],
        projection.sh_degree,
        projection.campos,
        projection.compact_radii,
        projection.geometry,
        *projection.projected_gradients,
        *gradient_buffers,
        projection.antialiasing,
        projection.debug,
    )


def finish_staged_projection_backward(projection: StagedProjection) -> None:
    """
    Backpropagate accumulated projected gradients through Gaussian activations.
    """
    detailed_inputs, parameter_gradients = compute_staged_projection_backward(projection)
    differentiable_inputs: list[torch.Tensor] = []
    differentiable_gradients: list[torch.Tensor] = []
    for input_tensor, gradient in zip(
        (
            detailed_inputs[0],
            detailed_inputs[1],
            detailed_inputs[2],
            detailed_inputs[3],
            detailed_inputs[4],
            detailed_inputs[5],
        ),
        parameter_gradients,
    ):
        if input_tensor.requires_grad:
            differentiable_inputs.append(input_tensor)
            differentiable_gradients.append(gradient)
    if differentiable_inputs:
        torch.autograd.backward(differentiable_inputs, differentiable_gradients)
    del detailed_inputs
