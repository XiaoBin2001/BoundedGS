#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

from typing import NamedTuple
from typing import Any
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple: tuple[Any, ...]) -> tuple[Any, ...]:
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def _build_forward_args(
    means3D: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    raster_settings: GaussianRasterizationSettings,
) -> tuple[Any, ...]:
    return (
        raster_settings.bg,
        means3D,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        dc,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.antialiasing,
        raster_settings.inference_only,
        raster_settings.render_depth,
        raster_settings.debug,
    )

def rasterize_gaussians(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    raster_settings: GaussianRasterizationSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        opacities,
        scales,
        rotations,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        dc: torch.Tensor,
        sh: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Restructure arguments the way that the C++ lib expects them
        args = _build_forward_args(
            means3D,
            dc,
            sh,
            opacities,
            scales,
            rotations,
            raster_settings,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_buckets, color, invdepths, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, invdepths, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(color, invdepths, means3D, scales, rotations, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(
        ctx: Any,
        grad_out_color: torch.Tensor,
        _: torch.Tensor,
        grad_out_depth: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        color, invdepths, means3D, scales, rotations, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                color,
                invdepths,
                grad_out_color,
                dc,
                sh, 
                grad_out_depth, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
		        raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, _, grad_opacities, grad_means3D, _, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, _, grad_opacities, grad_means3D, _, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_dc,
            grad_sh,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    inference_only : bool
    render_depth : bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings) -> None:
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        dc: torch.Tensor | None = None,
        shs: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        rotations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raster_settings = self.raster_settings

        if dc is None or shs is None:
            raise Exception('Current rasterizer interface requires both dc and shs tensors.')
        if scales is None or rotations is None:
            raise Exception('Current rasterizer interface requires both scales and rotations tensors.')

        if dc.ndim != 3 or dc.shape[1] != 1 or dc.shape[2] != 3:
            raise Exception('dc tensor must have shape [N, 1, 3].')
        if shs.ndim != 3 or shs.shape[2] != 3:
            raise Exception('shs tensor must have shape [N, K, 3].')
        if dc.shape[0] != shs.shape[0]:
            raise Exception('dc and shs must have the same number of Gaussians.')
        if raster_settings.inference_only:
            args = _build_forward_args(
                means3D,
                dc,
                shs,
                opacities,
                scales,
                rotations,
                raster_settings,
            )
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args)
                try:
                    _, _, color, invdepths, radii, _, _, _, _ = _C.rasterize_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                _, _, color, invdepths, radii, _, _, _, _ = _C.rasterize_gaussians(*args)
            return color, radii, invdepths

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            opacities,
            scales, 
            rotations,
            raster_settings
        )

def _ensure_last_seen_state(
    state: dict[str, torch.Tensor],
    N: int,
    param: torch.Tensor,
    bootstrap_step: int,
) -> None:
    if "last_seen_step" not in state:
        state["last_seen_step"] = torch.full(
            (N,),
            int(max(bootstrap_step, 0)),
            dtype=torch.int32,
            device=param.device,
        )
    elif state["last_seen_step"].device != param.device or state["last_seen_step"].dtype != torch.int32:
        state["last_seen_step"] = state["last_seen_step"].to(device=param.device, dtype=torch.int32)


def _ensure_gaussian_adam_state(
    state: dict[str, torch.Tensor],
    N: int,
    param: torch.Tensor,
    max_steps: int,
    bootstrap_step: int,
) -> None:
    if len(state) == 0:
        state["step"] = torch.tensor(int(max(bootstrap_step, 0)), dtype=torch.int64, device=param.device)
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
    elif "step" not in state:
        state["step"] = torch.tensor(int(max(bootstrap_step, 0)), dtype=torch.int64, device=param.device)
    elif not torch.is_tensor(state["step"]):
        state["step"] = torch.tensor(int(state["step"]), dtype=torch.int64, device=param.device)
    else:
        state["step"] = state["step"].to(device=param.device, dtype=torch.int64)

    init_last_seen_step = int(state["step"].item()) if int(state["step"].item()) > 0 else bootstrap_step
    _ensure_last_seen_state(state, N, param, init_last_seen_step)
    if "lr_history" not in state:
        state["lr_history"] = torch.zeros((max_steps,), dtype=torch.float32, device=param.device)
    elif state["lr_history"].device != param.device:
        state["lr_history"] = state["lr_history"].to(device=param.device, dtype=torch.float32)
    elif state["lr_history"].dtype != torch.float32:
        state["lr_history"] = state["lr_history"].to(dtype=torch.float32)


def _ensure_lr_history_capacity(state: dict[str, torch.Tensor], required_size: int, device: torch.device) -> torch.Tensor:
    lr_history = state["lr_history"]
    if lr_history.numel() >= required_size:
        return lr_history
    new_size = max(required_size, lr_history.numel() * 2, 1)
    new_history = torch.zeros((new_size,), dtype=torch.float32, device=device)
    if lr_history.numel() > 0:
        new_history[:lr_history.numel()] = lr_history
    state["lr_history"] = new_history
    return new_history


def _resolve_gaussian_adam_feature_width(param: torch.Tensor, N: int) -> int:
    """
        Resolve how many contiguous scalar values each Gaussian owns in one GaussianAdam parameter group.
    """
    if int(N) < 0:
        raise ValueError("GaussianAdam received negative N: {}".format(N))
    if int(N) == 0 or param.numel() == 0:
        return 0
    if param.numel() % int(N) != 0:
        raise ValueError(
            "GaussianAdam parameter numel {} is not divisible by N {}.".format(param.numel(), N)
        )
    return int(param.numel() // int(N))


class GaussianAdam(torch.optim.Adam):
    supports_compact_gradients = True

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None and hasattr(param.grad, "_boundedgs_index_map"):
                    del param.grad._boundedgs_index_map
        super().zero_grad(set_to_none=set_to_none)

    def __init__(self, params: Any, lr: float, eps: float, max_steps: int) -> None:
        super().__init__(params=params, lr=lr, eps=eps)
        self.max_steps = max(int(max_steps), 1)

    @torch.no_grad()
    def step(self, N: int, global_step: int) -> None:
        for group in self.param_groups:
            eps = group["eps"]
            lr = float(group["lr"])

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]
            bootstrap_step = int(global_step) - 1 if int(global_step) > 1 else 0
            _ensure_gaussian_adam_state(state, N, param, self.max_steps, bootstrap_step)

            step_tensor = state["step"]
            recorded_step = int(step_tensor.item())
            current_step = recorded_step + 1
            if recorded_step <= 0 and int(global_step) > 1:
                current_step = int(global_step)
            lr_history = _ensure_lr_history_capacity(state, current_step, param.device)
            lr_history[current_step - 1] = lr
            visible = torch.ones((N,), dtype=torch.bool, device=param.device)
            feature_width = _resolve_gaussian_adam_feature_width(param, N)

            if feature_width > 0:
                gradient = param.grad
                prefix = ()
                update = _C.adamUpdateTorch
                if gradient.is_sparse:
                    if gradient.sparse_dim() != 1 or not gradient.is_coalesced():
                        raise ValueError("GaussianAdam requires coalesced row-sparse gradients")
                    index_map = getattr(gradient, "_boundedgs_index_map", None)
                    if index_map is None:
                        index_map = torch.full((N,), -1, dtype=torch.int32, device=param.device)
                        index_map[gradient.indices()[0]] = torch.arange(gradient._nnz(), dtype=torch.int32, device=param.device)
                    prefix = (index_map,)
                    update = _C.adamUpdateCompactTorch
                    gradient = gradient.values()
                update(
                    *prefix,
                    param,
                    gradient,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["last_seen_step"],
                    lr_history,
                    visible,
                    0.9,
                    0.999,
                    eps,
                    current_step,
                    N,
                    feature_width,
                )
            state["step"].fill_(current_step)
