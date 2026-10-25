from __future__ import annotations

from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward
from typing import Any

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        C1: float,
        C2: float,
        img1: torch.Tensor,
        img2: torch.Tensor,
        padding: str = "same",
        train: bool = True,
    ) -> torch.Tensor:
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(
        ctx: Any,
        opt_grad: torch.Tensor,
    ) -> tuple[None, None, torch.Tensor, None, None, None]:
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None

def fused_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: str = "same",
    train: bool = True,
) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()
