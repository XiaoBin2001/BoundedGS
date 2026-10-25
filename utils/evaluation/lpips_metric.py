"""Full-image LPIPS with an optional memory-bounded VGG forward."""
from __future__ import annotations

from typing import Any

import torch


# VGG16's deepest LPIPS feature has a 196-pixel receptive field and stride 16.
LPIPS_TILE_SIZE = 512
LPIPS_TILE_HALO = 128
LPIPS_FEATURE_STRIDES = (1, 2, 4, 8, 16)


def _tile_starts(size: int, tile_size: int) -> range:
    return range(0, int(size), int(tile_size))


@torch.inference_mode()
def tiled_lpips(
    lpips_model: Any,
    image: torch.Tensor,
    target: torch.Tensor,
    tile_size: int = LPIPS_TILE_SIZE,
    halo: int = LPIPS_TILE_HALO,
) -> torch.Tensor:
    """Evaluate the standard full-image LPIPS reduction through aligned tiles."""
    if image.shape != target.shape or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            "LPIPS inputs must have matching NCHW RGB shapes; got {} and {}.".format(
                tuple(image.shape),
                tuple(target.shape),
            )
        )
    tile_size = int(tile_size)
    halo = int(halo)
    if tile_size <= 0 or tile_size % 16 != 0:
        raise ValueError("LPIPS tile_size must be a positive multiple of 16.")
    if halo < 128 or halo % 16 != 0:
        raise ValueError("LPIPS halo must be a multiple of 16 and at least 128 pixels.")

    batch_size, _, image_height, image_width = image.shape
    if image_height < 16 or image_width < 16:
        raise ValueError("LPIPS input dimensions must both be at least 16 pixels.")
    layer_sums = torch.zeros(
        (batch_size, len(LPIPS_FEATURE_STRIDES)),
        dtype=torch.float64,
        device=image.device,
    )
    layer_counts = [0 for _ in LPIPS_FEATURE_STRIDES]

    for core_y0 in _tile_starts(image_height, tile_size):
        core_y1 = min(core_y0 + tile_size, image_height)
        patch_y0 = max(0, core_y0 - halo)
        patch_y1 = min(image_height, core_y1 + halo)
        for core_x0 in _tile_starts(image_width, tile_size):
            core_x1 = min(core_x0 + tile_size, image_width)
            patch_x0 = max(0, core_x0 - halo)
            patch_x1 = min(image_width, core_x1 + halo)
            image_features = lpips_model.net(
                image[:, :, patch_y0:patch_y1, patch_x0:patch_x1]
            )
            target_features = lpips_model.net(
                target[:, :, patch_y0:patch_y1, patch_x0:patch_x1]
            )
            if len(image_features) != len(LPIPS_FEATURE_STRIDES):
                raise ValueError(
                    "Low-memory LPIPS expects {} VGG feature levels; got {}.".format(
                        len(LPIPS_FEATURE_STRIDES),
                        len(image_features),
                    )
                )

            for layer_index, (image_feature, target_feature, linear, stride) in enumerate(
                zip(
                    image_features,
                    target_features,
                    lpips_model.lin,
                    LPIPS_FEATURE_STRIDES,
                )
            ):
                full_feature_height = image_height // stride
                full_feature_width = image_width // stride
                feature_y0 = core_y0 // stride
                feature_y1 = full_feature_height if core_y1 == image_height else core_y1 // stride
                feature_x0 = core_x0 // stride
                feature_x1 = full_feature_width if core_x1 == image_width else core_x1 // stride
                local_y0 = feature_y0 - patch_y0 // stride
                local_y1 = feature_y1 - patch_y0 // stride
                local_x0 = feature_x0 - patch_x0 // stride
                local_x1 = feature_x1 - patch_x0 // stride
                response = linear((image_feature - target_feature).square())
                core_response = response[:, :, local_y0:local_y1, local_x0:local_x1]
                if core_response.shape[-2:] != (
                    feature_y1 - feature_y0,
                    feature_x1 - feature_x0,
                ):
                    raise RuntimeError("LPIPS feature crop does not match the aligned full-image grid.")
                layer_sums[:, layer_index] += core_response.sum(
                    dim=(1, 2, 3),
                    dtype=torch.float64,
                )
                layer_counts[layer_index] += int(core_response.shape[-2] * core_response.shape[-1])

            del image_features, target_features

    expected_counts = [
        (image_height // stride) * (image_width // stride)
        for stride in LPIPS_FEATURE_STRIDES
    ]
    if layer_counts != expected_counts:
        raise RuntimeError(
            "LPIPS tiled reduction covered {} feature cells; expected {}.".format(
                layer_counts,
                expected_counts,
            )
        )
    counts = torch.tensor(layer_counts, dtype=torch.float64, device=image.device)
    score = (layer_sums / counts.unsqueeze(0)).sum(dim=1)
    return score.to(dtype=image.dtype).reshape(batch_size, 1, 1, 1)


@torch.inference_mode()
def compute_lpips(
    lpips_model: Any,
    image: torch.Tensor,
    target: torch.Tensor,
    low_memory: bool,
) -> torch.Tensor:
    """Select the normal or memory-bounded implementation of the same LPIPS metric."""
    if bool(low_memory):
        return tiled_lpips(lpips_model, image, target)
    return lpips_model(image, target)
