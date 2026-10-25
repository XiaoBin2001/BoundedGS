"""
Graphics math utilities.

These helpers build camera transforms and projection matrices used by Camera,
and provide a tiny point-cloud container shared by dataset loading and Gaussian
initialization.
"""
from __future__ import annotations

import torch
import math
import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

class BasicPointCloud(NamedTuple):
    """
        Minimal point-cloud record with positions, RGB colors, and normals.
    """
    points : NDArray[np.floating]
    colors : NDArray[np.floating]
    normals : NDArray[np.floating]


def getWorld2View2(
    R: NDArray[np.floating],
    t: NDArray[np.floating],
    translate: NDArray[np.floating] = np.array([.0, .0, .0]),
    scale: float = 1.0,
) -> NDArray[np.float32]:
    """
        Build a world-to-view matrix and apply optional scene translation/scale.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    """
        Build the perspective projection matrix used by the main renderer.
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov: float, pixels: int) -> float:
    """
        Convert field of view in radians to focal length in pixels.
    """
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal: float, pixels: int) -> float:
    """
        Convert focal length in pixels to field of view in radians.
    """
    return 2 * math.atan(pixels / (2 * focal))
