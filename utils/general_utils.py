"""
General utilities shared across training and evaluation.

The functions here are intentionally small: math transforms, image-to-tensor
conversion, learning-rate scheduling, deterministic startup, directory creation,
checkpoint iteration scanning, and RGB/SH color conversion.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from errno import EEXIST
from os import makedirs, path
from typing import Callable

import numpy as np
import random
import torch
from PIL import Image as PILImage

from utils.experiment_utils import BEIJING_TIMEZONE


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
        Convert opacity values from sigmoid space back to raw logits.
    """
    return torch.log(x/(1-x))


def PILtoTorch(pil_image: PILImage.Image, resolution: tuple[int, int]) -> torch.Tensor:
    """
        Resize a PIL image and convert it to a channel-first uint8 tensor.
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1).contiguous()
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1).contiguous()

def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable[[int], float]:
    """
        Return an exponential learning-rate schedule function.

        The returned helper starts near `lr_init`, ends at `lr_final`, and can
        apply a smooth warmup delay when `lr_delay_steps` is positive.
    """

    def helper(step: int) -> float:
        """
            Compute the scheduled learning rate for one step.
        """
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """
        Convert normalized quaternions into 3x3 rotation matrices.
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def safe_state(silent: bool) -> None:
    """
        Set deterministic seeds, select CUDA device 0, and optionally timestamp stdout.
    """
    old_f = sys.stdout
    class F:
        """
            Stdout wrapper that appends timestamps unless quiet mode is enabled.
        """
        def __init__(self, silent: bool) -> None:
            self.silent = silent

        def write(self, x: str) -> None:
            """
                Write stdout text, appending a timestamp to completed lines.
            """
            if not self.silent:
                if x.endswith("\n"):
                    timestamp = datetime.now(BEIJING_TIMEZONE).strftime("%y/%m/%d %H:%M:%S")
                    old_f.write(x.replace("\n", " [{}]\n".format(timestamp)))
                else:
                    old_f.write(x)

        def flush(self) -> None:
            """
                Flush the wrapped stdout stream.
            """
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def mkdir_p(folder_path: str) -> None:
    """
        Create a folder and ignore the error when it already exists.
    """
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder: str) -> int:
    """
        Return the largest saved iteration number from an iteration folder.
    """
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


C0 = 0.28209479177387814


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """
        Convert RGB values in `[0, 1]` to the DC spherical-harmonic coefficient.
    """
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """
        Convert a DC spherical-harmonic coefficient back to RGB.
    """
    return sh * C0 + 0.5
