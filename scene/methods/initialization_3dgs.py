"""
Gaussian model initialization for the 3DGS scene representation.

The training script calls this before Scene loads the point cloud. The helper
chooses the starting SH degree for the selected method and creates the
GaussianModel instance with the configured optimizer type.
"""
from typing import Any

from scene.gaussian_model import GaussianModel as GaussianModel3DGS


def build_gaussian_model_3dgs(dataset: Any, opt: Any) -> GaussianModel3DGS:
    """
        Create a GaussianModel with the configured optimizer type.
    """
    gaussian_model = GaussianModel3DGS(
        int(dataset.sh_degree),
        opt.optimizer_type,
    )
    gaussian_model.initial_opacity = float(opt.initial_opacity)
    return gaussian_model
