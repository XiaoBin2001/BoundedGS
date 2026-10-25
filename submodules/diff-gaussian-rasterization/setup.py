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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="diff-gaussian-rasterization-boundedgs",
    packages=['diff_gaussian_rasterization_boundedgs'],
    package_dir={'diff_gaussian_rasterization_boundedgs': 'diff_gaussian_rasterization'},
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_boundedgs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "ext.cpp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
