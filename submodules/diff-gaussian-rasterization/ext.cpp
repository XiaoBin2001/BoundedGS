/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("project_gaussians", &ProjectGaussiansCUDA);
  m.def("project_gaussians_indexed_raw", &ProjectGaussiansIndexedRawCUDA);
  m.def("estimate_rect_raster_buffer_sizes", &EstimateRectRasterBufferSizesCUDA);
  // These calls may wait for CUDA counts. Let the JPEG/Leaf packing workers run
  // while the main thread is inside the C++ raster path. They access separate GT
  // tensors; model and geometry mutation remains on the training thread.
  m.def("count_gaussian_tile_intersections_rect", &CountGaussianTileIntersectionsRectCUDA,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("build_gaussian_group_candidates_raw", &BuildGaussianGroupCandidatesRawCUDA);
  m.def("rasterize_gaussians_rect", &RasterizeGaussiansRectCUDA,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("rasterize_gaussians_rect_precounted", &RasterizeGaussiansRectPrecountedCUDA,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("rasterize_gaussians_rect_backward", &RasterizeGaussiansRectBackwardCUDA,
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("project_gaussians_backward", &ProjectGaussiansBackwardCUDA);
  m.def("project_gaussians_indexed_raw_backward", &ProjectGaussiansIndexedRawBackwardCUDA);
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("project_gaussians_indexed_raw_backward_compact", &ProjectGaussiansIndexedRawBackwardCompactCUDA);
  m.def("adamUpdateTorch", &adamUpdateTorch);
  m.def("adamUpdateCompactTorch", &adamUpdateCompactTorch);
}
