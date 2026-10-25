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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor>
ProjectGaussiansCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor>
ProjectGaussiansIndexedRawCUDA(
	const torch::Tensor& candidate_indices,
	const torch::Tensor& means3D,
	const torch::Tensor& raw_opacity,
	const torch::Tensor& raw_scales,
	const torch::Tensor& raw_rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug);

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>
EstimateRectRasterBufferSizesCUDA(
	const int num_rendered,
	const int rect_height,
	const int rect_width,
	const bool render_depth);

int
CountGaussianTileIntersectionsRectCUDA(
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const int full_height,
	const int full_width,
	const int rect_y,
	const int rect_x,
	const int rect_height,
	const int rect_width,
	const bool debug);


std::tuple<torch::Tensor, torch::Tensor>
BuildGaussianGroupCandidatesRawCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& raw_opacity,
	const torch::Tensor& raw_scales,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& group_y_boundaries,
	const torch::Tensor& group_x_boundaries,
	const bool debug);

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansRectCUDA(
	const torch::Tensor& background,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const int full_height,
	const int full_width,
	const int rect_y,
	const int rect_x,
	const int rect_height,
	const int rect_width,
	const bool render_depth,
	const bool background_is_zero,
	const bool debug);

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansRectPrecountedCUDA(
	const torch::Tensor& background,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const int full_height,
	const int full_width,
	const int rect_y,
	const int rect_x,
	const int rect_height,
	const int rect_width,
	const int num_rendered,
	const bool render_depth,
	const bool background_is_zero,
	const bool debug);


void RasterizeGaussiansRectBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const int full_height,
	const int full_width,
	const int rect_y,
	const int rect_x,
	const int rect_height,
	const int rect_width,
	const torch::Tensor& out_color,
	const torch::Tensor& out_invdepth,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const torch::Tensor& dL_dmeans2D,
	const torch::Tensor& dL_dconic,
	const torch::Tensor& dL_dopacity,
	const torch::Tensor& dL_dcolors,
	const torch::Tensor& dL_dinvdepths,
	const bool background_is_zero,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ProjectGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& dL_dmeans2D,
	const torch::Tensor& dL_dconic,
	const torch::Tensor& dL_dopacity,
	const torch::Tensor& dL_dcolors,
	const torch::Tensor& dL_dinvdepths,
	const bool antialiasing,
	const bool debug);

void ProjectGaussiansIndexedRawBackwardCUDA(
	const torch::Tensor& candidate_indices,
	const torch::Tensor& means3D,
	const torch::Tensor& raw_opacity,
	const torch::Tensor& raw_scales,
	const torch::Tensor& raw_rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& dL_dmeans2D,
	const torch::Tensor& dL_dconic,
	torch::Tensor dL_dopacity,
	const torch::Tensor& dL_dcolors,
	const torch::Tensor& dL_dinvdepths,
	torch::Tensor out_means3D,
	torch::Tensor out_dc,
	torch::Tensor out_sh,
	torch::Tensor out_opacity,
	torch::Tensor out_scales,
	torch::Tensor out_rotations,
	const bool antialiasing,
	const bool debug);


std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool inference_only,
	const bool render_depth,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	    const torch::Tensor& projmatrix,
		const float tan_fovx, 
		const float tan_fovy,
		const torch::Tensor& out_color,
		const torch::Tensor& out_invdepth,
	    const torch::Tensor& dL_dout_color,
		const torch::Tensor& dc,
		const torch::Tensor& sh,
	const torch::Tensor& dL_dout_invdepth,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const bool antialiasing,
	const bool debug);

void adamUpdateTorch(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &last_seen_step,
	torch::Tensor &lr_history,
	torch::Tensor &visible,
	const float b1,
	const float b2,
	const float eps,
	const int global_step,
	const uint32_t N,
	const uint32_t M
);

void ProjectGaussiansIndexedRawBackwardCompactCUDA(
	const torch::Tensor& gradient_index_map,
	const torch::Tensor& candidate_indices,
	const torch::Tensor& means3D,
	const torch::Tensor& raw_opacity,
	const torch::Tensor& raw_scales,
	const torch::Tensor& raw_rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& dL_dmeans2D,
	const torch::Tensor& dL_dconic,
	torch::Tensor dL_dopacity,
	const torch::Tensor& dL_dcolors,
	const torch::Tensor& dL_dinvdepths,
	torch::Tensor out_means3D,
	torch::Tensor out_dc,
	torch::Tensor out_sh,
	torch::Tensor out_opacity,
	torch::Tensor out_scales,
	torch::Tensor out_rotations,
	const bool antialiasing,
	const bool debug);

void adamUpdateCompactTorch(
	const torch::Tensor& gradient_index_map,
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &last_seen_step,
	torch::Tensor &lr_history,
	torch::Tensor &visible,
	const float b1,
	const float b2,
	const float eps,
	const int global_step,
	const uint32_t N,
	const uint32_t M
);
