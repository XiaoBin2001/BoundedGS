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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "raster_math.h"
#include <functional>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const raster_math::vec3* scales,
		const float scale_modifier,
		const raster_math::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const raster_math::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Project Gaussians once in full-image coordinates without building a
	// Gaussian/tile list. The resulting GeometryState can be reused by
	// multiple rectangle rasterization calls.
	void project(int P, int D, int M,
		const float* orig_points,
		const raster_math::vec3* scales,
		const float scale_modifier,
		const raster_math::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const raster_math::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Project a compact candidate list while reading raw parameter rows from
	// the full Gaussian tensors. Activations are applied inside the kernel.
	void project_indexed_raw(int P, int D, int M,
		const int* candidate_indices,
		const float* orig_points,
		const raster_math::vec3* raw_scales,
		const float scale_modifier,
		const raster_math::vec4* raw_rotations,
		const float* raw_opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* viewmatrix,
		const float* projmatrix,
		const raster_math::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar, float* sampled_ard,
		int W, int H,
		int pixel_offset_x, int pixel_offset_y,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth,
		bool background_is_zero);

}


#endif
