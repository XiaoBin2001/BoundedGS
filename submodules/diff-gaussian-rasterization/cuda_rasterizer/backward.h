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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "raster_math.h"

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int R, int B,
		int projection_W, int projection_H,
		int pixel_offset_x, int pixel_offset_y,
		const uint32_t* per_bucket_tile_offset,
		const uint32_t* bucket_to_tile,
		const float* sampled_T, const float* sampled_ar, const float* sampled_ard,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const uint32_t* max_contrib,
		const float* pixel_colors,
		const float* pixel_indepths,
		const float* dL_dpixels,
		const float* dL_invdepths,
		float4* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dinvdepths,
		bool use_background_gradient);
	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* dc,
		const float* shs,
		const bool* clamped,
		const float* opacities,
		const raster_math::vec3* scales,
		const raster_math::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const raster_math::vec3* campos,
		const float4* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_dinvdepth,
		float* dL_dopacity,
		raster_math::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_ddc,
		float* dL_dsh,
		raster_math::vec3* dL_dscale,
		raster_math::vec4* dL_drot,
		bool antialiasing);

	void preprocess_indexed_raw(
		int P, int D, int M,
		const int* candidate_indices,
        const int* gradient_index_map,
		const float3* means,
		const int* radii,
		const float* dc,
		const float* shs,
		const bool* clamped,
		const float* raw_opacities,
		const raster_math::vec3* raw_scales,
		const raster_math::vec4* raw_rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const raster_math::vec3* campos,
		const float4* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_dinvdepth,
		float* dL_dopacity,
		float* dL_dcolor,
		float* dL_dcov3D,
		raster_math::vec3* out_means,
		float* out_dc,
		float* out_sh,
		float* out_opacity,
		raster_math::vec3* out_scales,
		raster_math::vec4* out_rotations,
		bool antialiasing);
}

#endif
