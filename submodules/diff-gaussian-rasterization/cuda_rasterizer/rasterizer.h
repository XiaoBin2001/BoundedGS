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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
		public:
				static int count_rect(
					const int P,
					const int candidate_count,
					const int* candidate_indices,
					const int full_width, int full_height,
					const int rect_x, int rect_y,
					const int rect_width, int rect_height,
					const int* radii,
					char* geom_buffer,
					bool debug = false);

				static void project(
				std::function<char* (size_t)> geometryBuffer,
				const int P, int D, int M,
				const int width, int height,
				const float* means3D,
				const float* dc,
				const float* shs,
				const float* colors_precomp,
				const float* opacities,
				const float* scales,
				const float scale_modifier,
				const float* rotations,
				const float* cov3D_precomp,
				const float* viewmatrix,
				const float* projmatrix,
				const float* cam_pos,
				const float tan_fovx, float tan_fovy,
				const bool prefiltered,
				bool antialiasing,
				int* radii = nullptr,
				bool debug = false);

				static void project_indexed_raw(
					std::function<char* (size_t)> geometryBuffer,
					const int P, int D, int M,
					const int width, int height,
					const int* candidate_indices,
					const float* means3D,
					const float* dc,
					const float* shs,
					const float* raw_opacities,
					const float* raw_scales,
					const float scale_modifier,
					const float* raw_rotations,
					const float* viewmatrix,
					const float* projmatrix,
					const float* cam_pos,
					const float tan_fovx, float tan_fovy,
					const bool prefiltered,
					bool antialiasing,
					int* radii = nullptr,
					bool debug = false);

				static std::tuple<int,int> forward_rect(
				std::function<char* (size_t)> binningBuffer,
				std::function<char* (size_t)> imageBuffer,
				std::function<char* (size_t)> sampleBuffer,
				const int P,
				const int candidate_count,
				const int* candidate_indices,
				const float* background,
				const int full_width, int full_height,
				const int rect_x, int rect_y,
				const int rect_width, int rect_height,
				const int* radii,
				char* geom_buffer,
				float* out_color,
				float* depth,
				bool background_is_zero,
				bool debug = false);

				static std::tuple<int,int> forward_rect_precounted(
					std::function<char* (size_t)> binningBuffer,
					std::function<char* (size_t)> imageBuffer,
					std::function<char* (size_t)> sampleBuffer,
					const int P,
					const int candidate_count,
					const int* candidate_indices,
					const float* background,
					const int full_width, int full_height,
					const int rect_x, int rect_y,
					const int rect_width, int rect_height,
					const int* radii,
					char* geom_buffer,
					float* out_color,
					float* depth,
					const int precounted_num_rendered,
					bool background_is_zero,
					bool debug = false);

				static std::tuple<int,int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* depth,
			bool antialiasing,
			bool inference_only,
			int* radii = nullptr,
			bool debug = false);

					static void backward_rect(
					const int P, int R, int B,
					const float* background,
					const int full_width, int full_height,
					const int rect_x, int rect_y,
					const int rect_width, int rect_height,
					char* geom_buffer,
					char* binning_buffer,
					char* image_buffer,
					char* sample_buffer,
					const float* pixel_colors,
					const float* pixel_invdepths,
					const float* dL_dpix,
					const float* dL_invdepths,
					float* dL_dmean2D,
					float* dL_dconic,
					float* dL_dopacity,
					float* dL_dcolor,
					float* dL_dinvdepth,
					bool background_is_zero,
					bool debug);

					static void backward_project(
					const int P, int D, int M,
					const int width, int height,
					const float* means3D,
					const float* dc,
					const float* shs,
					const float* opacities,
					const float* scales,
					const float scale_modifier,
					const float* rotations,
					const float* viewmatrix,
					const float* projmatrix,
					const float* campos,
					const float tan_fovx, float tan_fovy,
					const int* radii,
					char* geom_buffer,
					const float* dL_dmean2D,
					const float* dL_dconic,
					float* dL_dopacity,
					const float* dL_dcolor,
					const float* dL_dinvdepth,
					float* dL_dmean3D,
					float* dL_dcov3D,
					float* dL_ddc,
					float* dL_dsh,
					float* dL_dscale,
					float* dL_drot,
					bool antialiasing,
					bool debug);

					static void backward_project_indexed_raw(
						const int P, int D, int M,
						const int width, int height,
						const int* candidate_indices,
        const int* gradient_index_map,
						const float* means3D,
						const float* dc,
						const float* shs,
						const float* raw_opacities,
						const float* raw_scales,
						const float scale_modifier,
						const float* raw_rotations,
						const float* viewmatrix,
						const float* projmatrix,
						const float* campos,
						const float tan_fovx, float tan_fovy,
						const int* radii,
						char* geom_buffer,
						const float* dL_dmean2D,
						const float* dL_dconic,
						float* dL_dopacity,
						const float* dL_dcolor,
						const float* dL_dinvdepth,
						float* dL_dcov3D,
						float* out_means3D,
						float* out_dc,
						float* out_sh,
						float* out_opacity,
						float* out_scales,
						float* out_rotations,
						bool antialiasing,
						bool debug);

					static void backward(
				const int P, int D, int M, int R, int B,
				const float* background,
			const int width, int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
				const int* radii,
				char* geom_buffer,
				char* binning_buffer,
				char* image_buffer,
				char* sample_buffer,
				const float* pixel_colors,
				const float* pixel_invdepths,
			const float* dL_dpix,
			const float* dL_invdepths,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dinvdepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_ddc,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool antialiasing,
			bool debug);	};
};

#endif
