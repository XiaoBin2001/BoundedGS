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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/adam.h"
#include "cuda_rasterizer/auxiliary.h"
#include <fstream>
#include <string>
#include <functional>
#include "cuda_rasterizer/raster_math.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

__forceinline__ __device__ bool coarseGaussianScreenBounds(
	const int idx,
	const float* means3D,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const bool inputs_are_raw,
	float4& bounds,
	bool& covers_all_groups)
{
	covers_all_groups = false;
	const float3 mean = {
		means3D[3 * idx],
		means3D[3 * idx + 1],
		means3D[3 * idx + 2]
	};
	const float3 p_view = transformPoint4x3(mean, viewmatrix);
	const float opacity_input = opacities[idx];
	const float3 scale_input = {
		scales[3 * idx],
		scales[3 * idx + 1],
		scales[3 * idx + 2]
	};
	const float opacity = inputs_are_raw
		? 1.0f / (1.0f + expf(-opacity_input))
		: opacity_input;
	const float3 scale = inputs_are_raw
		? make_float3(expf(scale_input.x), expf(scale_input.y), expf(scale_input.z))
		: scale_input;

	// Preserve unusual non-finite inputs for the detailed projection rather
	// than risking a false negative in the coarse stage.
	if (!isfinite(mean.x) || !isfinite(mean.y) || !isfinite(mean.z) ||
		!isfinite(p_view.x) || !isfinite(p_view.y) || !isfinite(p_view.z) ||
		!isfinite(opacity) || !isfinite(scale.x) || !isfinite(scale.y) || !isfinite(scale.z))
	{
		covers_all_groups = true;
		return true;
	}

	if (p_view.z <= 0.2f || opacity <= (1.0f / 255.0f))
		return false;

	const float focal_x = image_width / (2.0f * tan_fovx);
	const float focal_y = image_height / (2.0f * tan_fovy);
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = min(limx, max(-limx, p_view.x / p_view.z));
	const float tytz = min(limy, max(-limy, p_view.y / p_view.z));

	raster_math::mat3 J = raster_math::mat3(
		focal_x / p_view.z, 0.0f, -(focal_x * txtz) / p_view.z,
		0.0f, focal_y / p_view.z, -(focal_y * tytz) / p_view.z,
		0.0f, 0.0f, 0.0f);
	raster_math::mat3 W = raster_math::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	raster_math::mat3 T = W * J;

	// max(scale)^2 * I is a positive-semidefinite upper bound of the
	// oriented 3D covariance. Projecting it gives a conservative screen-space
	// rectangle without reading rotations or constructing detailed geometry.
	const float max_scale = max(scale.x, max(scale.y, scale.z)) * fabsf(scale_modifier);
	const float max_variance = max_scale * max_scale;
	const raster_math::mat3 coarse_cov = max_variance * raster_math::transpose(T) * T;
	constexpr float h_var = 0.3f;
	const float variance_x = max(0.0f, coarse_cov[0][0] + h_var);
	const float variance_y = max(0.0f, coarse_cov[1][1] + h_var);
	const float alpha_extent = 2.0f * logf(opacity * 255.0f);
	const float extent_x = sqrtf(max(0.0f, alpha_extent * variance_x)) + 2.0f;
	const float extent_y = sqrtf(max(0.0f, alpha_extent * variance_y)) + 2.0f;

	const float4 p_hom = transformPoint4x4(mean, projmatrix);
	const float p_w = 1.0f / (p_hom.w + 0.0000001f);
	const float center_x = ndc2Pix(p_hom.x * p_w, image_width);
	const float center_y = ndc2Pix(p_hom.y * p_w, image_height);
	if (!isfinite(center_x) || !isfinite(center_y) || !isfinite(extent_x) || !isfinite(extent_y))
	{
		covers_all_groups = true;
		return true;
	}
	bounds = make_float4(
		center_x - extent_x,
		center_y - extent_y,
		center_x + extent_x,
		center_y + extent_y);
	return true;
}

__device__ bool rawGaussianGroupRange(
	const int idx,
	const float* means3D,
	const float* raw_opacities,
	const float* raw_scales,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const int* group_y_boundaries,
	const int group_rows,
	const int* group_x_boundaries,
	const int group_columns,
	int2& row_range,
	int2& column_range)
{
	float4 bounds;
	bool covers_all_groups = false;
	if (!coarseGaussianScreenBounds(
		idx,
		means3D,
		raw_opacities,
		raw_scales,
		scale_modifier,
		viewmatrix,
		projmatrix,
		tan_fovx,
		tan_fovy,
		image_height,
		image_width,
		true,
		bounds,
		covers_all_groups))
		return false;
	if (covers_all_groups)
	{
		row_range = { 0, group_rows };
		column_range = { 0, group_columns };
		return true;
	}

	int first_column = 0;
	while (first_column < group_columns)
	{
		const int width = group_x_boundaries[first_column + 1] - group_x_boundaries[first_column];
		const int padded_end = group_x_boundaries[first_column]
			+ ((width + BLOCK_X - 1) / BLOCK_X) * BLOCK_X;
		if (bounds.x < padded_end)
			break;
		++first_column;
	}
	int last_column = first_column;
	while (last_column < group_columns && group_x_boundaries[last_column] <= bounds.z)
		++last_column;
	if (first_column >= last_column)
		return false;

	int first_row = 0;
	while (first_row < group_rows)
	{
		const int height = group_y_boundaries[first_row + 1] - group_y_boundaries[first_row];
		const int padded_end = group_y_boundaries[first_row]
			+ ((height + BLOCK_Y - 1) / BLOCK_Y) * BLOCK_Y;
		if (bounds.y < padded_end)
			break;
		++first_row;
	}
	int last_row = first_row;
	while (last_row < group_rows && group_y_boundaries[last_row] <= bounds.w)
		++last_row;
	if (first_row >= last_row)
		return false;

	row_range = { first_row, last_row };
	column_range = { first_column, last_column };
	return true;
}

__global__ void countRawGaussianGroupReferences(
	const int P,
	const float* means3D,
	const float* raw_opacities,
	const float* raw_scales,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const int* group_y_boundaries,
	const int group_rows,
	const int* group_x_boundaries,
	const int group_columns,
	uint32_t* references_per_gaussian)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	int2 row_range;
	int2 column_range;
	if (!rawGaussianGroupRange(
		idx,
		means3D,
		raw_opacities,
		raw_scales,
		scale_modifier,
		viewmatrix,
		projmatrix,
		tan_fovx,
		tan_fovy,
		image_height,
		image_width,
		group_y_boundaries,
		group_rows,
		group_x_boundaries,
		group_columns,
		row_range,
		column_range))
	{
		references_per_gaussian[idx] = 0;
		return;
	}

	references_per_gaussian[idx] =
		(row_range.y - row_range.x) * (column_range.y - column_range.x);
}

__global__ void fillRawGaussianGroupReferences(
	const int P,
	const float* means3D,
	const float* raw_opacities,
	const float* raw_scales,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const int* group_y_boundaries,
	const int group_rows,
	const int* group_x_boundaries,
	const int group_columns,
	const uint32_t* gaussian_offsets,
	uint32_t* group_keys,
	uint32_t* gaussian_indices)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	int2 row_range;
	int2 column_range;
	if (!rawGaussianGroupRange(
		idx,
		means3D,
		raw_opacities,
		raw_scales,
		scale_modifier,
		viewmatrix,
		projmatrix,
		tan_fovx,
		tan_fovy,
		image_height,
		image_width,
		group_y_boundaries,
		group_rows,
		group_x_boundaries,
		group_columns,
		row_range,
		column_range))
		return;

	uint32_t output_index = idx == 0 ? 0 : gaussian_offsets[idx - 1];
	for (int row = row_range.x; row < row_range.y; ++row)
	{
		for (int column = column_range.x; column < column_range.y; ++column)
		{
			group_keys[output_index] = row * group_columns + column;
			gaussian_indices[output_index] = static_cast<uint32_t>(idx);
			++output_index;
		}
	}
}

__device__ bool gaussianGroupRange(
	const float2 point,
	const int radius,
	const dim3 full_grid,
	const int* group_y_boundaries,
	const int group_rows,
	const int* group_x_boundaries,
	const int group_columns,
	int2& row_range,
	int2& column_range)
{
	// Projection radius is ceil(3 sigma_max). The rasterizer's 1/255 alpha
	// cutoff is at most sqrt(2 log(255)) ~= 3.33 sigma, so 4/3 radius is
	// a conservative square bound. Exact ellipse/tile tests still run later.
	const float extent = (4.0f / 3.0f) * static_cast<float>(radius);
	const int2 rect_min = {
		max(0, min(static_cast<int>(full_grid.x), static_cast<int>(floorf((point.x - extent) / BLOCK_X)))),
		max(0, min(static_cast<int>(full_grid.y), static_cast<int>(floorf((point.y - extent) / BLOCK_Y))))
	};
	const int2 rect_max = {
		max(0, min(static_cast<int>(full_grid.x), static_cast<int>(floorf((point.x + extent) / BLOCK_X)) + 1)),
		max(0, min(static_cast<int>(full_grid.y), static_cast<int>(floorf((point.y + extent) / BLOCK_Y)) + 1))
	};
	if (rect_max.x <= rect_min.x || rect_max.y <= rect_min.y)
		return false;

	int first_column = 0;
	while (first_column + 1 < group_columns && group_x_boundaries[first_column + 1] <= rect_min.x)
		++first_column;
	int last_column = first_column + 1;
	while (last_column < group_columns && group_x_boundaries[last_column] < rect_max.x)
		++last_column;

	int first_row = 0;
	while (first_row + 1 < group_rows && group_y_boundaries[first_row + 1] <= rect_min.y)
		++first_row;
	int last_row = first_row + 1;
	while (last_row < group_rows && group_y_boundaries[last_row] < rect_max.y)
		++last_row;

	row_range = { first_row, last_row };
	column_range = { first_column, last_column };
	return true;
}

__global__ void findGaussianGroupOffsets(
	const uint32_t* sorted_group_keys,
	const int candidate_count,
	const int group_count,
	int* group_offsets)
{
	const int group_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (group_index > group_count)
		return;

	int lower = 0;
	int upper = candidate_count;
	while (lower < upper)
	{
		const int middle = lower + (upper - lower) / 2;
		if (sorted_group_keys[middle] < static_cast<uint32_t>(group_index))
			lower = middle + 1;
		else
			upper = middle;
	}
	group_offsets[group_index] = lower;
}


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
	const bool debug)
{
	TORCH_CHECK(means3D.is_cuda() && raw_opacity.is_cuda() && raw_scales.is_cuda(),
		"raw candidate-table Gaussian inputs must be CUDA tensors");
	TORCH_CHECK(viewmatrix.is_cuda() && projmatrix.is_cuda(),
		"raw candidate-table camera matrices must be CUDA tensors");
	TORCH_CHECK(group_y_boundaries.is_cuda() && group_x_boundaries.is_cuda(),
		"raw candidate-table rectangle boundaries must be CUDA tensors");
	TORCH_CHECK(means3D.scalar_type() == torch::kFloat32 &&
		raw_opacity.scalar_type() == torch::kFloat32 &&
		raw_scales.scalar_type() == torch::kFloat32 &&
		viewmatrix.scalar_type() == torch::kFloat32 &&
		projmatrix.scalar_type() == torch::kFloat32,
		"raw candidate-table Gaussian and camera inputs must use float32");
	TORCH_CHECK(group_y_boundaries.scalar_type() == torch::kInt32 &&
		group_x_boundaries.scalar_type() == torch::kInt32,
		"raw candidate-table rectangle boundaries must use int32");
	TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
		"means3D must have dimensions (num_points, 3)");
	TORCH_CHECK(raw_scales.ndimension() == 2 && raw_scales.size(1) == 3,
		"raw_scales must have dimensions (num_points, 3)");
	TORCH_CHECK(raw_opacity.numel() == means3D.size(0) && raw_scales.size(0) == means3D.size(0),
		"raw_opacity and raw_scales must match the number of points");
	TORCH_CHECK(means3D.is_contiguous() && raw_opacity.is_contiguous() &&
		raw_scales.is_contiguous() && viewmatrix.is_contiguous() && projmatrix.is_contiguous() &&
		group_y_boundaries.is_contiguous() && group_x_boundaries.is_contiguous(),
		"raw candidate-table inputs must be contiguous");
	TORCH_CHECK(group_y_boundaries.numel() >= 2 && group_x_boundaries.numel() >= 2,
		"rectangle boundary tensors must contain at least two entries");

	const int P = means3D.size(0);
	const int group_rows = group_y_boundaries.numel() - 1;
	const int group_columns = group_x_boundaries.numel() - 1;
	const int group_count = group_rows * group_columns;
	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto byte_opts = means3D.options().dtype(torch::kByte);
	torch::Tensor group_offsets = torch::zeros({group_count + 1}, int_opts);
	if (P == 0)
		return std::make_tuple(torch::empty({0}, int_opts), group_offsets);

	torch::Tensor references_per_gaussian = torch::empty({P}, int_opts);
	torch::Tensor gaussian_offsets = torch::empty({P}, int_opts);
	countRawGaussianGroupReferences<<<(P + 255) / 256, 256>>>(
		P,
		means3D.data_ptr<float>(),
		raw_opacity.data_ptr<float>(),
		raw_scales.data_ptr<float>(),
		scale_modifier,
		viewmatrix.data_ptr<float>(),
		projmatrix.data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		image_height,
		image_width,
		group_y_boundaries.data_ptr<int>(),
		group_rows,
		group_x_boundaries.data_ptr<int>(),
		group_columns,
		reinterpret_cast<uint32_t*>(references_per_gaussian.data_ptr<int>()));
	CHECK_CUDA(, debug)

	size_t scan_size = 0;
	cub::DeviceScan::InclusiveSum(
		nullptr,
		scan_size,
		reinterpret_cast<uint32_t*>(references_per_gaussian.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_offsets.data_ptr<int>()),
		P);
	torch::Tensor scanning_space = torch::empty({static_cast<long long>(scan_size)}, byte_opts);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		scanning_space.data_ptr(),
		scan_size,
		reinterpret_cast<uint32_t*>(references_per_gaussian.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_offsets.data_ptr<int>()),
		P), debug)

	uint32_t candidate_count = 0;
	CHECK_CUDA(cudaMemcpy(
		&candidate_count,
		gaussian_offsets.data_ptr<int>() + P - 1,
		sizeof(uint32_t),
		cudaMemcpyDeviceToHost), debug)
	if (candidate_count == 0)
		return std::make_tuple(torch::empty({0}, int_opts), group_offsets);
	TORCH_CHECK(candidate_count <= 2147483647u,
		"raw Gaussian rectangle candidate table exceeds the signed 32-bit tensor length limit");

	torch::Tensor group_keys_unsorted = torch::empty({candidate_count}, int_opts);
	torch::Tensor group_keys = torch::empty({candidate_count}, int_opts);
	torch::Tensor gaussian_indices_unsorted = torch::empty({candidate_count}, int_opts);
	// The payload is a uint32 global Gaussian ID. PyTorch 2.1 exposes it as an
	// int32 tensor because CUDA index_select/index_add consume signed int32 IDs.
	torch::Tensor gaussian_indices = torch::empty({candidate_count}, int_opts);
	fillRawGaussianGroupReferences<<<(P + 255) / 256, 256>>>(
		P,
		means3D.data_ptr<float>(),
		raw_opacity.data_ptr<float>(),
		raw_scales.data_ptr<float>(),
		scale_modifier,
		viewmatrix.data_ptr<float>(),
		projmatrix.data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		image_height,
		image_width,
		group_y_boundaries.data_ptr<int>(),
		group_rows,
		group_x_boundaries.data_ptr<int>(),
		group_columns,
		reinterpret_cast<uint32_t*>(gaussian_offsets.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(group_keys_unsorted.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_indices_unsorted.data_ptr<int>()));
	CHECK_CUDA(, debug)

	int group_key_bits = 0;
	while ((1 << group_key_bits) < group_count)
		++group_key_bits;
	size_t sorting_size = 0;
	cub::DeviceRadixSort::SortPairs(
		nullptr,
		sorting_size,
		reinterpret_cast<uint32_t*>(group_keys_unsorted.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(group_keys.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_indices_unsorted.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_indices.data_ptr<int>()),
		static_cast<int>(candidate_count),
		0,
		group_key_bits);
	torch::Tensor sorting_space = torch::empty({static_cast<long long>(sorting_size)}, byte_opts);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		sorting_space.data_ptr(),
		sorting_size,
		reinterpret_cast<uint32_t*>(group_keys_unsorted.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(group_keys.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_indices_unsorted.data_ptr<int>()),
		reinterpret_cast<uint32_t*>(gaussian_indices.data_ptr<int>()),
		static_cast<int>(candidate_count),
		0,
		group_key_bits), debug)

	findGaussianGroupOffsets<<<(group_count + 256) / 256, 256>>>(
		reinterpret_cast<uint32_t*>(group_keys.data_ptr<int>()),
		static_cast<int>(candidate_count),
		group_count,
		group_offsets.data_ptr<int>());
	CHECK_CUDA(, debug)
	return std::make_tuple(gaussian_indices, group_offsets);
}

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
	const bool debug)
{
	TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
		"means3D must have dimensions (num_points, 3)");
	const int P = means3D.size(0);
	const int M = sh.size(0) == 0 ? 0 : sh.size(1);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	auto byte_opts = means3D.options().dtype(torch::kByte);
	torch::Tensor radii = torch::zeros({P}, int_opts);
	torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
	torch::Tensor colors = torch::empty({0}, float_opts);
	torch::Tensor cov3D_precomp = torch::empty({0}, float_opts);

	if (P > 0)
	{
		std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
		CudaRasterizer::Rasterizer::project(
			geomFunc,
			P,
			degree,
			M,
			image_width,
			image_height,
			means3D.contiguous().data_ptr<float>(),
			dc.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			antialiasing,
			radii.data_ptr<int>(),
			debug);
	}
	return std::make_tuple(radii, geomBuffer);
}

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
	const bool debug)
{
	TORCH_CHECK(candidate_indices.is_cuda() && candidate_indices.is_contiguous()
		&& candidate_indices.scalar_type() == torch::kInt32,
		"candidate_indices must be contiguous CUDA int32");
	TORCH_CHECK(means3D.is_cuda() && means3D.is_contiguous()
		&& means3D.scalar_type() == torch::kFloat32
		&& means3D.ndimension() == 2 && means3D.size(1) == 3,
		"means3D must be contiguous CUDA float32 with shape (N, 3)");
	const int64_t full_count = means3D.size(0);
	auto check_full_float = [full_count](const torch::Tensor& tensor, const char* name) {
		TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
			&& tensor.scalar_type() == torch::kFloat32,
			name, " must be contiguous CUDA float32");
		TORCH_CHECK(tensor.size(0) == full_count,
			name, " must have the same leading dimension as means3D");
	};
	check_full_float(raw_opacity, "raw_opacity");
	check_full_float(raw_scales, "raw_scales");
	check_full_float(raw_rotations, "raw_rotations");
	check_full_float(dc, "dc");
	check_full_float(sh, "sh");
	TORCH_CHECK(raw_opacity.numel() == full_count,
		"raw_opacity must contain one value per Gaussian");
	TORCH_CHECK(raw_scales.ndimension() == 2 && raw_scales.size(1) == 3,
		"raw_scales must have shape (N, 3)");
	TORCH_CHECK(raw_rotations.ndimension() == 2 && raw_rotations.size(1) == 4,
		"raw_rotations must have shape (N, 4)");

	const int P = candidate_indices.numel();
	const int M = sh.size(0) == 0 ? 0 : sh.size(1);
	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto byte_opts = means3D.options().dtype(torch::kByte);
	torch::Tensor radii = torch::zeros({P}, int_opts);
	torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
	if (P > 0)
	{
		std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
		CudaRasterizer::Rasterizer::project_indexed_raw(
			geomFunc,
			P,
			degree,
			M,
			image_width,
			image_height,
			candidate_indices.data_ptr<int>(),
			means3D.data_ptr<float>(),
			dc.data_ptr<float>(),
			sh.data_ptr<float>(),
			raw_opacity.data_ptr<float>(),
			raw_scales.data_ptr<float>(),
			scale_modifier,
			raw_rotations.data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			antialiasing,
			radii.data_ptr<int>(),
			debug);
	}
	return std::make_tuple(radii, geomBuffer);
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>
EstimateRectRasterBufferSizesCUDA(
	const int num_rendered,
	const int rect_height,
	const int rect_width,
	const bool render_depth)
{
	TORCH_CHECK(num_rendered >= 0, "num_rendered must be non-negative");
	TORCH_CHECK(rect_height > 0 && rect_width > 0, "rectangle dimensions must be positive");
	const int64_t num_pixels = static_cast<int64_t>(rect_height) * rect_width;
	const int64_t num_tiles =
		static_cast<int64_t>((rect_width + BLOCK_X - 1) / BLOCK_X)
		* ((rect_height + BLOCK_Y - 1) / BLOCK_Y);
	const int64_t possible_nonempty_tiles = std::min<int64_t>(num_tiles, num_rendered);
	const int64_t upper_bucket_count =
		(static_cast<int64_t>(num_rendered) + 31 * possible_nonempty_tiles + 31) / 32;
	const int64_t binning_bytes = static_cast<int64_t>(
		CudaRasterizer::required<CudaRasterizer::BinningState>(num_rendered));
	const int64_t image_bytes = static_cast<int64_t>(
		CudaRasterizer::requiredImage(num_pixels, num_tiles));
	const int64_t sample_bytes = static_cast<int64_t>(
		CudaRasterizer::requiredSample(upper_bucket_count, render_depth));
	const int64_t output_bytes =
		4 * NUM_CHANNELS_3DGS * num_pixels
		+ (render_depth ? 4 * num_pixels : 0);
	return std::make_tuple(
		upper_bucket_count,
		binning_bytes,
		image_bytes,
		sample_bytes,
		output_bytes);
}

int CountGaussianTileIntersectionsRectCUDA(
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const int full_height,
	const int full_width,
	const int rect_y,
	const int rect_x,
	const int rect_height,
	const int rect_width,
	const bool debug)
{
	TORCH_CHECK(radii.is_cuda() && geomBuffer.is_cuda(),
		"radii and geometry must be CUDA tensors");
	TORCH_CHECK(radii.scalar_type() == torch::kInt32 && radii.is_contiguous(),
		"radii must be contiguous int32");
	TORCH_CHECK(geomBuffer.scalar_type() == torch::kUInt8 && geomBuffer.is_contiguous(),
		"geometry must be contiguous uint8");
	const int P = radii.numel();
	if (P == 0)
		return 0;
	return CudaRasterizer::Rasterizer::count_rect(
		P,
		P,
		nullptr,
		full_width,
		full_height,
		rect_x,
		rect_y,
		rect_width,
		rect_height,
		radii.data_ptr<int>(),
		reinterpret_cast<char*>(geomBuffer.data_ptr()),
		debug);
}

static std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansRectImpl(
	const torch::Tensor& background,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const int full_height,
	const int full_width,
	const int rect_y,
	const int rect_x,
	const int rect_height,
	const int rect_width,
	const torch::Tensor* gaussian_indices,
	const int precounted_num_rendered,
	const bool render_depth,
	const bool background_is_zero,
	const bool debug)
{
	TORCH_CHECK(rect_x >= 0 && rect_y >= 0 && rect_width > 0 && rect_height > 0,
		"rectangle must have a non-negative origin and positive size");
	TORCH_CHECK(rect_x + rect_width <= full_width && rect_y + rect_height <= full_height,
		"rectangle exceeds full image bounds");
	const int P = radii.numel();
	int candidate_count = P;
	const int* candidate_ptr = nullptr;
	if (gaussian_indices != nullptr)
	{
		TORCH_CHECK(gaussian_indices->is_cuda() && gaussian_indices->is_contiguous(),
			"Gaussian rectangle candidates must be a contiguous CUDA tensor");
		TORCH_CHECK(gaussian_indices->scalar_type() == torch::kInt32,
			"Gaussian rectangle candidates must use int32");
		candidate_count = gaussian_indices->numel();
		TORCH_CHECK(candidate_count <= P,
			"Gaussian rectangle candidate count cannot exceed the projected Gaussian count");
		candidate_ptr = gaussian_indices->data_ptr<int>();
	}
	auto float_opts = background.options().dtype(torch::kFloat32);
	auto byte_opts = background.options().dtype(torch::kByte);
	torch::Tensor out_color = background_is_zero && P > 0 && candidate_count > 0
		? torch::empty({NUM_CHANNELS_3DGS, rect_height, rect_width}, float_opts)
		: background
			.reshape({NUM_CHANNELS_3DGS, 1, 1})
			.expand({NUM_CHANNELS_3DGS, rect_height, rect_width})
			.clone();
	torch::Tensor out_invdepth = render_depth
		? torch::zeros({1, rect_height, rect_width}, float_opts)
		: torch::empty({0}, float_opts);
	torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
	torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
	torch::Tensor sampleBuffer = torch::empty({0}, byte_opts);

	int rendered = 0;
	int num_buckets = 0;
	if (P > 0 && candidate_count > 0)
	{
		std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
		std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
		std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);
		std::tuple<int, int> result;
		if (precounted_num_rendered >= 0)
		{
			result = CudaRasterizer::Rasterizer::forward_rect_precounted(
				binningFunc, imgFunc, sampleFunc, P, candidate_count, candidate_ptr,
				background.contiguous().data_ptr<float>(), full_width, full_height,
				rect_x, rect_y, rect_width, rect_height,
				radii.contiguous().data_ptr<int>(),
				reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
				out_color.data_ptr<float>(),
				render_depth ? out_invdepth.data_ptr<float>() : nullptr,
				precounted_num_rendered, background_is_zero, debug);
		}
		else
		{
			result = CudaRasterizer::Rasterizer::forward_rect(
				binningFunc, imgFunc, sampleFunc, P, candidate_count, candidate_ptr,
				background.contiguous().data_ptr<float>(), full_width, full_height,
				rect_x, rect_y, rect_width, rect_height,
				radii.contiguous().data_ptr<int>(),
				reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
				out_color.data_ptr<float>(),
				render_depth ? out_invdepth.data_ptr<float>() : nullptr,
				background_is_zero, debug);
		}
		rendered = std::get<0>(result);
		num_buckets = std::get<1>(result);
	}
	return std::make_tuple(
		rendered,
		num_buckets,
		out_color,
		out_invdepth,
		binningBuffer,
		imgBuffer,
		sampleBuffer);
}

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
	const bool debug)
{
	return RasterizeGaussiansRectImpl(
		background,
		radii,
		geomBuffer,
		full_height,
		full_width,
		rect_y,
		rect_x,
		rect_height,
		rect_width,
		nullptr,
		-1,
		render_depth,
		background_is_zero,
		debug);
}

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
	const bool debug)
{
	TORCH_CHECK(num_rendered >= 0, "precounted num_rendered must be non-negative");
	return RasterizeGaussiansRectImpl(
		background,
		radii,
		geomBuffer,
		full_height,
		full_width,
		rect_y,
		rect_x,
		rect_height,
		rect_width,
		nullptr,
		num_rendered,
		render_depth,
		background_is_zero,
		debug);
}


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
	const bool debug)
{
	const int P = radii.numel();
	if (P == 0)
		return;
	const bool use_depth = dL_dout_invdepth.numel() > 0;
	TORCH_CHECK(!use_depth || out_invdepth.numel() > 0,
		"A depth output is required when a depth gradient is provided");
	TORCH_CHECK(!use_depth || dL_dinvdepths.numel() == P,
		"dL_dinvdepths must match radii when depth gradients are enabled");
	CudaRasterizer::Rasterizer::backward_rect(
		P,
		R,
		B,
		background.contiguous().data_ptr<float>(),
		full_width,
		full_height,
		rect_x,
		rect_y,
		rect_width,
		rect_height,
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
		out_color.contiguous().data_ptr<float>(),
		use_depth ? out_invdepth.contiguous().data_ptr<float>() : nullptr,
		dL_dout_color.contiguous().data_ptr<float>(),
		use_depth ? dL_dout_invdepth.contiguous().data_ptr<float>() : nullptr,
		dL_dmeans2D.contiguous().data_ptr<float>(),
		dL_dconic.contiguous().data_ptr<float>(),
		dL_dopacity.contiguous().data_ptr<float>(),
		dL_dcolors.contiguous().data_ptr<float>(),
		use_depth ? dL_dinvdepths.contiguous().data_ptr<float>() : nullptr,
		background_is_zero,
		debug);
}

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
	const bool debug)
{
	const int P = means3D.size(0);
	const int M = sh.size(0) == 0 ? 0 : sh.size(1);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, float_opts);
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, float_opts);
	torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, float_opts);
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, float_opts);
	torch::Tensor dL_dscales = torch::zeros({P, 3}, float_opts);
	torch::Tensor dL_drotations = torch::zeros({P, 4}, float_opts);

	if (P > 0)
	{
		CudaRasterizer::Rasterizer::backward_project(
			P,
			degree,
			M,
			image_width,
			image_height,
			means3D.contiguous().data_ptr<float>(),
			dc.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			opacities.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			dL_dmeans2D.contiguous().data_ptr<float>(),
			dL_dconic.contiguous().data_ptr<float>(),
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dcolors.contiguous().data_ptr<float>(),
			dL_dinvdepths.numel() == 0
				? nullptr
				: dL_dinvdepths.contiguous().data_ptr<float>(),
			dL_dmeans3D.data_ptr<float>(),
			dL_dcov3D.data_ptr<float>(),
			dL_ddc.data_ptr<float>(),
			dL_dsh.data_ptr<float>(),
			dL_dscales.data_ptr<float>(),
			dL_drotations.data_ptr<float>(),
			antialiasing,
			debug);
	}

	return std::make_tuple(
		dL_dmeans3D,
		dL_ddc,
		dL_dsh,
		dL_dopacity,
		dL_dscales,
		dL_drotations);
}

static void ProjectGaussiansIndexedRawBackwardImpl(
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
	const bool debug)
{
	TORCH_CHECK(candidate_indices.is_cuda() && candidate_indices.is_contiguous()
		&& candidate_indices.scalar_type() == torch::kInt32,
		"candidate_indices must be contiguous CUDA int32");
	TORCH_CHECK(means3D.is_cuda() && means3D.is_contiguous()
		&& means3D.scalar_type() == torch::kFloat32
		&& means3D.ndimension() == 2 && means3D.size(1) == 3,
		"means3D must be contiguous CUDA float32 with shape (N, 3)");
	const int64_t full_count = means3D.size(0);
	const int64_t compact_count = candidate_indices.numel();
	auto check_full_float = [full_count](const torch::Tensor& tensor, const char* name) {
		TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
			&& tensor.scalar_type() == torch::kFloat32,
			name, " must be contiguous CUDA float32");
		TORCH_CHECK(tensor.size(0) == full_count,
			name, " must have the same leading dimension as means3D");
	};
	auto check_compact_float = [compact_count](const torch::Tensor& tensor, const char* name) {
		TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
			&& tensor.scalar_type() == torch::kFloat32,
			name, " must be contiguous CUDA float32");
		TORCH_CHECK(tensor.size(0) == compact_count,
			name, " must have the same leading dimension as candidate_indices");
	};
	check_full_float(raw_opacity, "raw_opacity");
	check_full_float(raw_scales, "raw_scales");
	check_full_float(raw_rotations, "raw_rotations");
	check_full_float(dc, "dc");
	check_full_float(sh, "sh");
	const bool compact_output = gradient_index_map.defined();
	if (compact_output) {
		TORCH_CHECK(gradient_index_map.is_cuda() && gradient_index_map.is_contiguous()
			&& gradient_index_map.scalar_type() == torch::kInt32 && gradient_index_map.numel() == full_count,
			"gradient_index_map must contain N contiguous CUDA int32 entries");
	}
	auto check_output = [&](const torch::Tensor& out, const torch::Tensor& source, const char* name) {
		TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat32,
			name, " must be contiguous CUDA float32");
		TORCH_CHECK(out.dim() == source.dim(), name, " rank mismatch");
		for (int d=1; d<source.dim(); ++d) TORCH_CHECK(out.size(d)==source.size(d), name, " feature shape mismatch");
		TORCH_CHECK(out.size(0)==(compact_output ? out_means3D.size(0) : full_count), name, " row mismatch");
	};
	check_output(out_means3D, means3D, "out_means3D");
	check_output(out_dc, dc, "out_dc");
	check_output(out_sh, sh, "out_sh");
	check_output(out_scales, raw_scales, "out_scales");
	check_output(out_rotations, raw_rotations, "out_rotations");
	check_full_float(out_opacity, "out_opacity");
	TORCH_CHECK(out_opacity.sizes()==raw_opacity.sizes(), "out_opacity shape mismatch");

	TORCH_CHECK(radii.is_cuda() && radii.is_contiguous()
		&& radii.scalar_type() == torch::kInt32
		&& radii.numel() == compact_count,
		"radii must be contiguous CUDA int32 matching candidate_indices");
	TORCH_CHECK(geomBuffer.is_cuda() && geomBuffer.is_contiguous()
		&& geomBuffer.scalar_type() == torch::kUInt8,
		"geomBuffer must be contiguous CUDA uint8");
	check_compact_float(dL_dmeans2D, "dL_dmeans2D");
	check_compact_float(dL_dconic, "dL_dconic");
	check_compact_float(dL_dopacity, "dL_dopacity");
	check_compact_float(dL_dcolors, "dL_dcolors");
	if (dL_dinvdepths.numel() > 0)
		check_compact_float(dL_dinvdepths, "dL_dinvdepths");
	else
		TORCH_CHECK(dL_dinvdepths.is_cuda() && dL_dinvdepths.is_contiguous()
			&& dL_dinvdepths.scalar_type() == torch::kFloat32,
			"Empty dL_dinvdepths must be contiguous CUDA float32");
	if (compact_count == 0)
		return;

	const int M = sh.size(0) == 0 ? 0 : sh.size(1);
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor dL_dcov3D = torch::empty({compact_count, 6}, float_opts);
	CudaRasterizer::Rasterizer::backward_project_indexed_raw(
		static_cast<int>(compact_count),
		degree,
		M,
		image_width,
		image_height,
		candidate_indices.data_ptr<int>(),
		compact_output ? gradient_index_map.data_ptr<int>() : nullptr,
		means3D.data_ptr<float>(),
		dc.data_ptr<float>(),
		sh.data_ptr<float>(),
		raw_opacity.data_ptr<float>(),
		raw_scales.data_ptr<float>(),
		scale_modifier,
		raw_rotations.data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		radii.data_ptr<int>(),
		reinterpret_cast<char*>(geomBuffer.data_ptr()),
		dL_dmeans2D.data_ptr<float>(),
		dL_dconic.data_ptr<float>(),
		dL_dopacity.data_ptr<float>(),
		dL_dcolors.data_ptr<float>(),
		dL_dinvdepths.numel() == 0 ? nullptr : dL_dinvdepths.data_ptr<float>(),
		dL_dcov3D.data_ptr<float>(),
		out_means3D.data_ptr<float>(),
		out_dc.data_ptr<float>(),
		out_sh.data_ptr<float>(),
		out_opacity.data_ptr<float>(),
		out_scales.data_ptr<float>(),
		out_rotations.data_ptr<float>(),
		antialiasing,
		debug);
}

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
	const bool debug)
{
	ProjectGaussiansIndexedRawBackwardImpl(torch::Tensor(), candidate_indices, means3D, raw_opacity, raw_scales, raw_rotations, scale_modifier, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, dc, sh, degree, campos, radii, geomBuffer, dL_dmeans2D, dL_dconic, dL_dopacity, dL_dcolors, dL_dinvdepths, out_means3D, out_dc, out_sh, out_opacity, out_scales, out_rotations, antialiasing, debug);
}

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
	const bool debug)
{
	ProjectGaussiansIndexedRawBackwardImpl(gradient_index_map, candidate_indices, means3D, raw_opacity, raw_scales, raw_rotations, scale_modifier, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, dc, sh, degree, campos, radii, geomBuffer, dL_dmeans2D, dL_dconic, dL_dopacity, dL_dcolors, dL_dinvdepths, out_means3D, out_dc, out_sh, out_opacity, out_scales, out_rotations, antialiasing, debug);
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS_3DGS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = render_depth && !inference_only
    ? torch::full({1, H, W}, 0.0, float_opts)
    : torch::empty({0}, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  auto byte_opts = means3D.options().dtype(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
  torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
  torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
  torch::Tensor sampleBuffer = torch::empty({0}, byte_opts);
  torch::Tensor colors = torch::empty({0}, float_opts);
  torch::Tensor cov3D_precomp = torch::empty({0}, float_opts);
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

  int rendered = 0;
  int num_buckets = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  auto tup = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		sampleFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		dc.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepth.numel() == 0 ? nullptr : out_invdepth.contiguous().data<float>(),
		antialiasing,
		inference_only,
		radii.contiguous().data<int>(),
		debug);
		
		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
  }
	  return std::make_tuple(rendered, num_buckets, out_color, out_invdepth, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer);
}

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
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS_3DGS}, means3D.options());
	const bool use_depth = out_invdepth.numel() > 0 && dL_dout_invdepth.numel() > 0;
	torch::Tensor dL_dinvdepths = use_depth
		? torch::zeros({P, 1}, means3D.options())
		: torch::empty({0}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options()); // quats {P, 3, 3}
  torch::Tensor colors = torch::empty({0}, means3D.options().dtype(torch::kFloat32));
  torch::Tensor cov3D_precomp = torch::empty({0}, means3D.options().dtype(torch::kFloat32));

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  dc.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),	
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
		  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
		  out_color.contiguous().data<float>(),
		  use_depth ? out_invdepth.contiguous().data<float>() : nullptr,
		  dL_dout_color.contiguous().data<float>(),
		  use_depth ? dL_dout_invdepth.contiguous().data<float>() : nullptr,
		  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  use_depth ? dL_dinvdepths.contiguous().data<float>() : nullptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_ddc.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

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
){
	TORCH_CHECK(param.numel() == param_grad.numel(), "param and param_grad must have the same number of elements");
	TORCH_CHECK(param.numel() == exp_avg.numel(), "param and exp_avg must have the same number of elements");
	TORCH_CHECK(param.numel() == exp_avg_sq.numel(), "param and exp_avg_sq must have the same number of elements");
	TORCH_CHECK(last_seen_step.numel() >= N, "last_seen_step must contain at least N elements");
	TORCH_CHECK(visible.numel() >= N, "visible must contain at least N elements");
	TORCH_CHECK(static_cast<uint64_t>(N) * static_cast<uint64_t>(M) == static_cast<uint64_t>(param.numel()), "N * M must match param.numel()");
	if (N == 0 || M == 0)
	{
		return;
	}
	ADAM::adamUpdateTorch(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		nullptr,
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		last_seen_step.contiguous().data<int>(),
		lr_history.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		b1,
		b2,
		eps,
		global_step,
		N,
		M);
}

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
){
	TORCH_CHECK(gradient_index_map.is_cuda() && gradient_index_map.is_contiguous() && gradient_index_map.scalar_type()==torch::kInt32 && gradient_index_map.numel()==N, "invalid compact gradient map");
	TORCH_CHECK(param_grad.is_cuda() && param_grad.is_contiguous() && param_grad.scalar_type()==torch::kFloat32 && param_grad.numel()%std::max(M,1u)==0, "invalid compact gradients");
	TORCH_CHECK(param.numel() == exp_avg.numel(), "param and exp_avg must have the same number of elements");
	TORCH_CHECK(param.numel() == exp_avg_sq.numel(), "param and exp_avg_sq must have the same number of elements");
	TORCH_CHECK(last_seen_step.numel() >= N, "last_seen_step must contain at least N elements");
	TORCH_CHECK(visible.numel() >= N, "visible must contain at least N elements");
	TORCH_CHECK(static_cast<uint64_t>(N) * static_cast<uint64_t>(M) == static_cast<uint64_t>(param.numel()), "N * M must match param.numel()");
	if (N == 0 || M == 0)
	{
		return;
	}
	ADAM::adamUpdateTorch(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		gradient_index_map.data_ptr<int>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		last_seen_step.contiguous().data<int>(),
		lr_history.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		b1,
		b2,
		eps,
		global_step,
		N,
		M);
}
