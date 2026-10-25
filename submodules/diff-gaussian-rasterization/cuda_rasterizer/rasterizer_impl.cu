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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

namespace
{
	void checkCudaRuntime(cudaError_t result, const char* stage)
	{
		if (result != cudaSuccess)
		{
			throw std::runtime_error(
				std::string("CUDA failure at ") + stage + ": " + cudaGetErrorString(result));
		}
	}
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float4* __restrict__ conic_opacity,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	uint32_t* tiles_touched,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (tiles_touched[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		duplicateToTilesTouched(
			points_xy[idx], conic_opacity[idx], grid,
			idx, off, depths[idx],
			gaussian_keys_unsorted,
			gaussian_values_unsorted);
	}
}

// Count Gaussian/tile overlaps for one rectangle in full-image tile coordinates.
// Clipping the full-image ellipse/tile result avoids recomputing intersections
// after translating the projected center into rectangle-local floating-point space.
__global__ void countTilesForRect(
	int candidate_count,
	const int* candidate_indices,
	const float2* points_xy,
	const float4* __restrict__ conic_opacity,
	const int* radii,
	uint32_t* tiles_touched,
	dim3 full_grid,
	dim3 rect_tile_grid,
	int rect_tile_x,
	int rect_tile_y)
{
	auto local_idx = cg::this_grid().thread_rank();
	if (local_idx >= candidate_count)
		return;
	const int idx = candidate_indices == nullptr ? local_idx : candidate_indices[local_idx];

	tiles_touched[local_idx] = 0;
	if (radii[idx] <= 0)
		return;

	const int2 rect_tile_min = { rect_tile_x, rect_tile_y };
	const int2 rect_tile_max = {
		rect_tile_x + static_cast<int>(rect_tile_grid.x),
		rect_tile_y + static_cast<int>(rect_tile_grid.y)
	};
	tiles_touched[local_idx] = duplicateToTilesTouchedClipped(
		points_xy[idx],
		conic_opacity[idx],
		full_grid,
		rect_tile_min,
		rect_tile_max,
		rect_tile_grid,
		rect_tile_min,
		0,
		0,
		UINT32_MAX,
		0.0f,
		nullptr,
		nullptr);
}

__global__ void duplicateWithKeysRect(
	int candidate_count,
	const int* candidate_indices,
	const float2* points_xy,
	const float4* __restrict__ conic_opacity,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const uint32_t* tiles_touched,
	dim3 full_grid,
	dim3 rect_tile_grid,
	int rect_tile_x,
	int rect_tile_y)
{
	auto local_idx = cg::this_grid().thread_rank();
	if (local_idx >= candidate_count || tiles_touched[local_idx] == 0)
		return;
	const int idx = candidate_indices == nullptr ? local_idx : candidate_indices[local_idx];

	const uint32_t off = (local_idx == 0) ? 0 : offsets[local_idx - 1];
	const uint32_t write_end = offsets[local_idx];
	const int2 rect_tile_min = { rect_tile_x, rect_tile_y };
	const int2 rect_tile_max = {
		rect_tile_x + static_cast<int>(rect_tile_grid.x),
		rect_tile_y + static_cast<int>(rect_tile_grid.y)
	};
	const uint32_t generated_count = duplicateToTilesTouchedClipped(
		points_xy[idx],
		conic_opacity[idx],
		full_grid,
		rect_tile_min,
		rect_tile_max,
		rect_tile_grid,
		rect_tile_min,
		idx,
		off,
		write_end,
		depths[idx],
		gaussian_keys_unsorted,
		gaussian_values_unsorted);

	const uint32_t expected_count = write_end - off;
	const uint32_t written_count = min(generated_count, expected_count);
	// The count and fill kernels are compiled separately, so an ellipse exactly
	// on a tile boundary can very rarely round differently. Keep every Gaussian
	// inside its scanned slice and mark any unfilled tail as non-renderable.
	for (uint32_t output_index = off + written_count; output_index < write_end; ++output_index)
	{
		gaussian_keys_unsorted[output_index] = UINT64_MAX;
		gaussian_values_unsorted[output_index] = 0;
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(
	int L,
	uint64_t* point_list_keys,
	uint2* ranges,
	uint32_t tile_count)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile < tile_count;

	if (idx == 0)
	{
		if (valid_tile)
			ranges[currtile].x = 0;
	}
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			if (prevtile < tile_count)
				ranges[prevtile].y = idx;
			if (valid_tile) 
				ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
}

// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;
	
	uint2 range = ranges[idx];
	int num_splats = range.y - range.x;
	int num_buckets = (num_splats + 31) / 32;
	bucketCount[idx] = (uint32_t) num_buckets;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t num_pixels, size_t num_tiles)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, num_pixels, 128);
	obtain(chunk, img.n_contrib, num_pixels, 128);
	obtain(chunk, img.ranges, num_tiles, 128);
	obtain(chunk, img.max_contrib, num_tiles, 128);
	obtain(chunk, img.bucket_count, num_tiles, 128);
	obtain(chunk, img.bucket_offsets, num_tiles, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, num_tiles);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

	return img;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(
	char *& chunk,
	size_t C,
	bool include_depth) {
	SampleState sample;
	obtain(chunk, sample.bucket_to_tile, C, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHANNELS_3DGS * C * BLOCK_SIZE, 128);
	sample.ard = nullptr;
	if (include_depth)
		obtain(chunk, sample.ard, C * BLOCK_SIZE, 128);
	return sample;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__global__ void zero(int N, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;
	space[idx] = 0;
}

__global__ void set(int N, uint32_t* where, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;

	int off = (idx == 0) ? 0 : where[idx-1];

	space[off] = 1;
}

void CudaRasterizer::Rasterizer::project(
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
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
	if (radii == nullptr)
		radii = geomState.internal_radii;

	CHECK_CUDA(FORWARD::project(
		P, D, M,
		means3D,
		(raster_math::vec3*)scales,
		scale_modifier,
		(raster_math::vec4*)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		(raster_math::vec3*)cam_pos,
		width,
		height,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.tiles_touched,
		prefiltered,
		antialiasing), debug)
}

void CudaRasterizer::Rasterizer::project_indexed_raw(
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
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
	if (radii == nullptr)
		radii = geomState.internal_radii;

	CHECK_CUDA(FORWARD::project_indexed_raw(
		P, D, M,
		candidate_indices,
		means3D,
		(raster_math::vec3*)raw_scales,
		scale_modifier,
		(raster_math::vec4*)raw_rotations,
		raw_opacities,
		dc,
		shs,
		geomState.clamped,
		viewmatrix,
		projmatrix,
		(raster_math::vec3*)cam_pos,
		width,
		height,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.tiles_touched,
		prefiltered,
		antialiasing), debug)
}

int CudaRasterizer::Rasterizer::count_rect(
	const int P,
	const int candidate_count,
	const int* candidate_indices,
	const int full_width, int full_height,
	const int rect_x, int rect_y,
	const int rect_width, int rect_height,
	const int* radii,
	char* geom_buffer,
	bool debug)
{
	if (
		rect_x < 0 || rect_y < 0 || rect_width <= 0 || rect_height <= 0
		|| rect_x + rect_width > full_width || rect_y + rect_height > full_height)
	{
		throw std::runtime_error("rectangle must be a non-empty rectangle inside the full image.");
	}
	if (rect_x % BLOCK_X != 0 || rect_y % BLOCK_Y != 0)
	{
		throw std::runtime_error("rectangle origins must align to rasterizer tile boundaries.");
	}
	if (candidate_count <= 0)
		return 0;
	if (candidate_count > P)
		throw std::runtime_error("rectangle candidate count exceeds projected Gaussian count.");
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	const dim3 full_tile_grid(
		(full_width + BLOCK_X - 1) / BLOCK_X,
		(full_height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const dim3 tile_grid(
		(rect_width + BLOCK_X - 1) / BLOCK_X,
		(rect_height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const int rect_tile_x = rect_x / BLOCK_X;
	const int rect_tile_y = rect_y / BLOCK_Y;

	countTilesForRect <<<(candidate_count + 255) / 256, 256>>>(
		candidate_count,
		candidate_indices,
		geomState.means2D,
		geomState.conic_opacity,
		radii,
		geomState.tiles_touched,
		full_tile_grid,
		tile_grid,
		rect_tile_x,
		rect_tile_y);
	CHECK_CUDA(, debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space,
		geomState.scan_size,
		geomState.tiles_touched,
		geomState.point_offsets,
		candidate_count), debug)
	int num_rendered = 0;
	checkCudaRuntime(cudaMemcpy(
		&num_rendered,
		geomState.point_offsets + candidate_count - 1,
		sizeof(int),
		cudaMemcpyDeviceToHost), "rectangle tile-count readback");
	return num_rendered;
}

std::tuple<int,int> CudaRasterizer::Rasterizer::forward_rect(
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
	float* invdepth,
	bool background_is_zero,
	bool debug)
{
	const int num_rendered = count_rect(
		P,
		candidate_count,
		candidate_indices,
		full_width,
		full_height,
		rect_x,
		rect_y,
		rect_width,
		rect_height,
		radii,
		geom_buffer,
		debug);
	return forward_rect_precounted(
		binningBuffer,
		imageBuffer,
		sampleBuffer,
		P,
		candidate_count,
		candidate_indices,
		background,
		full_width,
		full_height,
		rect_x,
		rect_y,
		rect_width,
		rect_height,
		radii,
		geom_buffer,
		out_color,
		invdepth,
		num_rendered,
		background_is_zero,
		debug);
}

std::tuple<int,int> CudaRasterizer::Rasterizer::forward_rect_precounted(
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
	float* invdepth,
	const int precounted_num_rendered,
	bool background_is_zero,
	bool debug)
{
	if (
		rect_x < 0 || rect_y < 0 || rect_width <= 0 || rect_height <= 0
		|| rect_x + rect_width > full_width || rect_y + rect_height > full_height)
	{
		throw std::runtime_error("rectangle must be a non-empty rectangle inside the full image.");
	}
	if (rect_x % BLOCK_X != 0 || rect_y % BLOCK_Y != 0)
	{
		throw std::runtime_error("rectangle origins must align to rasterizer tile boundaries.");
	}
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	const dim3 full_tile_grid(
		(full_width + BLOCK_X - 1) / BLOCK_X,
		(full_height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const dim3 tile_grid(
		(rect_width + BLOCK_X - 1) / BLOCK_X,
		(rect_height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int rect_tile_x = rect_x / BLOCK_X;
	const int rect_tile_y = rect_y / BLOCK_Y;
	const int num_tiles = tile_grid.x * tile_grid.y;
	const int num_pixels = rect_width * rect_height;

	size_t img_chunk_size = requiredImage(num_pixels, num_tiles);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, num_pixels, num_tiles);

	if (precounted_num_rendered < 0)
		throw std::runtime_error("Precounted rectangle reference count must be non-negative.");
	const int num_rendered = precounted_num_rendered;

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	if (num_rendered > 0)
	{
		duplicateWithKeysRect <<<(candidate_count + 255) / 256, 256>>>(
			candidate_count,
			candidate_indices,
			geomState.means2D,
			geomState.conic_opacity,
			geomState.depths,
			geomState.point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted,
			geomState.tiles_touched,
			full_tile_grid,
			tile_grid,
			rect_tile_x,
			rect_tile_y);
		CHECK_CUDA(, debug)

		const int bit = getHigherMsb(tile_grid.x * tile_grid.y);
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space,
			binningState.sorting_size,
			binningState.point_list_keys_unsorted,
			binningState.point_list_keys,
			binningState.point_list_unsorted,
			binningState.point_list,
			num_rendered,
			0,
			32 + bit), debug)
	}

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, num_tiles * sizeof(uint2)), debug)
	if (num_rendered > 0)
	{
		identifyTileRanges <<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges,
			num_tiles);
		CHECK_CUDA(, debug)
	}

	perTileBucketCount <<<(num_tiles + 255) / 256, 256>>>(
		num_tiles,
		imgState.ranges,
		imgState.bucket_count);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		imgState.bucket_count_scanning_space,
		imgState.bucket_count_scan_size,
		imgState.bucket_count,
		imgState.bucket_offsets,
		num_tiles), debug)

	unsigned int bucket_sum = 0;
	checkCudaRuntime(cudaMemcpy(
		&bucket_sum,
		imgState.bucket_offsets + num_tiles - 1,
		sizeof(unsigned int),
		cudaMemcpyDeviceToHost), "rectangle bucket-count readback");

	const bool use_depth = invdepth != nullptr;
	size_t sample_chunk_size = requiredSample(bucket_sum, use_depth);
	char* sample_chunkptr = sampleBuffer(sample_chunk_size);
	SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum, use_depth);

	CHECK_CUDA(FORWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		sampleState.ard,
		rect_width,
		rect_height,
		rect_x,
		rect_y,
		geomState.means2D,
		geomState.rgb,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		background,
		out_color,
		geomState.depths,
		invdepth,
		background_is_zero), debug)

	return std::make_tuple(num_rendered, static_cast<int>(bucket_sum));
}

void CudaRasterizer::Rasterizer::backward_rect(
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
	bool debug)
{
	const dim3 tile_grid(
		(rect_width + BLOCK_X - 1) / BLOCK_X,
		(rect_height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int num_tiles = tile_grid.x * tile_grid.y;
	const int num_pixels = rect_width * rect_height;

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(image_buffer, num_pixels, num_tiles);
	const bool use_depth = dL_invdepths != nullptr;
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B, use_depth);

	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		rect_width,
		rect_height,
		R,
		B,
		full_width,
		full_height,
		rect_x,
		rect_y,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		sampleState.ard,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.rgb,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		pixel_colors,
		pixel_invdepths,
		dL_dpix,
		dL_invdepths,
		(float4*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth,
		!background_is_zero), debug)
}

void CudaRasterizer::Rasterizer::backward_project(
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
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	CHECK_CUDA(BACKWARD::preprocess(
		P,
		D,
		M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		opacities,
		(raster_math::vec3*)scales,
		(raster_math::vec4*)rotations,
		scale_modifier,
		geomState.cov3D,
		viewmatrix,
		projmatrix,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		(raster_math::vec3*)campos,
		(const float4*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(raster_math::vec3*)dL_dmean3D,
		(float*)dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(raster_math::vec3*)dL_dscale,
		(raster_math::vec4*)dL_drot,
		antialiasing), debug)
}

void CudaRasterizer::Rasterizer::backward_project_indexed_raw(
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
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	CHECK_CUDA(BACKWARD::preprocess_indexed_raw(
		P,
		D,
		M,
		candidate_indices,
		gradient_index_map,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		raw_opacities,
		(raster_math::vec3*)raw_scales,
		(raster_math::vec4*)raw_rotations,
		scale_modifier,
		geomState.cov3D,
		viewmatrix,
		projmatrix,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		(raster_math::vec3*)campos,
		(const float4*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(float*)dL_dcolor,
		dL_dcov3D,
		(raster_math::vec3*)out_means3D,
		out_dc,
		out_sh,
		out_opacity,
		(raster_math::vec3*)out_scales,
		(raster_math::vec4*)out_rotations,
		antialiasing), debug)
}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int,int> CudaRasterizer::Rasterizer::forward(
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
	float* invdepth,
	bool antialiasing,
	bool inference_only,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int num_tiles = tile_grid.x * tile_grid.y;
	const int num_pixels = width * height;

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = requiredImage(num_pixels, num_tiles);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, num_pixels, num_tiles);

	if (NUM_CHANNELS_3DGS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(raster_math::vec3*)scales,
		scale_modifier,
		(raster_math::vec4*)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(raster_math::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		geomState.tiles_touched,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges,
			num_tiles);
	CHECK_CUDA(, debug)

	 	// bucket count
	unsigned int bucket_sum = 0;
	SampleState sampleState{};
	if (!inference_only)
	{
		perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug)
		CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
		// create a state to store. size is number is the total number of buckets * block_size
		const bool use_depth = invdepth != nullptr;
		size_t sample_chunk_size = requiredSample(bucket_sum, use_depth);
		char* sample_chunkptr = sampleBuffer(sample_chunk_size);
		sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum, use_depth);
	}

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		inference_only ? nullptr : imgState.bucket_offsets,
		inference_only ? nullptr : sampleState.bucket_to_tile,
		inference_only ? nullptr : sampleState.T,
		inference_only ? nullptr : sampleState.ar,
		inference_only ? nullptr : sampleState.ard,
		width, height,
		0, 0,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		inference_only ? nullptr : imgState.accum_alpha,
		inference_only ? nullptr : imgState.n_contrib,
		inference_only ? nullptr : imgState.max_contrib,
		background,
		out_color,
		inference_only ? nullptr : geomState.depths,
		inference_only ? nullptr : invdepth,
		false), debug)
	return std::make_tuple(num_rendered, bucket_sum);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
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
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int num_tiles = tile_grid.x * tile_grid.y;
	const int num_pixels = width * height;

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, num_pixels, num_tiles);
	const bool use_depth = dL_invdepths != nullptr;
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B, use_depth);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, R, B,
		width, height,
		0, 0,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		sampleState.ard,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		pixel_colors,
		pixel_invdepths,
		dL_dpix,
		dL_invdepths,
		(float4*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor, 
		dL_dinvdepth,
		true), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		opacities,
		(raster_math::vec3*)scales,
		(raster_math::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(raster_math::vec3*)campos,
		(float4*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(raster_math::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(raster_math::vec3*)dL_dscale,
		(raster_math::vec4*)dL_drot,
		antialiasing), debug)
}
