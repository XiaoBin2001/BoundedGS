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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint32_t *bucket_count;
		uint32_t *bucket_offsets;
		size_t bucket_count_scan_size;
		char * bucket_count_scanning_space;
		uint32_t* max_contrib;
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t num_pixels, size_t num_tiles);
	};

	struct BinningState
	{
		size_t scan_size;
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		int* scan_src;
		int* scan_dst;
		char* scan_space;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	struct SampleState
	{
		uint32_t *bucket_to_tile;
		float *T;
		float *ar;
		float *ard;
		static SampleState fromChunk(char*& chunk, size_t C, bool include_depth = true);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	inline size_t requiredImage(size_t num_pixels, size_t num_tiles)
	{
		char* size = nullptr;
		ImageState::fromChunk(size, num_pixels, num_tiles);
		return ((size_t)size) + 128;
	}

	inline size_t requiredSample(size_t bucket_count, bool include_depth)
	{
		char* size = nullptr;
		SampleState::fromChunk(size, bucket_count, include_depth);
		return ((size_t)size) + 128;
	}

};
