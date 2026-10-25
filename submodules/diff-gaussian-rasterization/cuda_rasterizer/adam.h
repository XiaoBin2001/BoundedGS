#ifndef CUDA_RASTERIZER_ADAM_H_INCLUDED
#define CUDA_RASTERIZER_ADAM_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ADAM {

void adamUpdateTorch(
    float* param,
    const float* param_grad,
    const int* gradient_index_map,
    float* exp_avg,
    float* exp_avg_sq,
    int* last_seen_step,
    const float* lr_history,
    const bool* tiles_touched,
    const float b1,
    const float b2,
    const float eps,
    const int global_step,
    const uint32_t N,
    const uint32_t M);
}

#endif
