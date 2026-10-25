#include "auxiliary.h"
#include "adam.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__
float computeAdamTorchDelta(
    float exp_avg_value,
    float exp_avg_sq_value,
    float lr,
    const float b1,
    const float b2,
    const float eps,
    const int step) {
    const float bias_correction1 = 1.0f - powf(b1, static_cast<float>(step));
    const float bias_correction2 = 1.0f - powf(b2, static_cast<float>(step));
    const float step_size = lr / bias_correction1;
    const float denom = sqrtf(exp_avg_sq_value) / sqrtf(bias_correction2) + eps;
    return -step_size * exp_avg_value / denom;
}

__device__ __forceinline__
void replayTorchZeroGradSteps(
    float& param_value,
    float& exp_avg_value,
    float& exp_avg_sq_value,
    const float* __restrict__ lr_history,
    const float b1,
    const float b2,
    const float eps,
    const int start_step,
    const int end_step) {
    for (int step = start_step; step <= end_step; ++step) {
        exp_avg_value = b1 * exp_avg_value;
        exp_avg_sq_value = b2 * exp_avg_sq_value;
        param_value += computeAdamTorchDelta(
            exp_avg_value,
            exp_avg_sq_value,
            lr_history[step - 1],
            b1,
            b2,
            eps,
            step
        );
    }
}

__global__
void adamUpdateTorchCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    const int* __restrict__ gradient_index_map,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    int* __restrict__ last_seen_step,
    const float* __restrict__ lr_history,
    const bool* tiles_touched,
    const float b1,
    const float b2,
    const float eps,
    const int global_step,
    const uint32_t N,
    const uint32_t M) {

	auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
    if (!tiles_touched[g_idx]) return;

    float Register_param = param[p_idx];
    float Register_exp_avg = exp_avg[p_idx];
    float Register_exp_avg_sq = exp_avg_sq[p_idx];
    const int gradient_row = gradient_index_map ? gradient_index_map[g_idx] : g_idx;
    const float Register_param_grad = gradient_row < 0 ? 0.0f : param_grad[static_cast<size_t>(gradient_row) * M + (p_idx % M)];
    const int previous_seen_step = max(last_seen_step[g_idx], 0);

    if (previous_seen_step + 1 <= global_step - 1) {
        replayTorchZeroGradSteps(
            Register_param,
            Register_exp_avg,
            Register_exp_avg_sq,
            lr_history,
            b1,
            b2,
            eps,
            previous_seen_step + 1,
            global_step - 1
        );
    }

    Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
    Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
    Register_param += computeAdamTorchDelta(
        Register_exp_avg,
        Register_exp_avg_sq,
        lr_history[global_step - 1],
        b1,
        b2,
        eps,
        global_step
    );

    param[p_idx] = Register_param;
    exp_avg[p_idx] = Register_exp_avg;
    exp_avg_sq[p_idx] = Register_exp_avg_sq;
    if ((p_idx % M) == 0) {
        last_seen_step[g_idx] = global_step;
    }
}

void ADAM::adamUpdateTorch(
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
    const uint32_t M) {

    const uint32_t cnt = N * M;
    if (cnt == 0) {
        return;
    }
    adamUpdateTorchCUDA<<<(cnt + 255) / 256, 256>>> (
        param,
        param_grad,
        gradient_index_map,
        exp_avg,
        exp_avg_sq,
        last_seen_step,
        lr_history,
        tiles_touched,
        b1,
        b2,
        eps,
        global_step,
        N,
        M
    );
}
