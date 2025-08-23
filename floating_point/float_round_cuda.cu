#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h> // For getCurrentCUDAStream

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

inline void gpuCheck(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
        const char* errName = cudaGetErrorName(code);
        const char* errString = cudaGetErrorString(code);
        TORCH_CHECK(false, "CUDA error: ", errName, " ", errString, " at ", file, ":", line);
    }
  }
#define CUDA_CHECK(ans) { gpuCheck((ans), __FILE__, __LINE__); }

__global__ void float_round_kernel_inplace(float* input,
                                           int N,
                                           float max_exp,
                                           float min_exp,
                                           int mantissa_upper_bound,
                                           float mantissa_scale,
                                           float inv_mantissa_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x_val = input[idx];
    if (x_val == 0.0f) return;

    // 1. Use standard math functions with fast math optimizations
    const float s = copysignf(1.0f, x_val);
    const float x_abs = fabsf(x_val);
    const float exponent_floor = floorf(log2f(x_abs));  // Will be optimized with --use_fast_math

    float exponent = fmaxf(fminf(exponent_floor, max_exp), min_exp);
    float exp2_val = exp2f(exponent);  // Compiler will optimize with --use_fast_math

    float scaled = x_abs / exp2_val;
    scaled = fmaxf(scaled, 1.0f);

    // 2. Use CUDA's built-in rounding
    const float mantissa_unrounded = (scaled - 1.0f) * mantissa_scale;
    const int mantissa = __float2int_rn(mantissa_unrounded);

    // 3. Branchless overflow handling
    const bool overflow = mantissa >= mantissa_upper_bound;
    const float exponent_overflow = fmaxf(fminf(exponent + 1.0f, max_exp), min_exp);
    const float exp2_val_overflow = exp2f(exponent_overflow);

    // 4. Select final values without branches
    const float final_exp2 = overflow ? exp2_val_overflow : exp2_val;
    const int final_mantissa = overflow ? 0 : mantissa;

    // 5. FMA is automatically used with --use_fast_math
    const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
    input[idx] = s * (1.0f + fraction) * final_exp2;
}

// Function that launches the kernel
torch::Tensor float_round_cuda_inplace(torch::Tensor input, int exponent_bits, int mantissa_bits, int bias) {
    CHECK_CUDA(input);

    int numel = input.numel();
    if (numel == 0) return input;

    // Precompute constants
    int max_exp_val = (1 << exponent_bits) - 1 - bias;
    float max_exp = static_cast<float>(max_exp_val);
    float min_exp = static_cast<float>(-bias);
    int mantissa_upper_bound = 1 << mantissa_bits;
    float mantissa_scale = static_cast<float>(mantissa_upper_bound);
    float inv_mantissa_scale = 1.0f / mantissa_scale;

    float* input_ptr = input.data_ptr<float>();
    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float_round_kernel_inplace<<<blocks, threads, 0, stream>>>(
        input_ptr, numel, max_exp, min_exp,
        mantissa_upper_bound, mantissa_scale, inv_mantissa_scale
    );
    CUDA_CHECK(cudaGetLastError());

    return input;
}