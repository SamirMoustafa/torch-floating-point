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

__device__ __forceinline__ float float_round_one(
    float x_val,
    float max_exp,
    float min_exp,
    int mantissa_upper_bound,
    int max_mant_at_max,
    float mantissa_scale,
    float inv_mantissa_scale) {
    if (x_val == 0.0f) return x_val;

    const float s = copysignf(1.0f, x_val);
    const float x_abs = fabsf(x_val);
    const float exponent_floor = floorf(log2f(x_abs));
    float exponent = fmaxf(fminf(exponent_floor, max_exp), min_exp);
    float exp2_val = exp2f(exponent);

    float scaled = fmaf(x_abs, __frcp_rn(exp2_val), 0.0f);
    scaled = fmaxf(scaled, 1.0f);

    const float mantissa_unrounded = fmaf(scaled - 1.0f, mantissa_scale, 0.0f);
    int mantissa = __float2int_rn(mantissa_unrounded);

    const bool at_max_exp = exponent >= max_exp;
    const int effective_ub = at_max_exp ? (max_mant_at_max + 1) : mantissa_upper_bound;

    float final_exp2 = exp2_val;
    int final_mantissa = mantissa;
    if (mantissa >= effective_ub) {
        if (at_max_exp) {
            final_mantissa = max_mant_at_max;
        } else {
            const float exponent_overflow = fmaxf(fminf(exponent + 1.0f, max_exp), min_exp);
            final_exp2 = exp2f(exponent_overflow);
            final_mantissa = 0;
        }
    }

    const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
    return fmaf(fmaf(fraction, final_exp2, final_exp2), s, 0.0f);
}

// Optimized kernel with improved memory access patterns
__global__ void float_round_kernel_inplace(float* input,
                                           int N,
                                           float max_exp,
                                           float min_exp,
                                           int mantissa_upper_bound,
                                           int max_mant_at_max,
                                           float mantissa_scale,
                                           float inv_mantissa_scale) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N; idx += stride) {
        input[idx] = float_round_one(
            input[idx], max_exp, min_exp, mantissa_upper_bound, max_mant_at_max,
            mantissa_scale, inv_mantissa_scale);
    }
}

// Vectorized kernel using float4 for maximum memory bandwidth
__global__ void float_round_kernel_vectorized(float4* input_vec,
                                             int N_vec,
                                             float max_exp,
                                             float min_exp,
                                             int mantissa_upper_bound,
                                             int max_mant_at_max,
                                             float mantissa_scale,
                                             float inv_mantissa_scale) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N_vec; idx += stride) {
        float4 vec = input_vec[idx];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float* x_ptr = reinterpret_cast<float*>(&vec) + i;
            *x_ptr = float_round_one(
                *x_ptr, max_exp, min_exp, mantissa_upper_bound, max_mant_at_max,
                mantissa_scale, inv_mantissa_scale);
        }

        input_vec[idx] = vec;
    }
}

// Shared memory optimized kernel for better cache utilization
__global__ void float_round_kernel_shared(float* input,
                                         int N,
                                         float max_exp,
                                         float min_exp,
                                         int mantissa_upper_bound,
                                         int max_mant_at_max,
                                         float mantissa_scale,
                                         float inv_mantissa_scale) {
    __shared__ float shared_data[1024];

    const int tid = threadIdx.x;

    for (int base_idx = blockIdx.x * blockDim.x; base_idx < N; base_idx += blockDim.x * gridDim.x) {
        int idx = base_idx + tid;

        if (idx < N) {
            shared_data[tid] = input[idx];
        } else {
            shared_data[tid] = 0.0f;
        }

        __syncthreads();

        if (idx < N) {
            shared_data[tid] = float_round_one(
                shared_data[tid], max_exp, min_exp, mantissa_upper_bound, max_mant_at_max,
                mantissa_scale, inv_mantissa_scale);
        }

        __syncthreads();

        if (idx < N) {
            input[idx] = shared_data[tid];
        }
    }
}

// Function that launches the optimized kernel
torch::Tensor float_round_cuda_inplace(
    torch::Tensor input,
    int exponent_bits,
    int mantissa_bits,
    int bias,
    int reserved_exponent,
    int max_mantissa_at_max_exponent) {
    CHECK_CUDA(input);

    int numel = input.numel();
    if (numel == 0) return input;

    int max_stored_exp = reserved_exponent ? ((1 << exponent_bits) - 2) : ((1 << exponent_bits) - 1);
    float max_exp = static_cast<float>(max_stored_exp - bias);
    float min_exp = static_cast<float>(-bias);
    int mantissa_upper_bound = 1 << mantissa_bits;
    int max_mant_at_max = max_mantissa_at_max_exponent;
    float mantissa_scale = static_cast<float>(mantissa_upper_bound);
    float inv_mantissa_scale = 1.0f / mantissa_scale;

    float* input_ptr = input.data_ptr<float>();

    int device_id = input.device().index();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    int max_blocks = prop.multiProcessorCount * max_blocks_per_sm;
    blocks = min(blocks, max_blocks);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (numel >= 1000000) {
        if (numel % 4 == 0) {
            float4* input_vec = reinterpret_cast<float4*>(input_ptr);
            int N_vec = numel / 4;
            float_round_kernel_vectorized<<<blocks, threads, 0, stream>>>(
                input_vec, N_vec, max_exp, min_exp,
                mantissa_upper_bound, max_mant_at_max, mantissa_scale, inv_mantissa_scale
            );
        } else {
            float_round_kernel_shared<<<blocks, threads, 0, stream>>>(
                input_ptr, numel, max_exp, min_exp,
                mantissa_upper_bound, max_mant_at_max, mantissa_scale, inv_mantissa_scale
            );
        }
    } else {
        float_round_kernel_inplace<<<blocks, threads, 0, stream>>>(
            input_ptr, numel, max_exp, min_exp,
            mantissa_upper_bound, max_mant_at_max, mantissa_scale, inv_mantissa_scale
        );
    }

    CUDA_CHECK(cudaGetLastError());

    return input;
}
