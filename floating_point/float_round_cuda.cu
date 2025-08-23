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

// Optimized kernel with improved memory access patterns
__global__ void float_round_kernel_inplace(float* input,
                                           int N,
                                           float max_exp,
                                           float min_exp,
                                           int mantissa_upper_bound,
                                           float mantissa_scale,
                                           float inv_mantissa_scale) {
    // Use vectorized loads for better memory coalescing
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Process multiple elements per thread to improve memory bandwidth utilization
    for (int idx = tid; idx < N; idx += stride) {
        float x_val = input[idx];

        // Early exit for zero values (reduces unnecessary computation)
        if (x_val == 0.0f) continue;

        // Use fast math intrinsics for better performance
        const float s = copysignf(1.0f, x_val);
        const float x_abs = fabsf(x_val);

        // Use fast log2 and exp2 intrinsics
        const float exponent_floor = log2f(x_abs);
        float exponent = fmaxf(fminf(exponent_floor, max_exp), min_exp);
        float exp2_val = exp2f(exponent);

        // Optimize division with reciprocal multiplication
        float scaled = fmaf(x_abs, __frcp_rn(exp2_val), 0.0f);
        scaled = fmaxf(scaled, 1.0f);

        // Use FMA for better instruction fusion
        const float mantissa_unrounded = fmaf(scaled - 1.0f, mantissa_scale, 0.0f);
        const int mantissa = __float2int_rn(mantissa_unrounded);

        // Branchless overflow handling with predicated execution
        const bool overflow = mantissa >= mantissa_upper_bound;
        const float exponent_overflow = fmaxf(fminf(fmaf(exponent, 1.0f, 1.0f), max_exp), min_exp);
        const float exp2_val_overflow = exp2f(exponent_overflow);

        // Select final values without branches using predication
        const float final_exp2 = overflow ? exp2_val_overflow : exp2_val;
        const int final_mantissa = overflow ? 0 : mantissa;

        // Use FMA for final computation
        const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
        input[idx] = fmaf(fmaf(fraction, final_exp2, final_exp2), s, 0.0f);
    }
}

// Vectorized kernel using float4 for maximum memory bandwidth
__global__ void float_round_kernel_vectorized(float4* input_vec,
                                             int N_vec,
                                             float max_exp,
                                             float min_exp,
                                             int mantissa_upper_bound,
                                             float mantissa_scale,
                                             float inv_mantissa_scale) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Process float4 elements (4 floats per thread)
    for (int idx = tid; idx < N_vec; idx += stride) {
        float4 vec = input_vec[idx];

        // Process each component of the float4 vector
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float* x_ptr = reinterpret_cast<float*>(&vec) + i;
            float x_val = *x_ptr;

            if (x_val == 0.0f) continue;

            // Use fast math intrinsics
            const float s = copysignf(1.0f, x_val);
            const float x_abs = fabsf(x_val);
            const float exponent_floor = log2f(x_abs);
            float exponent = fmaxf(fminf(exponent_floor, max_exp), min_exp);
            float exp2_val = exp2f(exponent);

            // Optimized computation with FMA
            float scaled = fmaf(x_abs, __frcp_rn(exp2_val), 0.0f);
            scaled = fmaxf(scaled, 1.0f);

            const float mantissa_unrounded = fmaf(scaled - 1.0f, mantissa_scale, 0.0f);
            const int mantissa = __float2int_rn(mantissa_unrounded);

            const bool overflow = mantissa >= mantissa_upper_bound;
            const float exponent_overflow = fmaxf(fminf(fmaf(exponent, 1.0f, 1.0f), max_exp), min_exp);
            const float exp2_val_overflow = exp2f(exponent_overflow);

            const float final_exp2 = overflow ? exp2_val_overflow : exp2_val;
            const int final_mantissa = overflow ? 0 : mantissa;

            const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
            *x_ptr = fmaf(fmaf(fraction, final_exp2, final_exp2), s, 0.0f);
        }

        // Store the processed float4 vector
        input_vec[idx] = vec;
    }
}

// Shared memory optimized kernel for better cache utilization
__global__ void float_round_kernel_shared(float* input,
                                         int N,
                                         float max_exp,
                                         float min_exp,
                                         int mantissa_upper_bound,
                                         float mantissa_scale,
                                         float inv_mantissa_scale) {
    __shared__ float shared_data[1024]; // Shared memory buffer

    const int tid = threadIdx.x;

    for (int base_idx = blockIdx.x * blockDim.x; base_idx < N; base_idx += blockDim.x * gridDim.x) {
        int idx = base_idx + tid;

        // Load data into shared memory with coalesced access
        if (idx < N) {
            shared_data[tid] = input[idx];
        } else {
            shared_data[tid] = 0.0f;
        }

        __syncthreads();

        // Process data from shared memory
        if (idx < N) {
            float x_val = shared_data[tid];

            if (x_val != 0.0f) {
                // Use fast math intrinsics
                const float s = copysignf(1.0f, x_val);
                const float x_abs = fabsf(x_val);
                const float exponent_floor = log2f(x_abs);
                float exponent = fmaxf(fminf(exponent_floor, max_exp), min_exp);
                float exp2_val = exp2f(exponent);

                // Optimized computation
                float scaled = fmaf(x_abs, __frcp_rn(exp2_val), 0.0f);
                scaled = fmaxf(scaled, 1.0f);

                const float mantissa_unrounded = fmaf(scaled - 1.0f, mantissa_scale, 0.0f);
                const int mantissa = __float2int_rn(mantissa_unrounded);

                const bool overflow = mantissa >= mantissa_upper_bound;
                const float exponent_overflow = fmaxf(fminf(fmaf(exponent, 1.0f, 1.0f), max_exp), min_exp);
                const float exp2_val_overflow = exp2f(exponent_overflow);

                const float final_exp2 = overflow ? exp2_val_overflow : exp2_val;
                const int final_mantissa = overflow ? 0 : mantissa;

                const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
                shared_data[tid] = fmaf(fmaf(fraction, final_exp2, final_exp2), s, 0.0f);
            }
        }

        __syncthreads();

        // Store back to global memory with coalesced access
        if (idx < N) {
            input[idx] = shared_data[tid];
        }
    }
}

// Function that launches the optimized kernel
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

    // Optimize block and grid size for better occupancy
    int device_id = input.device().index();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // Calculate optimal block size based on register usage and shared memory
    int threads = 256; // Reduced from 1024 to improve occupancy
    int blocks = (numel + threads - 1) / threads;

    // Ensure we don't exceed maximum blocks per SM
    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    int max_blocks = prop.multiProcessorCount * max_blocks_per_sm;
    blocks = min(blocks, max_blocks);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Choose kernel based on input size and optimization strategy
    if (numel >= 1000000) {
        // For large inputs, use vectorized kernel if possible
        if (numel % 4 == 0) {
            float4* input_vec = reinterpret_cast<float4*>(input_ptr);
            int N_vec = numel / 4;
            float_round_kernel_vectorized<<<blocks, threads, 0, stream>>>(
                input_vec, N_vec, max_exp, min_exp,
                mantissa_upper_bound, mantissa_scale, inv_mantissa_scale
            );
        } else {
            // Use shared memory kernel for better cache utilization
            float_round_kernel_shared<<<blocks, threads, 0, stream>>>(
                input_ptr, numel, max_exp, min_exp,
                mantissa_upper_bound, mantissa_scale, inv_mantissa_scale
            );
        }
    } else {
        // For smaller inputs, use optimized kernel
        float_round_kernel_inplace<<<blocks, threads, 0, stream>>>(
            input_ptr, numel, max_exp, min_exp,
            mantissa_upper_bound, mantissa_scale, inv_mantissa_scale
        );
    }

    CUDA_CHECK(cudaGetLastError());

    return input;
}