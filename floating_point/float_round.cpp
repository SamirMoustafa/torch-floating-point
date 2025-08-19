#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#ifdef WITH_CUDA
// CUDA Function declarations
torch::Tensor float_round_cuda_inplace(torch::Tensor input, int exponent_bits, int mantissa_bits, int bias);
#endif

// Macro to check if the tensor is contiguous
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


// CPU implementation using OpenMP
torch::Tensor float_round_cpu_inplace(torch::Tensor input, int exponent_bits, int mantissa_bits, int bias) {
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

    #pragma omp parallel for
    for (int idx = 0; idx < numel; ++idx) {
        float x_val = input_ptr[idx];
        if (x_val == 0.0f) continue;

        // Follow same logic as CUDA kernel
        const float s = std::copysign(1.0f, x_val);
        const float x_abs = std::fabs(x_val);
        const float exponent_floor = std::floor(std::log2(x_abs));

        float exponent = std::fmax(std::fmin(exponent_floor, max_exp), min_exp);
        float exp2_val = std::exp2(exponent);

        float scaled = x_abs / exp2_val;
        scaled = std::fmax(scaled, 1.0f);

        const float mantissa_unrounded = (scaled - 1.0f) * mantissa_scale;
        const int mantissa = static_cast<int>(std::round(mantissa_unrounded));

        const bool overflow = mantissa >= mantissa_upper_bound;
        const float exponent_overflow = std::fmax(std::fmin(exponent + 1.0f, max_exp), min_exp);
        const float exp2_val_overflow = std::exp2(exponent_overflow);

        const float final_exp2 = overflow ? exp2_val_overflow : exp2_val;
        const int final_mantissa = overflow ? 0 : mantissa;

        const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
        input_ptr[idx] = s * (1.0f + fraction) * final_exp2;
    }

    return input;
}

torch::Tensor float_round_inplace(torch::Tensor input, int exponent_bits, int mantissa_bits, int bias) {
    // Check if the input tensor is contiguous and of type float32
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    // Call the appropriate implementation based on the device
    if (input.device().is_cuda()) {
        #ifdef WITH_CUDA
        return float_round_cuda_inplace(input, exponent_bits, mantissa_bits, bias);
        #else
        TORCH_CHECK(false, "CUDA support not available");
        #endif
    } else {
        return float_round_cpu_inplace(input, exponent_bits, mantissa_bits, bias);
    }
}

// Autograd function definition
class RoundFunction : public torch::autograd::Function<RoundFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor input,
                                 int exponent_bits,
                                 int mantissa_bits,
                                 int bias) {
        ctx->saved_data["exponent_bits"] = exponent_bits;
        ctx->saved_data["mantissa_bits"] = mantissa_bits;
        ctx->saved_data["bias"] = bias;

        auto output = float_round_inplace(input.clone(), exponent_bits, mantissa_bits, bias);
        return output;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::tensor_list grad_outputs) {
        auto grad_output = float_round_inplace(grad_outputs[0].contiguous().clone(),
                                                ctx->saved_data["exponent_bits"].toInt(),
                                                ctx->saved_data["mantissa_bits"].toInt(),
                                                ctx->saved_data["bias"].toInt());
        return {grad_output, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

// Autograd-enabled wrapper
torch::Tensor float_round_autograd(torch::Tensor input, int exponent_bits, int mantissa_bits, int bias) {
    return RoundFunction::apply(input, exponent_bits, mantissa_bits, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inplace", &float_round_inplace, "Float rounding operation (CUDA/CPU, inplace)");
    m.def("autograd", &float_round_autograd, "Float rounding operation with autograd support");
}
