#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#ifdef WITH_CUDA
// CUDA Function declarations
torch::Tensor float_round_cuda_inplace(
    torch::Tensor input,
    int exponent_bits,
    int mantissa_bits,
    int bias,
    int reserved_exponent,
    int max_mantissa_at_max_exponent);
#endif

// Macro to check if the tensor is contiguous
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {

inline int round_ties_to_even(float x) {
    return static_cast<int>(std::nearbyint(x));
}

}  // namespace

// CPU implementation using OpenMP
torch::Tensor float_round_cpu_inplace(
    torch::Tensor input,
    int exponent_bits,
    int mantissa_bits,
    int bias,
    int reserved_exponent,
    int max_mantissa_at_max_exponent) {
    int numel = input.numel();
    if (numel == 0) return input;

    // Precompute constants (respect reserved max exponent / FN mantissa cap)
    int max_stored_exp = reserved_exponent ? ((1 << exponent_bits) - 2) : ((1 << exponent_bits) - 1);
    float max_exp = static_cast<float>(max_stored_exp - bias);
    // Normals start at unbiased exponent (1 - bias); [-bias, 1-bias) is subnormal.
    // Pure exponent formats (M0) have no subnormals; exp 0 encodes 2^(-bias).
    float min_normal_exp = (mantissa_bits == 0) ? static_cast<float>(-bias) : static_cast<float>(1 - bias);
    float subnormal_scale = std::exp2(static_cast<float>(1 - bias));
    int mantissa_upper_bound = 1 << mantissa_bits;
    int max_mant_at_max = max_mantissa_at_max_exponent;
    float mantissa_scale = static_cast<float>(std::max(mantissa_upper_bound, 1));
    float inv_mantissa_scale = 1.0f / mantissa_scale;

    float* input_ptr = input.data_ptr<float>();

    #pragma omp parallel for
    for (int idx = 0; idx < numel; ++idx) {
        float x_val = input_ptr[idx];
        if (x_val == 0.0f) continue;

        const float s = std::copysign(1.0f, x_val);
        const float x_abs = std::fabs(x_val);
        const float exponent_floor = std::floor(std::log2(x_abs));

        if (mantissa_bits > 0 && exponent_floor < min_normal_exp) {
            // Subnormal: value = (mant / 2^m) * 2^(1-bias)
            const float mant_unrounded = (x_abs / subnormal_scale) * mantissa_scale;
            int mant = round_ties_to_even(mant_unrounded);
            if (mant <= 0) {
                input_ptr[idx] = s * 0.0f;
            } else if (mant >= mantissa_upper_bound) {
                // Overflow into smallest normal: 1.0 * 2^(1-bias)
                input_ptr[idx] = s * subnormal_scale;
            } else {
                input_ptr[idx] = s * (static_cast<float>(mant) * inv_mantissa_scale) * subnormal_scale;
            }
            continue;
        }

        float exponent = std::fmax(std::fmin(exponent_floor, max_exp), min_normal_exp);
        float exp2_val = std::exp2(exponent);

        float scaled = x_abs / exp2_val;
        scaled = std::fmax(scaled, 1.0f);

        const float mantissa_unrounded = (scaled - 1.0f) * mantissa_scale;
        int mantissa = round_ties_to_even(mantissa_unrounded);

        const bool at_max_exp = exponent >= max_exp;
        const int effective_ub = at_max_exp ? (max_mant_at_max + 1) : mantissa_upper_bound;

        float final_exp2 = exp2_val;
        int final_mantissa = mantissa;
        if (mantissa >= effective_ub) {
            if (at_max_exp) {
                // Saturate to largest finite at max exponent (E4M3-FN: 448, not 480)
                final_mantissa = max_mant_at_max;
            } else {
                const float exponent_overflow = std::fmax(std::fmin(exponent + 1.0f, max_exp), min_normal_exp);
                final_exp2 = std::exp2(exponent_overflow);
                final_mantissa = 0;
            }
        } else if (mantissa < 0) {
            final_mantissa = 0;
        }

        const float fraction = static_cast<float>(final_mantissa) * inv_mantissa_scale;
        input_ptr[idx] = s * (1.0f + fraction) * final_exp2;
    }

    return input;
}

torch::Tensor float_round_inplace(
    torch::Tensor input,
    int exponent_bits,
    int mantissa_bits,
    int bias,
    int reserved_exponent,
    int max_mantissa_at_max_exponent) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    if (input.device().is_cuda()) {
        #ifdef WITH_CUDA
        return float_round_cuda_inplace(
            input, exponent_bits, mantissa_bits, bias, reserved_exponent, max_mantissa_at_max_exponent);
        #else
        TORCH_CHECK(false, "CUDA support not available");
        #endif
    } else {
        return float_round_cpu_inplace(
            input, exponent_bits, mantissa_bits, bias, reserved_exponent, max_mantissa_at_max_exponent);
    }
}

// C++ function for floating point rounding
torch::Tensor float_round(
    torch::Tensor input,
    int exponent_bits,
    int mantissa_bits,
    int bias,
    int reserved_exponent,
    int max_mantissa_at_max_exponent) {
    return float_round_inplace(
        input.clone(), exponent_bits, mantissa_bits, bias, reserved_exponent, max_mantissa_at_max_exponent);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "inplace",
        &float_round_inplace,
        "Float rounding operation (CUDA/CPU, inplace)",
        py::arg("input"),
        py::arg("exponent_bits"),
        py::arg("mantissa_bits"),
        py::arg("bias"),
        py::arg("reserved_exponent"),
        py::arg("max_mantissa_at_max_exponent"));
    m.def(
        "round",
        &float_round,
        "Float rounding operation (CUDA/CPU, non-inplace)",
        py::arg("input"),
        py::arg("exponent_bits"),
        py::arg("mantissa_bits"),
        py::arg("bias"),
        py::arg("reserved_exponent"),
        py::arg("max_mantissa_at_max_exponent"));
}
