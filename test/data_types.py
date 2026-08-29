import math
import unittest
import warnings

import numpy as np
from parameterized import parameterized
from torch import arange, bfloat16, finfo, float16, tensor, uint16

from floating_point.data_types import FloatingPoint
from test.nvidia_codec_goldens import E2M1_GOLDEN, E4M3_GOLDEN, E5M2_GOLDEN, E8M0_GOLDEN


def compare_values(expected_values, actual_values, tolerance_rtol=1e-1, tolerance_atol=1e-1):
    expected_values_as_tensor = tensor(expected_values)
    values = tensor(actual_values)
    values = values[~(values.isnan() | values.isinf())]
    difference_matrix = (expected_values_as_tensor.unsqueeze(1) - values).abs()
    min_diff, min_indices = difference_matrix.min(dim=1)
    values_diff_values = values[min_indices]
    nonzero_mask = min_diff != 0.0
    expected_values_filtered = expected_values_as_tensor[nonzero_mask]
    values_filtered = values_diff_values[nonzero_mask]
    if len(expected_values_filtered) > 0:
        warnings.warn(f"PyTorch: {expected_values_filtered.tolist()} != \nSimulated: {values_filtered.tolist()}")
    np.testing.assert_allclose(min_diff.sum().numpy(), 0.0, rtol=tolerance_rtol, atol=tolerance_atol)


def assert_codebook_matches(fp: FloatingPoint, golden):
    """Per-code compare against NVIDIA CUDA __nv_fp* goldens."""
    assert len(golden) == 2**fp.bits
    for code, expected in enumerate(golden):
        got = fp.bit_pattern_to_custom_fp(code)
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(got), f"code {code}: expected NaN, got {got}"
        elif isinstance(expected, float) and math.isinf(expected):
            assert math.isinf(got) and math.copysign(1.0, got) == math.copysign(1.0, expected), (
                f"code {code}: expected {expected}, got {got}")
        else:
            assert got == float(expected), f"code {code}: expected {expected}, got {got}"


FLOAT4E1M2F0_VALUES = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
FLOAT4E2M1F1_VALUES = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FLOAT4E3M0F3_VALUES = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


class TestFloatingPoint4Bits(unittest.TestCase):
    __float4e1m2f0__ = FloatingPoint(1, 1, 2, 0, 4, reserved_exponent=False)
    __float4e2m1f1__ = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
    __float4e3m0f3__ = FloatingPoint(1, 3, 0, 3, 4, reserved_exponent=False)

    @parameterized.expand([
        ("float4e1m2f0", __float4e1m2f0__, 0.25, FLOAT4E1M2F0_VALUES),
        ("float4e2m1f1", __float4e2m1f1__, 0.5, FLOAT4E2M1F1_VALUES),
        ("float4e3m0f3", __float4e3m0f3__, 1.0, FLOAT4E3M0F3_VALUES),
    ])
    def test_epsilon_and_values(self, name, fp, expected_epsilon, expected_values):
        self.assertEqual(fp.epsilon, expected_epsilon)
        self.assertEqual(fp.values, expected_values)


class TestNvidiaCodecGoldens(unittest.TestCase):
    """Per-code fidelity vs CUDA __nv_fp* goldens (committed codec tables)."""

    e2m1 = FloatingPoint(sign_bits=1, exponent_bits=2, mantissa_bits=1, bias=1, bits=4, reserved_exponent=False)
    e4m3fn = FloatingPoint(
        sign_bits=1, exponent_bits=4, mantissa_bits=3, bias=7, bits=8,
        max_mantissa_at_max_exponent=6, reserved_exponent=False)
    e5m2 = FloatingPoint(sign_bits=1, exponent_bits=5, mantissa_bits=2, bias=15, bits=8, reserved_exponent=True)
    e8m0 = FloatingPoint(sign_bits=0, exponent_bits=8, mantissa_bits=0, bias=127, bits=8, reserved_exponent=True)

    def test_e2m1_all_codes(self):
        assert_codebook_matches(self.e2m1, E2M1_GOLDEN)

    def test_e4m3fn_all_codes(self):
        assert_codebook_matches(self.e4m3fn, E4M3_GOLDEN)
        self.assertTrue(math.isnan(self.e4m3fn.bit_pattern_to_custom_fp(127)))
        self.assertTrue(math.isnan(self.e4m3fn.bit_pattern_to_custom_fp(255)))
        self.assertEqual(self.e4m3fn.maximum, 448.0)
        self.assertEqual(self.e4m3fn.bit_pattern_to_custom_fp(126), 448.0)
        finites = [v for v in self.e4m3fn.values if math.isfinite(v)]
        self.assertNotIn(480.0, finites)
        self.assertNotIn(-480.0, finites)

    def test_e5m2_all_codes(self):
        assert_codebook_matches(self.e5m2, E5M2_GOLDEN)

    def test_e8m0_all_codes(self):
        assert_codebook_matches(self.e8m0, E8M0_GOLDEN)
        self.assertEqual(self.e8m0.bit_pattern_to_custom_fp(0), 2**-127)
        self.assertEqual(self.e8m0.bit_pattern_to_custom_fp(127), 1.0)
        self.assertEqual(self.e8m0.bit_pattern_to_custom_fp(128), 2.0)
        self.assertTrue(math.isnan(self.e8m0.bit_pattern_to_custom_fp(255)))
        self.assertEqual(self.e8m0.minimum, 2**-127)


def generate_all_torch_fp16_values(dtype):
    uint16_tensor = arange(0, 2**16).to(dtype=uint16)
    float16_values = uint16_tensor.view(dtype)
    mask = float16_values.isnan() | float16_values.isinf()
    float16_values = float16_values[~mask]
    return float16_values.tolist()


@unittest.skip("Skipping FP16 tests to speed up CI runs")
class TestFloatingPoint16Bits(unittest.TestCase):
    __float16__ = FloatingPoint(1, 5, 10, 15, 16)
    __bfloat16__ = FloatingPoint(1, 8, 7, 127, 16)

    @parameterized.expand([
        ("float16", __float16__, finfo(float16).eps, generate_all_torch_fp16_values(float16)),
        ("bfloat16", __bfloat16__, finfo(bfloat16).eps, generate_all_torch_fp16_values(bfloat16)),
    ])
    def test_values(self, name, fp, eps, expected_values):
        self.assertEqual(fp.epsilon, eps)
        compare_values(expected_values, fp.values, tolerance_rtol=1e-3, tolerance_atol=1e-3)


if __name__ == "__main__":
    unittest.main()
