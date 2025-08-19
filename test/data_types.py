import math
import unittest
import warnings

import numpy as np
from parameterized import parameterized
from torch import (
    abs,
    arange,
    bfloat16,
    finfo,
    float8_e4m3fn,
    float8_e5m2,
    float16,
    from_numpy,
    tensor,
    testing,
    uint8,
    uint16,
)

from floating_point.data_types import FloatingPoint

FLOAT4E1M2F0_VALUES = [
    -3.5,
    -3.0,
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
]
FLOAT4E2M1F1_VALUES = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
FLOAT4E3M0F3_VALUES = [
    -16.0,
    -8.0,
    -4.0,
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    -0.0,
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
]


class TestFloatingPoint4Bits(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "float4e1m2f0",
                FloatingPoint(1, 1, 2, 0, 4, reserved_exponent=False),
                0.25,
                FLOAT4E1M2F0_VALUES,
            ),
            (
                "float4e2m1f1",
                FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False),
                0.5,
                FLOAT4E2M1F1_VALUES,
            ),
            (
                "float4e3m0f3",
                FloatingPoint(1, 3, 0, 3, 4, reserved_exponent=False),
                1.0,
                FLOAT4E3M0F3_VALUES,
            ),
        ]
    )
    def test_epsilon_and_values(self, name, fp, expected_epsilon, expected_values):
        self.assertEqual(fp.epsilon, expected_epsilon)
        self.assertEqual(fp.values, expected_values)


def eight_bits_to_torch_dtype(bit_pattern, dtype):
    if not (0 <= bit_pattern):
        raise ValueError("Bit pattern must be non-negative.")
    np_bits = np.array([bit_pattern], dtype=np.uint8)
    uint8_tensor = from_numpy(np_bits).type(uint8)
    float8_tensor = uint8_tensor.view(dtype)
    float_value = float(float8_tensor.item())
    return float_value


def generate_all_torch_fp8_values(dtype):
    float8_values = []
    for i in range(2**8):
        val = eight_bits_to_torch_dtype(i, dtype)
        float8_values += [val]
    return float8_values


FLOAT8E5M2_VALUES = generate_all_torch_fp8_values(float8_e5m2)
FLOAT8E5M2_VALUES = sorted(
    [
        *filter(
            lambda x: isinstance(x, (int, float)) and math.isfinite(x),
            FLOAT8E5M2_VALUES,
        )
    ]
)

FLOAT8E4M3FN_VALUES = generate_all_torch_fp8_values(float8_e4m3fn)
FLOAT8E4M3FN_VALUES = sorted(
    [
        *filter(
            lambda x: isinstance(x, (int, float)) and math.isfinite(x),
            FLOAT8E4M3FN_VALUES,
        )
    ]
)


class TestFloatingPoint8Bits(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "float8e5m2",
                FloatingPoint(1, 5, 2, 16, 8, reserved_exponent=False),
                finfo(float8_e5m2).eps,
                FLOAT8E5M2_VALUES,
            ),
            (
                "float8e4m3fn",
                FloatingPoint(
                    1,
                    4,
                    3,
                    7,
                    8,
                    max_mantissa_at_max_exponent=6,
                    reserved_exponent=False,
                ),
                finfo(float8_e4m3fn).eps,
                FLOAT8E4M3FN_VALUES,
            ),
        ]
    )
    def test_values(self, name, fp, eps, expected_values):
        self.assertEqual(fp.epsilon, eps)
        expected_values_as_tensor = tensor(expected_values)
        values = tensor(fp.values)
        difference_matrix = abs(expected_values_as_tensor.unsqueeze(1) - values)
        min_diff, min_indices = difference_matrix.min(dim=1)
        values_diff_values = values[min_indices]
        nonzero_mask = min_diff != 0.0
        expected_values_filtered = expected_values_as_tensor[nonzero_mask]
        values_filtered = values_diff_values[nonzero_mask]
        if len(expected_values_filtered) > 0:
            warnings.warn(
                f"Numerical differences found in {name}: \n"
                f"PyTorch FP8: {expected_values_filtered.tolist()} != \n"
                f"Simulated FP8: {values_filtered.tolist()}",
                stacklevel=2,
            )
        testing.assert_close(min_diff.sum(), tensor(0.0), rtol=1e-1, atol=1e-1)


# For GitHub Actions, we need to comment out the following code to fasten the tests
def generate_all_torch_fp16_values(dtype):
    uint16_tensor = arange(0, 2**16).to(dtype=uint16)
    float16_values = uint16_tensor.view(dtype)
    mask = float16_values.isnan() | float16_values.isinf()
    float16_values = float16_values[~mask]
    return float16_values.tolist()


@unittest.skip("Skipping FP16 tests to speed up CI runs")
class TestFloatingPoint16Bits(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "float16",
                FloatingPoint(1, 5, 10, 15, 16),
                finfo(float16).eps,
                generate_all_torch_fp16_values(float16),
            ),
            (
                "bfloat16",
                FloatingPoint(1, 8, 7, 127, 16),
                finfo(bfloat16).eps,
                generate_all_torch_fp16_values(bfloat16),
            ),
        ]
    )
    def test_values(self, name, fp, eps, expected_values):
        self.assertEqual(fp.epsilon, eps)
        expected_values_as_tensor = tensor(expected_values)
        values = tensor(fp.values)
        values = values[~(values.isnan() | values.isinf())]
        difference_matrix = abs(expected_values_as_tensor.unsqueeze(1) - values)
        min_diff, min_indices = difference_matrix.min(dim=1)
        values_diff_values = values[min_indices]
        nonzero_mask = min_diff != 0.0
        expected_values_filtered = expected_values_as_tensor[nonzero_mask]
        values_filtered = values_diff_values[nonzero_mask]
        if len(expected_values_filtered) > 0:
            warnings.warn(
                f"Numerical differences found in {name}: \n"
                f"PyTorch FP16: {expected_values_filtered.tolist()} != \n"
                f"Simulated FP16: {values_filtered.tolist()}",
                stacklevel=2,
            )
        testing.assert_close(min_diff.sum(), tensor(0.0), rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
