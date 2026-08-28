import math
import unittest

import torch
from parameterized import parameterized
from torch import FloatTensor, bfloat16, finfo, float8_e4m3fn, float8_e5m2, float16, randn

from floating_point import round
from floating_point.data_types import FloatingPoint
from floating_point.round import Round


class TestRoundFunctionDifferentiability(unittest.TestCase):
    @parameterized.expand([
        ("fp8e5m2", FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True), "cpu"),
        ("fp8e4m3", FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False), "cpu"),
    ] + ([("fp8e5m2", FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True), "cuda"),
          ("fp8e4m3", FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False), "cuda")]
    if torch.cuda.is_available() else []))
    def test_round_differentiability(self, name, data_type, device):
        rounder = Round(data_type)
        x = randn(100, requires_grad=True, device=device)
        x[x < data_type.minimum].fill_(data_type.minimum)
        x[x > data_type.maximum].fill_(data_type.maximum)
        x_val = x.clone().detach().requires_grad_(True)
        z = rounder(x).sum()
        z.backward()
        x_grad_round = x.grad.clone().detach()
        self.assertIsNotNone(x_grad_round, "Gradient is None, the function is not differentiable.")
        self.assertEqual(x_grad_round.shape, x.shape, "Gradient shape incorrect.")
        z = x_val.sum()
        z.backward()
        x_grad_val = x_val.grad.clone().detach()
        self.assertTrue(torch.allclose(x_grad_round, x_grad_val, rtol=1e-5, atol=1e-5), "Gradient mismatch.")

    @parameterized.expand([(d,) for d in (["cpu"] + (["cuda"] if torch.cuda.is_available() else []))])
    def test_clipped_ste_zeros_outside_range(self, device):
        fp = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
        rounder = Round(fp)
        x = torch.tensor([-10.0, -6.0, 0.0, 6.0, 10.0], device=device, requires_grad=True)
        rounder(x).sum().backward()
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0], device=device)
        self.assertTrue(torch.equal(x.grad, expected))

    @parameterized.expand([(d,) for d in (["cpu"] + (["cuda"] if torch.cuda.is_available() else []))])
    def test_ste_does_not_clip_grad_magnitude(self, device):
        fp = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
        rounder = Round(fp)
        x = torch.tensor([1.0], device=device, requires_grad=True)
        rounder(x).backward(torch.tensor([100.0], device=device))
        self.assertEqual(float(x.grad), 100.0)


class TestFloatingPointRounding(unittest.TestCase):
    __float8e5m2__ = FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True)
    __float8e4m3fn__ = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
    __float16__ = FloatingPoint(1, 5, 10, 15, 16)
    __bfloat16__ = FloatingPoint(1, 8, 7, 127, 16)
    __float32__ = FloatingPoint(1, 8, 23, 127, 32)

    @parameterized.expand([
        ("float8e5m2", __float8e5m2__, float8_e5m2, "cpu"),
        ("float8e4m3fn", __float8e4m3fn__, float8_e4m3fn, "cpu"),
        ("float16", __float16__, float16, "cpu"),
        ("bfloat16", __bfloat16__, bfloat16, "cpu"),
    ] + ([("float8e5m2", __float8e5m2__, float8_e5m2, "cuda"),
          ("float8e4m3fn", __float8e4m3fn__, float8_e4m3fn, "cuda"),
          ("float16", __float16__, float16, "cuda"),
          ("bfloat16", __bfloat16__, bfloat16, "cuda")]
    if torch.cuda.is_available() else []))
    def test_rounding(self, name, fp, dtype, device):
        a, b = finfo(dtype).min, finfo(dtype).max
        assert a == fp.minimum and b == fp.maximum
        a = -3e37 if a < -3e37 else a
        b = 3e37 if b > 3e37 else b
        x = FloatTensor(100).uniform_(a, b).clamp(min=a, max=b).to(device=device)
        quantized_x = round(
            x,
            fp.exponent_bits,
            fp.mantissa_bits,
            fp.bias,
            reserved_exponent=fp.reserved_exponent,
            max_mantissa_at_max_exponent=fp.max_mantissa_at_max_exponent,
        )
        torch_rounded_x = x.to(dtype).float()
        l1_error = (quantized_x - torch_rounded_x).abs().sum().item()
        self.assertTrue(l1_error == 0.0, f"Rounding mismatch in {l1_error}, for {name} ({dtype}) on {device}.")


class TestE4M3FNRoundSaturation(unittest.TestCase):
    @parameterized.expand(
        [("cpu",)] + ([("cuda",)] if torch.cuda.is_available() else [])
    )
    def test_round_does_not_emit_480(self, device):
        fp = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
        rounder = Round(fp)
        x = torch.tensor([448.0, 449.0, 480.0, 1000.0, -448.0, -480.0], dtype=torch.float32, device=device)
        y = rounder(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(torch.any(y.abs() == 480.0))
        self.assertEqual(float(y.max()), 448.0)
        self.assertEqual(float(y.min()), -448.0)
        self.assertFalse(math.isnan(float(y[0].cpu())))


class TestE2M1RoundCodebook(unittest.TestCase):
    """Issue #7: Round must land only on the FloatingPoint codebook (no fake 0.75)."""

    e2m1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)

    @parameterized.expand(
        [("cpu",)] + ([("cuda",)] if torch.cuda.is_available() else [])
    )
    def test_0_7_rounds_to_0_5(self, device):
        rounder = Round(self.e2m1)
        y = rounder(torch.tensor([0.7], dtype=torch.float32, device=device))
        self.assertEqual(float(y[0].cpu()), 0.5)

    @parameterized.expand(
        [("cpu",)] + ([("cuda",)] if torch.cuda.is_available() else [])
    )
    def test_midpoints_ties_to_even(self, device):
        # NVIDIA __nv_fp4_e2m1: 0.25 → 0, 0.75 → 1
        rounder = Round(self.e2m1)
        x = torch.tensor([0.25, 0.75, -0.25, -0.75], dtype=torch.float32, device=device)
        y = rounder(x).cpu().tolist()
        self.assertEqual(y, [0.0, 1.0, -0.0, -1.0])

    @parameterized.expand(
        [("cpu",)] + ([("cuda",)] if torch.cuda.is_available() else [])
    )
    def test_dense_outputs_in_codebook(self, device):
        rounder = Round(self.e2m1)
        codebook = set(self.e2m1.values)
        x = torch.linspace(-7.0, 7.0, 14001, dtype=torch.float32, device=device)
        y = rounder(x).cpu()
        unique = {float(v) for v in y.unique()}
        self.assertNotIn(0.75, unique)
        self.assertNotIn(-0.75, unique)
        illegal = unique - codebook
        self.assertEqual(illegal, set(), f"Illegal Round outputs: {sorted(illegal)}")


class TestE4M3FNSubnormalCodebook(unittest.TestCase):
    """Same class of bug: no invented values in the subnormal binade."""

    e4m3 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)

    @parameterized.expand(
        [("cpu",)] + ([("cuda",)] if torch.cuda.is_available() else [])
    )
    def test_subnormal_range_outputs_in_codebook(self, device):
        rounder = Round(self.e4m3)
        codebook = set(self.e4m3.values)
        # Below min normal 2^(1-bias) = 2^-6 ≈ 0.015625
        x = torch.linspace(-0.03, 0.03, 6001, dtype=torch.float32, device=device)
        y = rounder(x).cpu()
        unique = {float(v) for v in y.unique()}
        illegal = unique - codebook
        self.assertEqual(illegal, set(), f"Illegal Round outputs: {sorted(illegal)}")
        self.assertNotIn(0.0087890625, unique)


if __name__ == "__main__":
    unittest.main()
