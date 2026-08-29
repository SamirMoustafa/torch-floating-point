import unittest

import torch
from parameterized import parameterized
from torch import nn

from floating_point import BlockFormat, BlockRound, FloatingPoint, Round, block_round, sample_block_scaled
from floating_point.block_round import encode_scale

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

E2M1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
E4M3 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
UE8M0 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)
# Hardware packings as fixtures — not package exports.
NVFP4 = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest")
MXFP4 = BlockFormat(E2M1, UE8M0, 32, 6.0, "ue8m0_ceil")
MXFP8 = BlockFormat(E4M3, UE8M0, 32, 448.0, "ue8m0_ceil")
HOPPER_1D = BlockFormat(E4M3, UE8M0, 128, 448.0, "amax_over_M")


class ConstThreeRound(Round):
    """Constant map so reconstruct must be ``y = 3 * s``, not raw ``x``."""

    def forward(self, x):
        return torch.full_like(x, 3.0)


class Int4Round(Round):
    """Q4_0 element map: integers in [-8, 7]."""

    def forward(self, x):
        return x.round().clamp(-8.0, 7.0)


class TestBlockRoundNVFP4(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_sampler_absmax_roundtrip(self, device):
        torch.manual_seed(0)
        x = sample_block_scaled((4, 64), NVFP4, device=torch.device(device))
        y = block_round(x, NVFP4)
        self.assertTrue(torch.allclose(x, y, rtol=0, atol=0))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_plain_e2m1_grid_has_drift(self, device):
        codes = torch.tensor(
            [v for v in NVFP4.elem_fp.values if abs(v) <= 6], dtype=torch.float32, device=device)
        idx = torch.randint(0, len(codes), (8, 64), device=device)
        x = codes[idx]
        y = block_round(x, NVFP4)
        self.assertGreater(float((x - y).abs().max()), 0.0)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_ste_absmax_grad_x_only(self, device):
        x = torch.randn(2, 32, device=device, requires_grad=True)
        y, scales, elems = block_round(x, NVFP4, return_aux=True)
        self.assertTrue(y.requires_grad)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse(scales.requires_grad)
        self.assertEqual(tuple(scales.shape), (2, 2, 1))
        self.assertEqual(tuple(elems.shape), tuple(x.shape))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_learnable_scales_get_grad(self, device):
        x = torch.randn(2, 32, device=device, requires_grad=True)
        s = nn.Parameter(torch.ones(2, 2, 1, device=device))
        y = block_round(x, NVFP4, scales=s)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad.abs().sum()), 0.0)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_custom_rounder_is_used(self, device):
        x = torch.randn(2, 32, device=device)
        y, s, elems = block_round(x, NVFP4, rounder=ConstThreeRound, return_aux=True)
        self.assertTrue(torch.equal(elems, torch.full_like(elems, 3.0)))
        k = 16
        s_exp = s.expand(*s.shape[:-1], k).reshape_as(y)
        self.assertTrue(torch.allclose(y, 3.0 * s_exp, rtol=0, atol=0))
        y2, s2, e2 = BlockRound(NVFP4, rounder=ConstThreeRound)(x, return_aux=True)
        self.assertTrue(torch.equal(e2, torch.full_like(e2, 3.0)))
        self.assertTrue(torch.allclose(y2, 3.0 * s2.expand(*s2.shape[:-1], k).reshape_as(y2), rtol=0, atol=0))

    def test_bad_block_size_raises(self):
        x = torch.randn(3, 10)
        with self.assertRaises(ValueError):
            block_round(x, NVFP4)


class TestBlockRoundMXFP4(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_sampler_absmax_roundtrip(self, device):
        torch.manual_seed(2)
        x = sample_block_scaled((2, 64), MXFP4, device=torch.device(device))
        y = block_round(x, MXFP4)
        self.assertTrue(torch.allclose(x, y, rtol=0, atol=0))


class TestBlockRoundMXFP8(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_sampler_absmax_roundtrip(self, device):
        torch.manual_seed(1)
        x = sample_block_scaled((2, 64), MXFP8, device=torch.device(device))
        y = BlockRound(MXFP8)(x)
        self.assertTrue(torch.allclose(x, y, rtol=0, atol=0))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_ue8m0_round_up_covers_amax(self, device):
        amax = torch.tensor([[1.5 * 448.0]], dtype=torch.float32, device=device)
        s = encode_scale(amax, MXFP8)
        self.assertEqual(float(s), 2.0)
        self.assertLessEqual(672.0, 2.0 * 448.0)


class TestBlockRoundClass(unittest.TestCase):
    def test_holds_spec(self):
        br = BlockRound(NVFP4)
        self.assertIs(br.spec, NVFP4)
        self.assertEqual(br.spec.M, 6.0)
        self.assertEqual(br.spec.block_size, 16)
        self.assertEqual(br.spec.dims, (-1,))
        self.assertEqual(br.spec.s_global, 1.0)
        self.assertEqual(br.spec.zero_point, 0.0)

    def test_bad_scale_encode_raises(self):
        with self.assertRaises(ValueError):
            BlockFormat(E2M1, E4M3, 16, 6.0, "sniff")

    def test_2d_default_dims(self):
        spec = BlockFormat(E2M1, E4M3, (16, 16), 6.0, "nearest")
        self.assertEqual(spec.dims, (-2, -1))
        self.assertEqual(spec.block_size, (16, 16))


class TestEncodePolicies(unittest.TestCase):
    """OCP sample vs NVIDIA ceil vs AWS 2x, from the linked formulas."""

    def test_emax_and_numeric_table(self):
        # amax, ocp_floor, ocp_floor_x2, ue8m0_ceil, ue8m0_floor, amax_over_M
        rows = [
            (256.0, 1.0, 2.0, 1.0, 0.5, 256.0 / 448.0),
            (448.0, 1.0, 2.0, 1.0, 1.0, 1.0),
            (500.0, 1.0, 2.0, 2.0, 1.0, 500.0 / 448.0),
            (672.0, 2.0, 4.0, 2.0, 1.0, 1.5),
            (1024.0, 4.0, 8.0, 4.0, 2.0, 1024.0 / 448.0)]
        floor_s = BlockFormat(E4M3, UE8M0, 32, 448.0, "ocp_floor")
        x2_s = BlockFormat(E4M3, UE8M0, 32, 448.0, "ocp_floor_x2")
        ceil_s = BlockFormat(E4M3, UE8M0, 32, 448.0, "ue8m0_ceil")
        ufloor_s = BlockFormat(E4M3, UE8M0, 32, 448.0, "ue8m0_floor")
        amax_s = BlockFormat(E4M3, UE8M0, 32, 448.0, "amax_over_M")
        for amax, ocp, aws, nceil, nfloor, raw in rows:
            stat = torch.tensor([[amax]], dtype=torch.float32)
            self.assertEqual(float(encode_scale(stat, floor_s)), ocp, msg=f"ocp amax={amax}")
            self.assertEqual(float(encode_scale(stat, x2_s)), aws, msg=f"aws amax={amax}")
            self.assertEqual(float(encode_scale(stat, ceil_s)), nceil, msg=f"ceil amax={amax}")
            self.assertEqual(float(encode_scale(stat, ufloor_s)), nfloor, msg=f"ufloor amax={amax}")
            self.assertAlmostEqual(float(encode_scale(stat, amax_s)), raw, places=5, msg=f"amax/M amax={amax}")

    @parameterized.expand([(d,) for d in DEVICES])
    def test_hopper_amax_over_m(self, device):
        x = torch.zeros(1, 128, device=device)
        x[0, 0] = 672.0
        y, s, _ = block_round(x, HOPPER_1D, return_aux=True)
        self.assertAlmostEqual(float(s.reshape(-1)[0]), 1.5, places=5)
        self.assertAlmostEqual(float(y[0, 0]), 448.0 * 1.5, places=3)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_pad_tensor_s_global_no_nan(self, device):
        spec = BlockFormat(E4M3, UE8M0, 4, 448.0, "amax_over_M", pad="zero")
        x = torch.zeros(1, 6, device=device)
        x[0, 0] = 448.0
        x[0, 4] = 224.0
        y, s, _ = block_round(x, spec, s_global=torch.ones_like(x), return_aux=True)
        self.assertTrue(torch.allclose(s.reshape(-1).cpu(), torch.tensor([1.0, 0.5]), rtol=0, atol=0))
        self.assertTrue(torch.isfinite(y).all())

    @parameterized.expand([(d,) for d in DEVICES])
    def test_s_global_in_absmax(self, device):
        x = torch.zeros(1, 128, device=device)
        x[0, 0] = 896.0
        y, s, e = block_round(x, HOPPER_1D, s_global=2.0, return_aux=True)
        self.assertAlmostEqual(float(s.reshape(-1)[0]), 1.0, places=5)
        self.assertAlmostEqual(float(e[0, 0]), 448.0, places=5)
        self.assertAlmostEqual(float(y[0, 0]), 448.0 * 1.0 * 2.0, places=3)

    def test_s_global_zero_raises(self):
        x = torch.randn(2, 32)
        with self.assertRaises(ValueError):
            block_round(x, NVFP4, s_global=0.0)


class TestTwoD(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_row_tile_matches_last_dim(self, device):
        spec_1d = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest")
        spec_2d = BlockFormat(E2M1, E4M3, (1, 16), 6.0, "nearest")
        x = torch.randn(4, 32, device=device)
        y1 = block_round(x, spec_1d)
        y2 = block_round(x, spec_2d)
        self.assertTrue(torch.allclose(y1, y2, rtol=0, atol=0))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_column_dims(self, device):
        along_rows = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest", dims=(-2,))
        along_last = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest")
        x = torch.randn(32, 8, device=device)
        y = block_round(x, along_rows)
        self.assertEqual(tuple(y.shape), (32, 8))
        with self.assertRaises(ValueError):
            block_round(x, along_last)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_real_2d_tiles(self, device):
        m = 2.0
        spec = BlockFormat(E4M3, UE8M0, (2, 2), m, "amax_over_M")
        x = torch.zeros(4, 4, device=device)
        x[0, 0] = 10.0
        x[0, 2] = 8.0
        x[2, 0] = 4.0
        x[2, 2] = 2.0
        _, s, _ = block_round(x, spec, return_aux=True)
        want = torch.tensor([[10.0 / m, 8.0 / m], [4.0 / m, 2.0 / m]], device=device)
        self.assertTrue(torch.allclose(s.squeeze(), want, rtol=0, atol=1e-6))


class TestAffineInt4(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_zero_point_with_known_scales(self, device):
        spec = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest", zero_point=1.0)
        e = torch.zeros(1, 16, device=device)
        e[0, 0] = 2.0
        s = torch.ones(1, 1, 1, device=device) * 2.0
        x = (e - 1.0) * 2.0
        y = block_round(x, spec, scales=s)
        self.assertTrue(torch.allclose(x, y, rtol=0, atol=1e-6))

    def test_absmax_zero_point_requires_scales(self):
        spec = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest", zero_point=1.0)
        x = torch.zeros(1, 16)
        with self.assertRaises(ValueError):
            block_round(x, spec)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_signed_peak_q4_0(self, device):
        spec = BlockFormat(E2M1, E4M3, 32, 8.0, "signed_peak")
        x = torch.zeros(1, 32, device=device)
        x[0, 0] = -4.0
        x[0, 1] = 2.0
        y, s, _ = block_round(x, spec, rounder=Int4Round, return_aux=True)
        self.assertAlmostEqual(float(s.reshape(-1)[0]), 0.5, places=5)
        self.assertAlmostEqual(float(y[0, 0]), -4.0, places=5)
        self.assertAlmostEqual(float(y[0, 1]), 2.0, places=5)


class TestSamplerContract(unittest.TestCase):
    def test_rejects_zero_point(self):
        spec = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest", zero_point=1.0)
        with self.assertRaises(ValueError):
            sample_block_scaled((4, 64), spec)

    def test_rejects_ocp_floor_x2(self):
        spec = BlockFormat(E4M3, UE8M0, 32, 448.0, "ocp_floor_x2")
        with self.assertRaises(ValueError):
            sample_block_scaled((2, 32), spec)

    def test_rejects_signed_peak(self):
        spec = BlockFormat(E2M1, E4M3, 32, 8.0, "signed_peak")
        with self.assertRaises(ValueError):
            sample_block_scaled((1, 32), spec)


if __name__ == "__main__":
    unittest.main()
