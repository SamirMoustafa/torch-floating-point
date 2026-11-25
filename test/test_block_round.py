import math
import unittest

import torch
from parameterized import parameterized
from torch import nn

from floating_point import BlockFormat, BlockRound, FloatingPoint, Round, block_round, tensor_scale
from floating_point.block_round import _SCALE_ENCODES, encode_scale

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

E2M1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
E4M3 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
UE8M0 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)
UE4M3 = FloatingPoint(0, 4, 3, 7, 7, reserved_exponent=False)
UE2M2 = FloatingPoint(0, 2, 2, 3, 4, reserved_exponent=False, subnormals=False)  # issue #40: no subnormal row
# Hardware packings as fixtures — not package exports.
NVFP4 = BlockFormat(E2M1, E4M3, 16, 6.0, "nearest")
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

    @parameterized.expand([(d,) for d in DEVICES])
    def test_tensor_scale_covers_amax_beyond_elem_times_scale(self, device):
        peak = NVFP4.M * float(NVFP4.scale_fp.maximum)
        x = torch.zeros(1, 32, device=device)
        x[0, 0] = 10000.0
        sg = tensor_scale(x, NVFP4)
        self.assertAlmostEqual(float(sg), 10000.0 / peak, places=5)
        y, s, e = block_round(x, NVFP4, s_global=sg, return_aux=True)
        compress = peak / 10000.0
        s_from_compress = encode_scale((x * compress).abs().amax().reshape(1, 1), NVFP4)
        self.assertAlmostEqual(float(s.reshape(-1)[0]), float(s_from_compress), places=5)
        self.assertAlmostEqual(float(y[0, 0]), 10000.0, places=3)
        self.assertAlmostEqual(float(e[0, 0]), 6.0, places=5)
        y1 = block_round(x, NVFP4)
        self.assertAlmostEqual(float(y1[0, 0]), peak, places=3)

    def test_tensor_scale_zero_is_one(self):
        x = torch.zeros(1, 16)
        self.assertEqual(float(tensor_scale(x, NVFP4)), 1.0)


class TestBlockRoundMXFP8(unittest.TestCase):
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


class TestCeilScaleEncode(unittest.TestCase):
    """Issue #39: ``ceil`` is the smallest ``scale_fp`` code >= amax / M."""

    ceil_spec = BlockFormat(E2M1, UE4M3, 16, 6.0, "ceil")
    nearest_spec = BlockFormat(E2M1, UE4M3, 16, 6.0, "nearest")
    codes = sorted(v for v in UE4M3.values if v > 0)

    @staticmethod
    def blocks(device, lo_exp, hi_exp):
        g = torch.Generator().manual_seed(0)
        mag = torch.exp2(torch.randint(lo_exp, hi_exp + 1, (256, 1), generator=g).float())
        return (torch.randn(256, 64, generator=g) * mag).to(device)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_issue_example(self, device):
        x = torch.tensor([[1.0, -2.0, 3.0, 5.7] * 4], device=device)
        y, s, _ = block_round(x, self.nearest_spec, return_aux=True)
        self.assertEqual(float(s), 0.9375)
        self.assertEqual(float(y[0, 3]), 5.625)
        y, s, _ = block_round(x, self.ceil_spec, return_aux=True)
        self.assertEqual(float(s), 1.0)
        self.assertEqual(float(y[0, 3]), 6.0)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_matches_searchsorted_workaround(self, device):
        x = self.blocks(device, -12, 11)  # amax / M from below the smallest code to above the largest
        codes = torch.tensor(self.codes, device=device)
        peaks = x.reshape(*x.shape[:-1], -1, 16).abs().amax(-1)
        s_ref = codes[torch.searchsorted(codes, (peaks / 6.0).clamp(max=codes[-1]))]
        y, s, _ = block_round(x, self.ceil_spec, return_aux=True)
        self.assertTrue(torch.equal(s.squeeze(-1), s_ref))
        self.assertTrue(torch.equal(y, block_round(x, self.nearest_spec, scales=s_ref)))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_never_clips_block_max(self, device):
        x = self.blocks(device, -4, 4)
        peaks = x.reshape(*x.shape[:-1], -1, 16).abs().amax(-1, keepdim=True)
        _, s, _ = block_round(x, self.ceil_spec, return_aux=True)
        self.assertTrue((peaks / s <= 6.0).all())
        _, s, _ = block_round(x, self.nearest_spec, return_aux=True)
        self.assertTrue((peaks / s > 6.0).any())

    @parameterized.expand([(d,) for d in DEVICES])
    def test_matches_ue8m0_ceil_on_power_of_two_codebook(self, device):
        # amax / M on a 10-bit mantissa grid: column 0 is the exact powers of two, the rest stay clear
        # of the few float32 steps above a power of two where ue8m0_ceil's ceil(log2) is not exact.
        q = torch.exp2(torch.arange(-20.0, 21.0)).unsqueeze(1) * (1 + torch.arange(1024.0) / 1024)
        stat = (448.0 * q).to(device)
        s = encode_scale(stat, BlockFormat(E4M3, UE8M0, 32, 448.0, "ceil"))
        self.assertTrue(torch.equal(s, encode_scale(stat, MXFP8)))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_exact_code_zero_and_saturation(self, device):
        codes = torch.tensor(self.codes, device=device)
        lo, hi = self.codes[0], self.codes[-1]
        self.assertTrue(torch.equal(encode_scale(codes * 6.0, self.ceil_spec), codes))
        stat = torch.tensor([[0.0, 1e-6, 1e6, float("inf")]], device=device)
        self.assertEqual(encode_scale(stat, self.ceil_spec).tolist(), [[lo, lo, hi, hi]])
        nan = torch.tensor([[float("nan")]], device=device)
        self.assertTrue(torch.isnan(encode_scale(nan, self.ceil_spec)).all())
        y, s, _ = block_round(torch.zeros(1, 16, device=device), self.ceil_spec, return_aux=True)
        self.assertEqual(float(s), lo)
        self.assertTrue(torch.equal(y, torch.zeros_like(y)))


class TestNoSubnormalScale(unittest.TestCase):
    """Issue #40: a scale codebook with subnormals=False (exponent code 0 is a normal binade, no zero code)."""

    @parameterized.expand([(d,) for d in DEVICES])
    def test_scales_are_codes_and_zero_block_stays_zero(self, device):
        codes = set(UE2M2.values)
        x = torch.randn(64, 16, device=device) * torch.logspace(-3, 1, 64, device=device).unsqueeze(1)
        x[0] = 0.0
        for encode in sorted(_SCALE_ENCODES):
            y, s, _ = block_round(x, BlockFormat(E2M1, UE2M2, 16, 6.0, encode), return_aux=True)
            self.assertTrue(torch.equal(y[0], torch.zeros_like(y[0])), msg=encode)
            if encode != "amax_over_M":  # no codebook snap
                self.assertLessEqual({float(v) for v in s.flatten()}, codes, msg=encode)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_nearest_reaches_bottom_binade(self, device):
        # 0.15625 = 1.25 * 2^-3 is a code only without the subnormal row.
        x = torch.zeros(1, 16, device=device)
        x[0, 0] = 6.0 * 0.15625
        y, s, _ = block_round(x, BlockFormat(E2M1, UE2M2, 16, 6.0, "nearest"), return_aux=True)
        self.assertEqual(float(s), 0.15625)
        self.assertEqual(float(y[0, 0]), 6.0 * 0.15625)


class TestOcpFloorCarryScaleEncode(unittest.TestCase):
    """Issue #45: ``ocp_floor`` plus one exponent step only when the rounded peak carries."""

    carry_spec = BlockFormat(E2M1, UE8M0, 4, 6.0, "ocp_floor_carry")
    floor_spec = BlockFormat(E2M1, UE8M0, 4, 6.0, "ocp_floor")
    x2_spec = BlockFormat(E2M1, UE8M0, 4, 6.0, "ocp_floor_x2")

    @staticmethod
    def blocks(device):
        g = torch.Generator().manual_seed(0)
        mag = torch.exp2(torch.randint(-8, 9, (256, 1), generator=g).float())
        return (torch.randn(256, 64, generator=g) * mag).to(device)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_issue_example(self, device):
        x = torch.tensor([[7.5, 3.0, 1.0, -0.5, 5.5, 3.0, 1.0, -0.5]], device=device)
        y, s, _ = block_round(x, self.carry_spec, return_aux=True)
        self.assertEqual(s.flatten().tolist(), [2.0, 1.0])
        self.assertEqual(y.tolist(), [[8.0, 3.0, 1.0, -0.0, 6.0, 3.0, 1.0, -0.5]])

    @parameterized.expand([(d,) for d in DEVICES])
    def test_matches_explicit_scales_workaround(self, device):
        x = self.blocks(device)
        peaks = x.reshape(*x.shape[:-1], -1, 4).abs().amax(-1)
        e = torch.floor(torch.log2(peaks))
        carry = peaks / 2**e >= 2 - 2.0 ** -(E2M1.mantissa_bits + (E2M1.exponent_bits > 0))
        s_ref = 2 ** (e - math.floor(math.log2(6.0)) + carry)
        y, s, _ = block_round(x, self.carry_spec, return_aux=True)
        self.assertTrue(torch.equal(s.squeeze(-1), s_ref))
        self.assertTrue(torch.equal(y, block_round(x, self.floor_spec, scales=s_ref)))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_floor_without_carry_x2_with_carry(self, device):
        x = self.blocks(device)
        peaks = x.reshape(*x.shape[:-1], -1, 4).abs().amax(-1, keepdim=True)
        carry = torch.frexp(peaks).mantissa >= 1.75 / 2  # peak in [7, 8) * 2**k
        self.assertTrue(carry.any() and not carry.all())
        _, s, _ = block_round(x, self.carry_spec, return_aux=True)
        _, s_floor, _ = block_round(x, self.floor_spec, return_aux=True)
        _, s_x2, _ = block_round(x, self.x2_spec, return_aux=True)
        self.assertTrue(torch.equal(s, torch.where(carry, s_x2, s_floor)))
        self.assertTrue((peaks / s < 7.0).all())  # the rounded peak never exceeds the element maximum 6

    @parameterized.expand([(d,) for d in DEVICES])
    def test_threshold_e2m1_and_integer_element(self, device):
        # E2M1: threshold 1.75, inclusive. 7 already rounds to 8 (ties to even) under scale 1.
        stat = torch.tensor([[6.0, 6.875, 7.0, 7.5, 8.0, 13.5, 14.0]], device=device)
        self.assertEqual(encode_scale(stat, self.carry_spec).tolist(), [[1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0]])
        # Integer element (exponent_bits=0), codes k / 64 up to 127 / 64: threshold 2 - 2**-7 = 255 / 128.
        mxint8 = BlockFormat(FloatingPoint(1, 0, 7, 0, 8), UE8M0, 4, 127 / 64, "ocp_floor_carry")
        stat = torch.tensor([[1.0, 127 / 64, 509 / 256, 255 / 128, 511 / 256, 2.0]], device=device)
        self.assertEqual(encode_scale(stat, mxint8).tolist(), [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])

    @parameterized.expand([(d,) for d in DEVICES])
    def test_zero_block_and_scale_clamp(self, device):
        zeros = torch.zeros(1, 4, device=device)
        y, s, _ = block_round(zeros, self.carry_spec, return_aux=True)
        _, s_floor, _ = block_round(zeros, self.floor_spec, return_aux=True)
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue(torch.equal(s, s_floor))
        self.assertTrue(torch.equal(y, zeros))
        # The exponent is clamped to the scale codebook, as in ocp_floor.
        stat = torch.tensor([[float("inf")]], device=device)
        self.assertEqual(float(encode_scale(stat, self.carry_spec)), UE8M0.maximum)


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


if __name__ == "__main__":
    unittest.main()
