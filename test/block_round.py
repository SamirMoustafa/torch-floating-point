import unittest

import torch
from parameterized import parameterized
from torch import nn

from floating_point import BlockRound, FloatingPoint, block_round, sample_block_scaled
from floating_point.block_round import encode_scale

E2M1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
E4M3 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
E8M0 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestBlockRoundNVFP4(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_sampler_absmax_roundtrip(self, device):
        torch.manual_seed(0)
        x = sample_block_scaled((4, 64), E2M1, E4M3, M=6, block_size=16, device=torch.device(device))
        y = block_round(x, E2M1, E4M3, M=6, block_size=16)
        self.assertTrue(torch.allclose(x, y, rtol=0, atol=0))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_plain_e2m1_grid_has_drift(self, device):
        # Independently rounded E2M1 values (no shared scale) are not absmax-invariant.
        codes = torch.tensor(
            [v for v in E2M1.values if abs(v) <= 6], dtype=torch.float32, device=device
        )
        idx = torch.randint(0, len(codes), (8, 64), device=device)
        x = codes[idx]
        y = block_round(x, E2M1, E4M3, M=6, block_size=16)
        self.assertGreater(float((x - y).abs().max()), 0.0)

    @parameterized.expand([(d,) for d in DEVICES])
    def test_ste_absmax_grad_x_only(self, device):
        x = torch.randn(2, 32, device=device, requires_grad=True)
        y, scales, elems = block_round(x, E2M1, E4M3, M=6, block_size=16, return_aux=True)
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
        y = block_round(x, E2M1, E4M3, M=6, block_size=16, scales=s)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad.abs().sum()), 0.0)

    def test_bad_block_size_raises(self):
        x = torch.randn(3, 10)
        with self.assertRaises(ValueError):
            block_round(x, E2M1, E4M3, M=6, block_size=16)


class TestBlockRoundMXFP8(unittest.TestCase):
    @parameterized.expand([(d,) for d in DEVICES])
    def test_sampler_absmax_roundtrip(self, device):
        torch.manual_seed(1)
        x = sample_block_scaled((2, 64), E4M3, E8M0, M=448, block_size=32, device=torch.device(device))
        y = BlockRound(E4M3, E8M0, M=448, block_size=32)(x)
        self.assertTrue(torch.allclose(x, y, rtol=0, atol=0))

    @parameterized.expand([(d,) for d in DEVICES])
    def test_ue8m0_round_up_covers_amax(self, device):
        # raw scale between two powers of two must round up
        raw = torch.tensor([[1.5]], dtype=torch.float32, device=device)
        s = encode_scale(raw, E8M0)
        self.assertEqual(float(s), 2.0)


class TestBlockRoundClass(unittest.TestCase):
    def test_defaults_m_to_elem_maximum(self):
        br = BlockRound(E2M1, E4M3, block_size=16)
        self.assertEqual(br.M, 6.0)


if __name__ == "__main__":
    unittest.main()
