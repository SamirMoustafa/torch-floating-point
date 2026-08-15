import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from floating_point.data_types import FloatingPoint
from floating_point.round import Round


def _is_ue8m0_scale(scale_fp: FloatingPoint) -> bool:
    return scale_fp.mantissa_bits == 0 and scale_fp.reserved_exponent and not scale_fp.is_signed


def _positive_finite_values(fp: FloatingPoint) -> List[float]:
    return [v for v in fp.values if math.isfinite(v) and v > 0.0]


def encode_scale(raw: Tensor, scale_fp: FloatingPoint) -> Tensor:
    """Encode positive raw scales into ``scale_fp`` codes.

    UE8M0 (M0 + reserved): CUDA-style round-up to the next representable power of two.
    Other formats: element-wise nearest ``Round`` on the absolute value.
    """
    lo = float(scale_fp.minimum) if scale_fp.minimum > 0.0 else min(_positive_finite_values(scale_fp) or [1.0])
    hi = float(scale_fp.maximum)
    raw_clamped = raw.clamp(min=lo, max=hi)
    if _is_ue8m0_scale(scale_fp):
        # Smallest 2^(E-bias) >= raw: E = ceil(log2(raw)) + bias, E in [0, 2^e-2]
        max_stored = (1 << scale_fp.exponent_bits) - 2
        exp = torch.ceil(torch.log2(raw_clamped)) + float(scale_fp.bias)
        exp = exp.clamp(0.0, float(max_stored))
        return torch.exp2(exp - float(scale_fp.bias))
    rounded = Round(scale_fp)(raw_clamped.abs())
    return rounded.abs().clamp(min=lo, max=hi)


def _as_blocks(x: Tensor, block_size: int) -> Tuple[Tensor, torch.Size]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension {x.shape[-1]} must be divisible by block_size={block_size}")
    orig_shape = x.shape
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    return blocks, orig_shape


def _normalize_scales(scales: Tensor, n_blocks: int) -> Tensor:
    """Ensure scales broadcast as (..., n_blocks, 1)."""
    if scales.ndim >= 2 and scales.shape[-1] == 1 and scales.shape[-2] == n_blocks:  # noqa: PLR2004
        return scales
    if scales.shape[-1] == n_blocks:
        return scales.unsqueeze(-1)
    raise ValueError(
        f"scales must have shape (..., n_blocks) or (..., n_blocks, 1) "
        f"with n_blocks={n_blocks}; got {tuple(scales.shape)}"
    )


def block_round(
    x: Tensor,
    elem_fp: FloatingPoint,
    scale_fp: FloatingPoint,
    M: Optional[float] = None,  # noqa: N803
    block_size: int = 16,
    scales: Optional[Tensor] = None,
    return_aux: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Block-scaled round: ``y = Round_elem(x / s) * s``.

    Absmax mode (``scales is None``): ``s = encode_scale(amax/M).detach()`` — STE on ``x`` only.
    Learnable mode (``scales`` given): STE on ``x``; gradients flow into ``scales`` via ``e * s``.
    """
    max_elem = float(elem_fp.maximum) if M is None else float(M)
    if max_elem <= 0:
        raise ValueError(f"M must be positive, got {max_elem}")
    blocks, orig_shape = _as_blocks(x, block_size)
    n_blocks = blocks.shape[-2]
    elem_rounder = Round(elem_fp)
    if scales is None:
        amax = blocks.detach().abs().amax(dim=-1, keepdim=True)
        raw = amax / max_elem
        zero_mask = amax == 0
        if zero_mask.any():
            lo = float(scale_fp.minimum) if scale_fp.minimum > 0 else min(_positive_finite_values(scale_fp) or [1.0])
            raw = torch.where(zero_mask, torch.full_like(raw, lo), raw)
        s = encode_scale(raw, scale_fp).detach()
    else:
        s = _normalize_scales(scales, n_blocks)
        if s.device != blocks.device or s.dtype != blocks.dtype:
            s = s.to(device=blocks.device, dtype=blocks.dtype)
    elems = elem_rounder(blocks / s)
    y = (elems * s).reshape(orig_shape)
    if return_aux:
        return y, s, elems.reshape(orig_shape)
    return y


class BlockRound:
    """Stateful block-scaled rounder (NVFP4 / MXFP8 presets)."""

    def __init__(
        self,
        elem_fp: FloatingPoint,
        scale_fp: FloatingPoint,
        M: Optional[float] = None,  # noqa: N803
        block_size: int = 16,
    ):
        self.elem_fp = elem_fp
        self.scale_fp = scale_fp
        self.M = float(elem_fp.maximum) if M is None else float(M)
        self.block_size = block_size

    def __call__(
        self, x: Tensor, scales: Optional[Tensor] = None, return_aux: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        return block_round(
            x, self.elem_fp, self.scale_fp, M=self.M, block_size=self.block_size, scales=scales, return_aux=return_aux
        )


def sample_block_scaled(
    shape: Union[torch.Size, Tuple[int, ...]],
    elem_fp: FloatingPoint,
    scale_fp: FloatingPoint,
    M: Optional[float] = None,  # noqa: N803
    block_size: int = 16,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Draw recoverable codebook blocks ``x_i = e_i * s`` with some ``|e| = M``."""
    max_elem = float(elem_fp.maximum) if M is None else float(M)
    if shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension {shape[-1]} must be divisible by block_size={block_size}")
    elem_codes = [v for v in elem_fp.values if math.isfinite(v) and abs(v) <= max_elem + 1e-12]
    if not any(abs(v - max_elem) < 1e-6 or abs(v + max_elem) < 1e-6 for v in elem_codes):  # noqa: PLR2004
        raise ValueError(f"elem_fp codebook has no ±M={max_elem} value")
    scale_codes = _positive_finite_values(scale_fp)
    if not scale_codes:
        raise ValueError("scale_fp has no positive finite codes")
    n_blocks = shape[-1] // block_size
    leading = int(math.prod(shape[:-1])) if len(shape) > 1 else 1
    total_blocks = leading * n_blocks
    elem_t = torch.tensor(elem_codes, dtype=dtype, device=device)
    scale_t = torch.tensor(scale_codes, dtype=dtype, device=device)
    peak_idx = [i for i, v in enumerate(elem_codes) if abs(abs(v) - max_elem) < 1e-6]  # noqa: PLR2004
    peak_t = torch.tensor(peak_idx, dtype=torch.long, device=device)
    s_idx = torch.randint(0, len(scale_codes), (total_blocks,), generator=generator, device=device)
    e_idx = torch.randint(0, len(elem_codes), (total_blocks, block_size), generator=generator, device=device)
    # Force one lane per block to ±M so absmax recovers s
    lane = torch.randint(0, block_size, (total_blocks,), generator=generator, device=device)
    peak_choice = peak_t[torch.randint(0, len(peak_idx), (total_blocks,), generator=generator, device=device)]
    e_idx[torch.arange(total_blocks, device=device), lane] = peak_choice
    return (elem_t[e_idx] * scale_t[s_idx].view(total_blocks, 1)).reshape(shape)
