import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn.functional import pad as pad_nd

from floating_point.data_types import FloatingPoint
from floating_point.round import Round

_SCALE_ENCODES = frozenset(
    {"nearest", "ue8m0_ceil", "ue8m0_floor", "ocp_floor", "ocp_floor_x2", "amax_over_M", "signed_peak"}
)
_PADS = frozenset({"error", "zero"})
_TILE_RANK = 2
_BlockSize = Union[int, Tuple[int, int]]


@dataclass(frozen=True)
class BlockFormat:
    """Block-scaled layout: element codebook, scale codebook, block geometry, M, encode."""

    elem_fp: FloatingPoint
    scale_fp: FloatingPoint
    block_size: _BlockSize
    M: float
    scale_encode: str
    dims: Tuple[int, ...] = (-1,)
    s_global: float = 1.0
    zero_point: float = 0.0
    pad: str = "error"

    def __post_init__(self) -> None:
        if self.scale_encode not in _SCALE_ENCODES:
            raise ValueError(f"scale_encode must be one of {sorted(_SCALE_ENCODES)}, got {self.scale_encode!r}")
        if self.pad not in _PADS:
            raise ValueError(f"pad must be one of {sorted(_PADS)}, got {self.pad!r}")
        if not math.isfinite(self.s_global) or self.s_global == 0.0:
            raise ValueError(f"s_global must be finite and non-zero, got {self.s_global}")
        if not math.isfinite(self.zero_point):
            raise ValueError(f"zero_point must be finite, got {self.zero_point}")
        nd = _block_ndims(self.block_size)
        dims = self.dims
        if not isinstance(dims, tuple) or not dims:
            raise ValueError(f"dims must be a non-empty tuple of axis indices, got {dims!r}")
        if nd == _TILE_RANK and dims == (-1,):
            object.__setattr__(self, "dims", (-2, -1))
        elif len(self.dims) != nd:
            raise ValueError(f"dims has length {len(self.dims)} but block_size has {nd} axis(es)")


def _block_ndims(block_size: _BlockSize) -> int:
    if isinstance(block_size, int):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        return 1
    if (
        isinstance(block_size, tuple)
        and len(block_size) == _TILE_RANK
        and all(isinstance(v, int) and v > 0 for v in block_size)
    ):
        return _TILE_RANK
    raise ValueError(f"block_size must be a positive int or a pair (h, w), got {block_size!r}")


def _block_hw(spec: BlockFormat) -> Tuple[int, ...]:
    if isinstance(spec.block_size, int):
        return (spec.block_size,)
    return (int(spec.block_size[0]), int(spec.block_size[1]))


def _positive_finite_values(fp: FloatingPoint) -> List[float]:
    return [v for v in fp.values if math.isfinite(v) and v > 0.0]


def _scale_bounds(scale_fp: FloatingPoint) -> Tuple[float, float]:
    lo = float(scale_fp.minimum) if scale_fp.minimum > 0.0 else min(_positive_finite_values(scale_fp) or [1.0])
    return lo, float(scale_fp.maximum)


def _emax_elem(fp: FloatingPoint) -> float:
    """Unbiased exponent of the largest finite normal in ``fp`` (OCP / AWS E_max)."""
    if fp.exponent_bits == 0:
        m = float(fp.maximum)
        if m <= 0.0:
            return 0.0
        return float(math.floor(math.log2(m)))
    max_stored = (2**fp.exponent_bits - 2) if fp.reserved_exponent else (2**fp.exponent_bits - 1)
    return float(max_stored - fp.bias)


def _log2_clamp(x: Tensor, lo: float) -> Tensor:
    return torch.log2(x.clamp(min=lo))


def _pow2_from_unbiased_exp(exp: Tensor, scale_fp: FloatingPoint) -> Tensor:
    max_stored = (1 << scale_fp.exponent_bits) - (2 if scale_fp.reserved_exponent else 1)
    stored = (exp + float(scale_fp.bias)).clamp(0.0, float(max_stored))
    return torch.exp2(stored - float(scale_fp.bias))


def encode_scale(stat: Tensor, spec: BlockFormat) -> Tensor:
    """Encode a per-block statistic into ``spec.scale_fp`` with ``spec.scale_encode``.

    ``stat`` is block absmax for every policy except ``signed_peak``, which takes the
    signed value of maximum magnitude. OCP / AWS policies use ``stat`` directly;
    ``nearest``, ``amax_over_M``, and the UE8M0 policies use ``stat / M``.
    """
    scale_fp = spec.scale_fp
    lo, hi = _scale_bounds(scale_fp)
    encode = spec.scale_encode
    if encode == "signed_peak":
        raw = torch.where(stat == 0, torch.full_like(stat, lo), stat / (-spec.M))
        s = Round(scale_fp)(raw)
        sign = torch.where(raw == 0, torch.ones_like(raw), torch.sign(raw))
        return torch.where(s == 0, sign * lo, s)
    if encode == "ocp_floor":
        exp = torch.floor(_log2_clamp(stat, lo)) - _emax_elem(spec.elem_fp)
        return _pow2_from_unbiased_exp(exp, scale_fp)
    if encode == "ocp_floor_x2":
        exp = torch.floor(_log2_clamp(stat, lo)) - (_emax_elem(spec.elem_fp) - 1.0)
        return _pow2_from_unbiased_exp(exp, scale_fp)
    raw = (stat / spec.M).clamp(min=lo, max=hi)
    zero_mask = stat == 0
    if zero_mask.any():
        raw = torch.where(zero_mask, torch.full_like(raw, lo), raw)
    if encode == "ue8m0_ceil":
        return _pow2_from_unbiased_exp(torch.ceil(_log2_clamp(raw, lo)), scale_fp)
    if encode == "ue8m0_floor":
        return _pow2_from_unbiased_exp(torch.floor(_log2_clamp(raw, lo)), scale_fp)
    if encode == "amax_over_M":
        return raw
    if encode == "nearest":
        rounded = Round(scale_fp)(raw.abs())
        return rounded.abs().clamp(min=lo, max=hi)
    raise ValueError(f"unknown scale_encode {encode!r}")


def _norm_dims(ndim: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    out = []
    for d in dims:
        nd = d if d >= 0 else ndim + d
        if nd < 0 or nd >= ndim:
            raise ValueError(f"dim {d} out of range for ndim={ndim}")
        out.append(nd)
    if len(set(out)) != len(out):
        raise ValueError(f"dims must be unique, got {dims}")
    return tuple(out)


@dataclass
class _Layout:
    perm: Tuple[int, ...]
    inv: Tuple[int, ...]
    orig_shape: Tuple[int, ...]
    work_shape: Tuple[int, ...]
    pad_h: int
    pad_w: int
    nd: int
    sizes: Tuple[int, ...]
    n_h: int
    n_w: int


def _layout_for(shape: Tuple[int, ...], spec: BlockFormat) -> _Layout:
    ndim = len(shape)
    sizes = _block_hw(spec)
    nd = len(sizes)
    dims = _norm_dims(ndim, spec.dims)
    if nd == 1:
        dim = dims[0]
        rest = [i for i in range(ndim) if i != dim]
        perm = (*rest, dim)
    else:
        d0, d1 = dims
        rest = [i for i in range(ndim) if i not in (d0, d1)]
        perm = (*rest, d0, d1)
    inv_l = [0] * ndim
    for i, p in enumerate(perm):
        inv_l[p] = i
    work_shape = tuple(shape[p] for p in perm)
    if nd == 1:
        (k,) = sizes
        length = work_shape[-1]
        pad_w = 0 if length % k == 0 else (k - length % k)
        if pad_w and spec.pad == "error":
            raise ValueError(f"Dimension {work_shape[-1]} must be divisible by block_size={k}")
        tiled_w = length + pad_w
        return _Layout(perm, tuple(inv_l), shape, work_shape, 0, pad_w, 1, sizes, 1, tiled_w // k)
    h, w = sizes
    height, width = work_shape[-2], work_shape[-1]
    pad_h = 0 if height % h == 0 else (h - height % h)
    pad_w = 0 if width % w == 0 else (w - width % w)
    if (pad_h or pad_w) and spec.pad == "error":
        raise ValueError(f"Shape {(height, width)} must be divisible by block_size={(h, w)}")
    return _Layout(
        perm,
        tuple(inv_l),
        shape,
        work_shape,
        pad_h,
        pad_w,
        _TILE_RANK,
        sizes,
        (height + pad_h) // h,
        (width + pad_w) // w,
    )


def _permute(x: Tensor, layout: _Layout) -> Tensor:
    if layout.perm == tuple(range(x.ndim)):
        return x
    return x.permute(layout.perm).contiguous()


def _unpermute(x: Tensor, layout: _Layout) -> Tensor:
    if layout.inv == tuple(range(x.ndim)):
        return x
    return x.permute(layout.inv)


def _pad_work(x: Tensor, layout: _Layout, value: float = 0.0) -> Tensor:
    if layout.nd == 1:
        if layout.pad_w == 0:
            return x
        return pad_nd(x, (0, layout.pad_w), value=value)
    if layout.pad_h == 0 and layout.pad_w == 0:
        return x
    return pad_nd(x, (0, layout.pad_w, 0, layout.pad_h), value=value)


def _crop_work(x: Tensor, layout: _Layout) -> Tensor:
    if layout.nd == 1:
        length = layout.work_shape[-1]
        return x if x.shape[-1] == length else x[..., :length]
    height, width = layout.work_shape[-2], layout.work_shape[-1]
    return x if x.shape[-2] == height and x.shape[-1] == width else x[..., :height, :width]


def _as_blocks(x: Tensor, layout: _Layout) -> Tensor:
    if layout.nd == 1:
        k = layout.sizes[0]
        return x.reshape(*x.shape[:-1], layout.n_w, k)
    h, w = layout.sizes
    leading = x.shape[:-2]
    return x.reshape(*leading, layout.n_h, h, layout.n_w, w)


def _from_blocks(blocks: Tensor, layout: _Layout) -> Tensor:
    if layout.nd == 1:
        k = layout.sizes[0]
        return blocks.reshape(*blocks.shape[:-2], layout.n_w * k)
    h, w = layout.sizes
    leading = blocks.shape[:-4]
    return blocks.reshape(*leading, layout.n_h * h, layout.n_w * w)


def _amax(blocks: Tensor, layout: _Layout) -> Tensor:
    if layout.nd == 1:
        return blocks.abs().amax(dim=-1, keepdim=True)
    return blocks.abs().amax(dim=(-3, -1), keepdim=True)


def _signed_peak(blocks: Tensor, layout: _Layout) -> Tensor:
    detached = blocks.detach()
    if layout.nd == 1:
        idx = detached.abs().argmax(dim=-1, keepdim=True)
        return detached.gather(-1, idx)
    t = detached.transpose(-3, -2)
    flat = t.reshape(*t.shape[:-2], t.shape[-2] * t.shape[-1])
    idx = flat.abs().argmax(dim=-1, keepdim=True)
    peak = flat.gather(-1, idx)
    return peak.reshape(*peak.shape[:-3], layout.n_h, 1, layout.n_w, 1)


def _normalize_scales(scales: Tensor, layout: _Layout) -> Tensor:
    n_blocks = layout.n_w if layout.nd == 1 else None
    if layout.nd == 1:
        if scales.ndim >= 2 and scales.shape[-1] == 1 and scales.shape[-2] == n_blocks:  # noqa: PLR2004
            return scales
        if scales.shape[-1] == n_blocks:
            return scales.unsqueeze(-1)
        raise ValueError(
            f"scales must have shape (..., n_blocks) or (..., n_blocks, 1) "
            f"with n_blocks={n_blocks}; got {tuple(scales.shape)}"
        )
    n_h, n_w = layout.n_h, layout.n_w
    if scales.shape[-4:] == (n_h, 1, n_w, 1):
        return scales
    if scales.shape[-2:] == (n_h, n_w):
        return scales.unsqueeze(-2).unsqueeze(-1)
    raise ValueError(
        f"scales must have shape (..., n_h, n_w) or (..., n_h, 1, n_w, 1) "
        f"with n_h={n_h}, n_w={n_w}; got {tuple(scales.shape)}"
    )


def _require_nonzero_finite_sg(g: Tensor) -> Tensor:
    if not torch.isfinite(g).all() or bool((g == 0).any()):
        raise ValueError("s_global must be finite and non-zero")
    return g


def _sg_blocks(
    s_global: Optional[Union[Tensor, float]],
    spec: BlockFormat,
    x: Tensor,
    blocks: Tensor,
    layout: _Layout,
) -> Tensor:
    if s_global is None:
        g: Union[Tensor, float] = spec.s_global
    else:
        g = s_global
    if not isinstance(g, Tensor):
        return _require_nonzero_finite_sg(blocks.new_tensor(float(g)))
    g = g.to(device=blocks.device, dtype=blocks.dtype)
    _require_nonzero_finite_sg(g)
    if g.ndim == 0 or g.numel() == 1:
        return g
    if tuple(g.shape) != tuple(x.shape):
        raise ValueError(f"s_global tensor shape {tuple(g.shape)} must be scalar or match x {tuple(x.shape)}")
    return _as_blocks(_pad_work(_permute(g, layout), layout, value=1.0), layout)


def _to_orig(y_blocks: Tensor, layout: _Layout) -> Tensor:
    return _unpermute(_crop_work(_from_blocks(y_blocks, layout), layout), layout)


def block_round(
    x: Tensor,
    spec: BlockFormat,
    *,
    scales: Optional[Tensor] = None,
    rounder: Type[Round] = Round,
    return_aux: bool = False,
    s_global: Optional[Union[Tensor, float]] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Block-scaled round: ``y = (rounder(x / (s * s_g)) - z) * s * s_g``.

    Absmax (``scales is None``): ``s = encode_scale(stat).detach()`` — STE on ``x`` only.
    Learnable (``scales`` given): gradients into ``scales`` via ``(e - z) * s``.
    """
    if spec.M <= 0:
        raise ValueError(f"M must be positive, got {spec.M}")
    if scales is None and spec.zero_point != 0:
        raise ValueError("zero_point != 0 requires scales=; absmax ignores the affine offset")
    layout = _layout_for(tuple(x.shape), spec)
    work = _pad_work(_permute(x, layout), layout)
    blocks = _as_blocks(work, layout)
    sg = _sg_blocks(s_global, spec, x, blocks, layout)
    z = spec.zero_point
    elem_rounder = rounder(spec.elem_fp)
    if scales is None:
        scaled = blocks / sg
        if spec.scale_encode == "signed_peak":
            stat = _signed_peak(scaled, layout)
        else:
            stat = _amax(scaled.detach(), layout)
        s = encode_scale(stat, spec).detach()
    else:
        s = _normalize_scales(scales, layout)
        if s.device != blocks.device or s.dtype != blocks.dtype:
            s = s.to(device=blocks.device, dtype=blocks.dtype)
    denom = s * sg
    elems = elem_rounder(blocks / denom + z)
    y_blocks = (elems - z) * denom
    y = _to_orig(y_blocks, layout)
    if return_aux:
        return y, s, _to_orig(elems, layout)
    return y


class BlockRound:
    """Stateful block-scaled rounder. ``rounder`` is a ``Round`` subclass."""

    def __init__(self, spec: BlockFormat, rounder: Type[Round] = Round):
        self.spec = spec
        self.rounder = rounder

    def __call__(
        self,
        x: Tensor,
        scales: Optional[Tensor] = None,
        return_aux: bool = False,
        s_global: Optional[Union[Tensor, float]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        return block_round(x, self.spec, scales=scales, rounder=self.rounder, return_aux=return_aux, s_global=s_global)
