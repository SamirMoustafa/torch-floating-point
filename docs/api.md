# API

Public names: `FloatingPoint`, `Round`, `BlockFormat`, `BlockRound`, `block_round`, `sample_block_scaled`, `round`, `inplace`.

## `FloatingPoint`

```python
FloatingPoint(
    sign_bits: int,
    exponent_bits: int,
    mantissa_bits: int,
    bias: int,
    bits: int,
    max_mantissa_at_max_exponent: int | None = None,
    reserved_exponent: bool = True,
)
```

Requires `bits == sign_bits + exponent_bits + mantissa_bits` and `sign_bits` in `{0, 1}`.

| Attr | Meaning |
| --- | --- |
| `.minimum` / `.maximum` | Finite range used to clamp before rounding |
| `.epsilon` | `2 ** (-mantissa_bits)` |
| `.values` | All `2**bits` decoded codes, sorted |
| `.is_signed` | `sign_bits > 0` |

## `Round`

```python
rounder = Round(fp)  # fp: FloatingPoint
y = rounder(x)  # STE autograd; needs the compiled extension
```

Clamps to `[fp.minimum, fp.maximum]`, then rounds with the C++/CUDA kernel. Subclass and override `forward` for a custom estimator; pass that class as `rounder=` on `BlockRound`.

## `BlockFormat` / `BlockRound`

```python
BlockFormat(
    elem_fp,
    scale_fp,
    block_size,
    M,
    scale_encode,
    dims=(-1,),
    s_global=1.0,
    zero_point=0.0,
    pad="error",
)
# block_size: int or (h, w)
# scale_encode in {nearest, ue8m0_ceil, ue8m0_floor, ocp_floor, ocp_floor_x2,
#                  amax_over_M, signed_peak}

BlockRound(spec)
BlockRound(spec, rounder=MyRound)
```

Call as `rounder(x, scales=None, return_aux=False, s_global=None)`. Reconstruct is \(y = (e - z)\,s\,s_{\mathrm{global}}\). Hardware packings (NVFP4, MX, …) are `BlockFormat(...)` in [Block scale](block.md), not package exports.

```python
block_round(x, spec, scales=None, rounder=Round, return_aux=False, s_global=None)
sample_block_scaled(shape, spec, generator=None, device=None, dtype=torch.float32, s_global=None)
```

## Functional round

```python
round(input, exponent_bits, mantissa_bits, bias, reserved_exponent=True, max_mantissa_at_max_exponent=None)
inplace(...)  # same args; writes through the extension
```

`round` is a signed format (`sign_bits=1`). `inplace` raises if the extension is missing.

## Autograd

See [Autograd](autograd.md) for clipped STE and custom estimators. `StraightThroughEstimator` is the `torch.autograd.Function` behind `Round`. Nested quantization of the same tensor raises `RuntimeError: Double quantization detected.`
