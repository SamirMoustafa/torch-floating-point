# API

Public names: `from floating_point import ...`

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

Clamps to `[fp.minimum, fp.maximum]`, then rounds with the C++/CUDA kernel.

## `BlockRound`

```python
BlockRound(elem_fp, scale_fp, M=None, block_size=16)
```

`M` defaults to `elem_fp.maximum`. Call as `rounder(x, scales=None, return_aux=False)`.

Functional form:

```python
block_round(x, elem_fp, scale_fp, M=None, block_size=16, scales=None, return_aux=False)
sample_block_scaled(shape, elem_fp, scale_fp, M=None, block_size=16, generator=None, device=None, dtype=torch.float32)
```

## Functional round

```python
round(input, exponent_bits, mantissa_bits, bias, reserved_exponent=True, max_mantissa_at_max_exponent=None)
inplace(...)  # same args; writes through the extension
```

`round` is a signed format (`sign_bits=1`). `inplace` raises if the extension is missing.

## Autograd

See [Autograd](autograd.md) for clipped STE and custom estimators. `StraightThroughEstimator` is the `torch.autograd.Function` behind `Round`. Nested quantization of the same tensor raises `RuntimeError: Double quantization detected.`
