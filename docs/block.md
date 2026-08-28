# Block scale

Shared per-block scale: `y_i = Round_elem(x_i / s) * s`.

[OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) ([Rouhani et al., 2023](https://arxiv.org/abs/2310.10537)) uses UE8M0 scales over blocks of 32 (MXFP8 / MXFP4) — the same packing NVIDIA Blackwell and AMD CDNA4 implement. **NVFP4** is NVIDIA’s variant: block 16 and E4M3 scales ([NVIDIA, 2025](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)). See [Formats](formats.md#references) for the full reference list.

Absmax mode detaches `s` (STE on `x` only). Pass `scales=` for learnable QAT scales with gradients.

```python
from floating_point import BlockRound, FloatingPoint, sample_block_scaled

fp4 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
fp8 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
ue8 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)

# NVFP4: E2M1 elements + E4M3 scales, block_size=16
nvfp4 = BlockRound(fp4, fp8, M=6, block_size=16)
y = nvfp4(x)                 # absmax scales, STE on x only
y = nvfp4(x, scales=s)       # gradients into scales

# MXFP8: E4M3 elements + UE8M0 scales, block_size=32
mxfp8 = BlockRound(fp8, ue8, M=448, block_size=32)

# MXFP4: E2M1 elements + UE8M0 scales, block_size=32
mxfp4 = BlockRound(fp4, ue8, M=6, block_size=32)
```

| Preset | Elements | Scales | `M` | `block_size` | Spec |
| --- | --- | --- | --- | --- | --- |
| NVFP4 | E2M1 | E4M3-FN | `6` | `16` | NVIDIA Blackwell |
| MXFP8 | E4M3-FN | UE8M0 | `448` | `32` | OCP MX (NVIDIA, AMD CDNA4, …) |
| MXFP4 | E2M1 | UE8M0 | `6` | `32` | OCP MX (NVIDIA, AMD CDNA4, …) |

Last dimension of `x` must be divisible by `block_size`.

## Absmax vs learnable

- **Absmax** (`scales is None`): `s = encode(amax / M).detach()` — straight-through on `x` only.
- **Learnable** (`scales=`): same element round; `y = e * s` so `s` gets gradients. See [Autograd](autograd.md).

`return_aux=True` yields `(y, s, elems)`.

## Recoverable samples

Draw codebook blocks that round-trip under absmax (`x_i = e_i * s` with some `|e| = M`):

```python
x = sample_block_scaled((8, 64), fp4, fp8, M=6, block_size=16)
```
