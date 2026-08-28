# Formats

A `FloatingPoint` object is a **decode spec**: how many bits, where they sit, and how special values are encoded. `Round` then maps real tensors onto that codebook.

```python
from floating_point import FloatingPoint, Round

fp = FloatingPoint(
    sign_bits=1,
    exponent_bits=4,
    mantissa_bits=3,
    bias=7,
    bits=8,                      # must equal S + E + M
    max_mantissa_at_max_exponent=None,  # default: 2^M - 1
    reserved_exponent=True,      # max exponent → Inf/NaN
)
y = Round(fp)(x)
```

| Field | Role |
| --- | --- |
| `sign_bits` | `0` (unsigned) or `1` |
| `exponent_bits` / `mantissa_bits` | IEEE-style split |
| `bias` | Stored exponent minus bias is the true power of two |
| `reserved_exponent` | If true, all-ones exponent is Inf/NaN (or NaN-only for M = 0) |
| `max_mantissa_at_max_exponent` | Finite cap at the top exponent (E4M3-FN uses `6`) |

`fp.minimum`, `fp.maximum`, and `fp.values` expose the numeric range and the full codebook.

## Common layouts (OFP8 / MX)

E4M3 and E5M2 are the [OCP OFP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) interchange encodings ([Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433)). E2M1 and UE8M0 are the element and scale types in the [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) spec ([Rouhani et al., 2023](https://arxiv.org/abs/2310.10537)). The same layouts ship on NVIDIA CUDA, AMD Instinct CDNA4, Armv9.2-A FP8, and others. AMD MI300 FP8 is HIP **FNUZ**, not OCP.

CUDA names in the table are aliases from `cuda_fp4.h` / `cuda_fp8.h`; this library’s decode goldens match those headers. For block-scaled `x = e · s`, see [Block scale](block.md).

| Format | CUDA alias | Config |
| --- | --- | --- |
| E2M1 | `__nv_fp4_e2m1` | `FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)` — max finite `±6` |
| E4M3-FN | `__nv_fp8_e4m3` | `FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)` — max `±448`; codes 127/255 are NaN |
| E5M2 | `__nv_fp8_e5m2` | `FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True)` |
| UE8M0 | `__nv_fp8_e8m0` | `FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)` — `2^(E−127)`; code 255 is NaN |

```python
from floating_point import FloatingPoint

fp4_e2m1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
fp8_e4m3fn = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
fp8_e5m2 = FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True)
fp8_e8m0 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)
```

!!! note "Element-wise vs block encode"
    Element-wise `Round(fp8_e8m0)` is nearest. **Block** UE8M0 scales round **up** to the next power of two, as in OCP MX (and CUDA MX) encode.

## References

- Open Compute Project. *OCP 8-bit Floating Point Specification (OFP8)*, Rev. 1.1, 2026. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf>
- Open Compute Project. *OCP Microscaling Formats (MX) Specification*, v1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- Micikevicius, P., et al. *FP8 Formats for Deep Learning*. 2022. <https://arxiv.org/abs/2209.05433>
- Rouhani, B. D., et al. *Microscaling Data Formats for Deep Learning*. 2023. <https://arxiv.org/abs/2310.10537>
- NVIDIA. *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*. 2025. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
