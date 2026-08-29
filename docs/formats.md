# Formats

A `FloatingPoint` object is a **decode spec**: how many bits, where they sit, and how special values are encoded. `Round` then maps real tensors onto that codebook. These constructors are documentation and test fixtures — the package does not export named hardware formats.

```python
from floating_point import FloatingPoint, Round

fp = FloatingPoint(
    sign_bits=1,
    exponent_bits=4,
    mantissa_bits=3,
    bias=7,
    bits=8,  # must equal S + E + M
    max_mantissa_at_max_exponent=None,  # default: 2^M - 1
    reserved_exponent=True,  # max exponent → Inf/NaN
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

## OFP8 and MX minifloats

E4M3 and E5M2 are the [OCP OFP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) interchange encodings ([Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433)). E2M1, E2M3, E3M2, and UE8M0 are the [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) element and scale types ([Rouhani et al., 2023](https://arxiv.org/abs/2310.10537)). CUDA names in the table are aliases from `cuda_fp4.h` / `cuda_fp8.h`; this library’s decode goldens match those headers. For block-scaled \(x = e \cdot s \cdot s_{\mathrm{global}}\), see [Block scale](block.md).

| Format | Config | Notes |
| --- | --- | --- |
| E2M1 | `FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)` | MXFP4 / NVFP4 elements; max \(\pm 6\). CUDA `__nv_fp4_e2m1` |
| E3M2 | `FloatingPoint(1, 3, 2, 3, 6, reserved_exponent=False)` | MXFP6; max \(\pm 28\) |
| E2M3 | `FloatingPoint(1, 2, 3, 1, 6, reserved_exponent=False)` | MXFP6; max \(\pm 7.5\) |
| E1M2 | `FloatingPoint(1, 1, 2, 0, 4, reserved_exponent=False)` | Extra FP4 vs OCP Table 1 ([Maia 200](https://arxiv.org/abs/2608.24664), [Huawei MxMatmul](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10029.html)); max \(\pm 3.5\) |
| E4M3-FN | `FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)` | OCP / Blackwell; max \(\pm 448\); codes 127/255 are NaN. CUDA `__nv_fp8_e4m3` |
| E4M3 max 240 | `FloatingPoint(1, 4, 3, 7, 8, reserved_exponent=True)` | Inf2/Trn2 cFP8 ([Neuron data types](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/data-types.html)), Graphcore F143 ([Poplar types](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/supported-types.html)) |
| E5M2 | `FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True)` | OFP8; CUDA `__nv_fp8_e5m2` |
| UE8M0 | `FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)` | MX scale; \(2^{E-127}\); code 255 is NaN. CUDA `__nv_fp8_e8m0` |
| UE4M3 | same constructor as E4M3-FN | NVFP4 block scale: E4M3 bits, **sign ignored** ([cuBLAS block scaling](https://docs.nvidia.com/cuda/cublas/index.html#element-1d-block-scaling-for-fp8-and-fp4-data-types)). `scale_encode="nearest"` already takes \(\lvert\cdot\rvert\) |
| E4M3 FNUZ | `FloatingPoint(1, 4, 3, 8, 8, reserved_exponent=False)` | Approximate [MI300 HIP FNUZ](https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/workload.html) (bias 8). Negative-zero NaN is not modeled |
| E5M2 FNUZ | `FloatingPoint(1, 5, 2, 16, 8, reserved_exponent=False)` | Same caveat |

```python
from floating_point import FloatingPoint

fp4_e2m1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
fp6_e3m2 = FloatingPoint(1, 3, 2, 3, 6, reserved_exponent=False)
fp6_e2m3 = FloatingPoint(1, 2, 3, 1, 6, reserved_exponent=False)
fp4_e1m2 = FloatingPoint(1, 1, 2, 0, 4, reserved_exponent=False)
fp8_e4m3fn = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
fp8_e4m3_240 = FloatingPoint(1, 4, 3, 7, 8, reserved_exponent=True)
fp8_e5m2 = FloatingPoint(1, 5, 2, 15, 8, reserved_exponent=True)
fp8_e8m0 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)
fp8_e4m3_fnuz = FloatingPoint(1, 4, 3, 8, 8, reserved_exponent=False)
```

!!! note "Element-wise vs block encode"
    Element-wise `Round(fp8_e8m0)` is nearest. **Block** UE8M0 scales use a named `scale_encode` (`ue8m0_ceil`, `ue8m0_floor`, `ocp_floor`, …) — see [Block scale](block.md).

## Integer and BFP magnitude

Sign+magnitude with no exponent field is the BFP / MXINT grid: value \(= \pm \mathrm{Mag}/2^{m}\) for mantissa width \(m\). OCP MXINT8 uses an implicit \(2^{-6}\) ([OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)); Tensix BFP8 mag/64 with a shared exponent over 16 elements ([tt-isa BFP](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md)).

```python
# MXINT8 / BFP8 mag: ±k/64 for k = 0..127; M = 127/64
mxint8 = FloatingPoint(1, 0, 7, 0, 8)
```

Signed integer grids used with `rounder=` (override `forward` to `round().clamp(lo, hi)`) rather than a minifloat decode:

| Grid | Typical `M` | Notes |
| --- | --- | --- |
| INT4 \(\{-8,\ldots,7\}\) | `8` | KleidiAI `qsi4c32` / Q4_0; pair with `signed_peak` and BF16/F16-range `scale_fp` |
| INT8 \(\{-128,\ldots,127\}\) | `127` or `128` | Symmetric or two’s-complement; affine `zero_point` covers ANE / Core ML / Hailo-style offsets |

## HiF8

[HiF8](https://arxiv.org/abs/2409.16626) is a **tapered scalar** (mantissa width depends on the exponent), not a block format. A single `FloatingPoint` cannot represent it. Simulate with a custom `Round` subclass; do not put it in `BlockFormat`.

## References

- Open Compute Project. *OCP 8-bit Floating Point Specification (OFP8)*, Rev. 1.1. <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf>
- Open Compute Project. *OCP Microscaling Formats (MX) Specification*, v1.0, 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- Micikevicius, P., et al. *FP8 Formats for Deep Learning*. 2022. <https://arxiv.org/abs/2209.05433>
- Rouhani, B. D., et al. *Microscaling Data Formats for Deep Learning*. 2023. <https://arxiv.org/abs/2310.10537>
- NVIDIA. *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*. 2025. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
