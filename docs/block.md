# Block scale

Shared per-block scale: \(y = (e - z)\,s\,s_{\mathrm{global}}\) with \(e = \mathrm{round}(x / (s\,s_{\mathrm{global}}) + z)\). The library takes a `BlockFormat` — two `FloatingPoint` codebooks, block geometry, \(M\), and a named scale-encode policy. It does not ship named hardware packings. Byte layouts, fused MMA, and vendor SDKs are out of scope; this is numeric reconstruct only. The [Explorer](explorer.md) plots that reconstruct for any recipe.

[OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) ([Rouhani et al., 2023](https://arxiv.org/abs/2310.10537)) uses UE8M0 scales over blocks of 32. **NVFP4** uses block 16, E4M3 (UE4M3) micro-scales, and an optional FP32 tensor scale ([NVIDIA, 2025](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)). See [Formats](formats.md) for constructors. The tables below are how to *match* those packings, not a catalog of exports.

The element map is a `Round` subclass (`rounder=`, default stock STE). Absmax detaches \(s\). Pass `scales=` for learnable QAT scales. Pass `s_global=` (or set `BlockFormat.s_global`) for a second-level tensor scale.

```python
from floating_point import BlockFormat, BlockRound, FloatingPoint

e2m1 = FloatingPoint(1, 2, 1, 1, 4, reserved_exponent=False)
e4m3 = FloatingPoint(1, 4, 3, 7, 8, max_mantissa_at_max_exponent=6, reserved_exponent=False)
ue8m0 = FloatingPoint(0, 8, 0, 127, 8, reserved_exponent=True)

nvfp4 = BlockFormat(e2m1, e4m3, 16, 6.0, "nearest")
mxfp8 = BlockFormat(e4m3, ue8m0, 32, 448.0, "ue8m0_ceil")
mxfp4 = BlockFormat(e2m1, ue8m0, 32, 6.0, "ue8m0_ceil")

y = BlockRound(nvfp4)(x)  # absmax scales, STE on x only
y = BlockRound(nvfp4)(x, scales=s)  # gradients into scales
y = BlockRound(nvfp4)(x, s_global=tensor_scale)
y = BlockRound(mxfp8)(x)
y = BlockRound(mxfp4, rounder=MyRound)(x)
```

## `BlockFormat` fields

| Field | Purpose | Default |
| --- | --- | --- |
| `elem_fp`, `scale_fp`, `M` | element and scale codebooks; \(M\) is the element max used in absmax | required |
| `block_size` | `int` (1-D) or `(h, w)` (2-D tiles) | required |
| `scale_encode` | named policy (closed set) | required |
| `dims` | axes that form the block | `(-1,)` (last dim). 2-D `block_size` defaults to `(-2, -1)` |
| `s_global` | constant second-level scale; override per call with `s_global=` | `1.0` |
| `zero_point` | affine reconstruct \(y = (e - z)\,s\,s_g\) | `0.0` |
| `pad` | `"error"` or `"zero"` if a dim is not divisible | `"error"` |

`scale_encode` is a string, not a callable (so a C++ kernel can stay nearest-only):

| Name | Formula | Used by |
| --- | --- | --- |
| `nearest` | \(s = \mathrm{Round}(\mathrm{amax}/M)\) into `scale_fp` | NVFP4 micro-scale |
| `ue8m0_ceil` | \(s = 2^{\lceil\log_2(\mathrm{amax}/M)\rceil}\) (float `ceil(log2)`, not TE `float_to_e8m0`) | NVIDIA TE / CUDA MX ([cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#element-1d-block-scaling-for-fp8-and-fp4-data-types), [TE MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)) |
| `ue8m0_floor` | \(s = 2^{\lfloor\log_2(\mathrm{amax}/M)\rfloor}\) (floor twin of `ue8m0_ceil`, not OCP §6.3) | same UE8M0 path as ceil |
| `ocp_floor` | \(X = 2^{\lfloor\log_2 a_{\max}\rfloor - e_{\max}}\) | OCP MX v1.0 §6.3 sample ([PDF](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [arXiv:2310.10537](https://arxiv.org/abs/2310.10537)) |
| `ocp_floor_x2` | \(X = 2^{\lfloor\log_2 a_{\max}\rfloor - (e_{\max}-1)}\) | AWS `quantize_mx` ([NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.quantize_mx.html)) |
| `amax_over_M` | \(s = \mathrm{amax}/M\) clamped to `scale_fp` range (no codebook snap) | Hopper / Qwix / Ironwood software FP32 scales |
| `signed_peak` | \(s = \mathrm{Round}(\mathrm{peak}/(-M))\) (sign of the max-magnitude value) | KleidiAI Q4_0 / `qsi4c32` |

Row vs column 1-D blocks: `dims=(-1,)` (last) or `dims=(-2,)` (chosen axis). 1×k / k×1 tiles: `block_size=(1, k)` or `(k, 1)`.

## Recipe catalog

One `BlockFormat(...)` per row. If a vendor’s block length \(k\) is unpublished, there is no fixture — do not invent \(k\).

### Last-dim \(k=32\), \(y=e\times s\), UE8M0 (OCP-shaped)

| Recipe | Encode | Source |
| --- | --- | --- |
| MXFP8 E4M3 / E5M2 | `ue8m0_ceil` (NVIDIA) vs `ocp_floor` (spec) vs `ocp_floor_x2` (AWS) | [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [TE MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html), [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf), [NKI quantize_mx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.quantize_mx.html) (**Trn3 / NC-v4 only**), [Huawei MxMatmul](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10029.html), [Maia 200](https://arxiv.org/abs/2608.24664) (compose, not a native type), [tt-metal MX](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/data_format/mx_common.hpp) (**Quasar pre-silicon**; face-major 2×16 ≠ last-dim 32) |
| MXFP6 E2M3 / E3M2 | same scale family | [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (Qualcomm Cloud AI **decompresses to FP16** — [MX blog](https://quic.github.io/cloud-ai-sdk-pages/latest/blogs/Microscaling/microscaling/); numeric reconstruct only) |
| MXFP4 E2M1 | `ue8m0_ceil` / `ocp_floor` | [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [CUTLASS Blackwell](https://docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/blackwell_functionality.html) |
| MXFP4 E1M2 | extra vs OCP Table 1 | [Maia 200](https://arxiv.org/html/2608.24664), [Huawei MxMatmul](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10029.html) |
| MXINT8 | E8M0, implicit \(2^{-6}\) | [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf); d-Matrix names MXINT but **k unpublished** — skip packing ([HC2025 PDF](https://www.d-matrix.ai/wp-content/uploads/2025/10/hc2025.dmatrix.SudeepBhoja.v01.pdf)) |

```python
mxfp8_ocp = BlockFormat(e4m3, ue8m0, 32, 448.0, "ocp_floor")
mxfp8_aws = BlockFormat(e4m3, ue8m0, 32, 448.0, "ocp_floor_x2")
mxint8 = BlockFormat(FloatingPoint(1, 0, 7, 0, 8), ue8m0, 32, 127 / 64, "ocp_floor")
```

### Last-dim \(k=16\), \(y=e\times s\)

| Recipe | Notes | Source |
| --- | --- | --- |
| NVFP4 | E2M1 × UE4M3 + **`s_global`** FP32 | [NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [TE NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) |
| Tensix BFP | 16-el **shared exp**, mag not minifloat \(e\) | [tt-isa FloatBitPatterns](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md) |
| MLX `nvfp4` | software | [mlx.core.quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) |
| TPU 8t MXFP8-16 | **forum only — no fixture** | [forum](https://discuss.google.dev/t/inside-the-optimization-of-fp8-training-on-ironwood/336681/13) vs [8t blog](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |

```python
nvfp4 = BlockFormat(e2m1, e4m3, 16, 6.0, "nearest")  # pass s_global= at call time
bfp8 = BlockFormat(FloatingPoint(1, 0, 7, 0, 8), ue8m0, 16, 127 / 64, "ocp_floor")
```

### Other 1-D block lengths (FP32 / affine scales)

| Recipe | `block_size` | Source |
| --- | --- | --- |
| Hopper / Furiosa FP8 128 or 128×128 | `128` or `(128, 128)` | [TE FP8 blockwise](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html), [Furiosa-LLM](https://developer.furiosa.ai/latest/en/furiosa_llm/model-preparation.html) |
| Qwix / Ironwood subchannel | caller `tile_size` (staff: 512); production is **per-axis** | [Ironwood performance](https://docs.cloud.google.com/tpu/docs/ironwood-performance), [Qwix](https://github.com/google/qwix) |
| Huawei PER_GROUP affine | `groupSize` multiple of 32, **fp16 scales not UE8M0** | [AscendQuant](https://asc.gitcode.com/api/SIMD-API/adv_api/quantization/AscendQuant.html) |
| KleidiAI qsi4c32 / qsi8d32 | 32, BF16/F16 scales, `signed_peak` | [KleidiAI tables](https://github.com/ARM-software/kleidiai/blob/main/docs/microkernel_tables.md) |
| Core ML / ANE INT4 block | default 32, affine `s, z` | [coremltools blockwise](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html) |
| MLX mxfp4/mxfp8 | 32, E8M0, software | [MLX quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) |

```python
hopper_fp8 = BlockFormat(e4m3, ue8m0, 128, 448.0, "amax_over_M")
q4_0 = BlockFormat(e2m1, e4m3, 32, 8.0, "signed_peak")  # pair with an INT4 rounder; see tests
```

### Per-tensor / per-channel / per-token (block = full axis)

Gaudi pow2 ([Habana FP8](https://docs.habana.ai/en/v1.23.0/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)), Graphcore `fp8Scale` ([Poplar types](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/supported-types.html)), Tesla CFloat8 bias ([Dojo PDF](https://digitalassets.tesla.com/tesla-contents/image/upload/tesla-dojo-technology.pdf)), Inf2/Trn2 cFP8 ([NeuronCore-v2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v2.html)), Arm OFP8 `FPMR.LSCALE` ([Arm FP8](https://developer.arm.com/documentation/102374/latest/Data-processing---floating-point/Support-for-8-bit-and-16-bit-floating-point)) — simulate as `block_size = axis_length` (or `Round` only).

### 2-D tiles

| Recipe | Shape | Source |
| --- | --- | --- |
| TE NVFP4 2D policy | 16×16 | [TE NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) |
| Hopper 128×128 | 128×128 | [TE FP8 blockwise](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html) |
| Gaudi MoE | example 30 — `block_size=(30, 30)` | [Gaudi custom ops](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Custom_Ops_PyTorch.html) |

Tensix / Groq / TPU MXU **32×32 / 320×320 / 256×256** are MMA tiles, not scale groups — do not use them as MX `block_size`.

```python
nvfp4_2d = BlockFormat(e2m1, e4m3, (16, 16), 6.0, "nearest")
hopper_2d = BlockFormat(e4m3, ue8m0, (128, 128), 448.0, "amax_over_M")
gaudi_moe = BlockFormat(e4m3, ue8m0, (30, 30), 448.0, "amax_over_M")
```

### Two-level / pair scales (not \(y=e\times s\) over 16)

ISCA MX9/6/4 \(k_1=16\), \(k_2=2\) ([arXiv:2302.08007](https://arxiv.org/abs/2302.08007)) — not implemented (no \(s_{\mathrm{sub}}\in\{1,1/2\}\) hook). Maia 100 packing unpublished ([Hot Chips 2024 PDF](https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf)). NVFP4’s FP32 `s_global` **is** implemented.

### Not simulated (unpublished \(k\) or not block-scale FP)

MTIA MX8/MX4 packing ([Meta blog](https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/)); TPU 8t MX \(k\); Jalapeño \(k\) ([STH HC2026](https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/)); Hailo group \(k\); Qualcomm MXFP6 DRAM \(k\); Cerebras W4 storage; Groq 3 FP8 packing; Cambricon / SN40L / analog CIM MX.

## Absmax vs learnable

- **Absmax** (`scales is None`): `s = encode(stat).detach()` — straight-through on `x` only. `stat` is block absmax, or the signed peak when `scale_encode="signed_peak"`.
- **Learnable** (`scales=`): same element round; `y = (e - z) * s * s_g` so `s` gets gradients. See [Autograd](autograd.md). Override the element map with `rounder=` — [Example](block-dasr.md).

`return_aux=True` yields `(y, s, elems)`.
