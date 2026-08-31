---
title: 'Torch Floating Point: a PyTorch library for custom floating-point formats with automatic differentiation'
tags:
  - Python
  - PyTorch
  - floating point
  - quantization
authors:
  - name: Samir Moustafa
    orcid: 0000-0002-0674-9667
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: Faculty of Computer Science, University of Vienna, Vienna, Austria
    index: 1
    ror: 03prydq77
  - name: CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences, Vienna, Austria
    index: 2
    ror: 02z2dfb58
date: 30 August 2026
bibliography: paper.bib
---

# Summary

Torch Floating Point (`torch-floating-point`) is a PyTorch library that rounds a tensor onto a user-chosen numeric layout inside autograd. The forward map is a staircase. The user specifies a layout as a `FloatingPoint` object, including floating-point and integer grids. Element-wise `Round` maps a tensor onto that layout. The default operator uses CPU kernels, or CUDA kernels compiled from source when a GPU is present, for nearest rounding and a clipped straight-through estimator (STE) in autograd [@Bengio2013STE]. Both the forward round and the backward map are replaceable by subclassing `Round`. A second operator, `BlockRound`, reconstructs values that share a scale over a block, so the same layouts can emulate published recipes such as Open Compute Project microscaling (MX) [@Rouhani2023MX] and NVIDIA NVFP4 [@NVIDIA2025NVFP4]. Native PyTorch dtypes are a named catalog. They are not an arbitrary user-specified layout.

```python
from floating_point import FloatingPoint, Round
# user-chosen floating-point or integer layout
fp = FloatingPoint(...)
y = Round(fp)(x)
```

# Statement of need

Native PyTorch dtypes (`float32`, `float16`, `bfloat16`, `float8_e4m3fn`, and related names) cover a handful of encodings, and it is easy to treat that catalog as the research space. Researchers who compare layouts still need formats that are not native dtypes. Those include sweeps over encodings outside the catalog, Open Compute 8-bit floating-point (OFP8) variants [@Micikevicius2022FP8], integer grids, and one-off encodings that a specification documents but PyTorch does not ship. The same need appears for block-scaled recipes, which differ between MX [@Rouhani2023MX] and NVFP4 [@NVIDIA2025NVFP4]. That work has to stay inside autograd, because the round is a quantization-aware training (QAT) operator rather than a one-shot cast. Training uses a straight-through estimator (STE) [@Bengio2013STE]. Layout studies therefore need a parametric object, not a frozen list of named dtypes.

That parametric object is useful only if the round stays in autograd, with a default STE and a user-replaceable forward and backward, and reconstructs a shared block scale. The packages named here do not combine native CPU or CUDA forward kernels with a two-level block reconstruct. Those packages cluster by language stack, by whether they train through a minifloat round, and by whether they expose a production catalog rather than a parametric layout.

# State of the field

The packages most relevant here cluster into three groups.

Parametric IEEE-style simulators exist, but they live in Julia, MATLAB, C, or NumPy rather than in PyTorch QAT. MicroFloatingPoints.jl provides parametric minifloats for interactive Julia work [@Goualard2024MicroFloatingPoints]. MATLAB `chop` and the C library CPFloat round arrays to custom formats for numerical analysis [@Higham2019Chop; @Fasi2023CPFloat]. Graphcore `gfloat` is a readable Python encode/decode of IEEE, OFP8, and MX layouts [@gfloat], and Google `ml_dtypes` supplies NumPy catalog dtypes for E4M3-FN, E2M1, E8M0, and related encodings [@mldtypes]. These tools are the right place to inspect a layout or to check a decode table. None of them ships a PyTorch autograd `Round` with CPU/CUDA forward kernels, or a two-level MX/NVFP4 `BlockRound`.

On the PyTorch training axis, QPyTorch introduced `FloatingPoint(exp, man)` with fused CUDA kernels and autograd [@Zhang2019QPyTorch], and MPTorch is the living library in that family, with per-operator exponent/mantissa widths and CUDA backends [@MPTorch]. Both let a researcher train through a minifloat round. They can set exponent and mantissa widths, but they do not implement OFP8 finite-max encodings, and they do not provide a two-level block-scale path. Brevitas provides parametric floating-point quantizers, an STE, and MX-style groupwise reconstruct in PyTorch QAT [@Brevitas]. It lacks native round kernels and this library's two-level block reconstruct.

Production MX and FP8 stacks cover named catalog formats. Microsoft `microxcaling` is the reference PyTorch MX emulation [@microxcaling]. PyTorch `torchao` exposes native float8 and MX dtypes as composable tensor types [@torchao]. NVIDIA Transformer Engine ships FP8, MXFP8, and NVFP4 layers for training at hardware-supported widths [@TransformerEngine]. Those packages are the right place to run a fused matrix multiply on a named format. The user chooses from that catalog and cannot design a layout of their own.

Forking QPyTorch would still leave OFP8 finite-max encodings and a two-level block scale to be added. Forking Brevitas would fight its job as a QAT quantizer stack. Landing a parametric layout object inside Transformer Engine or `torchao` would cut against their role as production catalogs. Adding PyTorch QAT to MicroFloatingPoints.jl would mean changing language and purpose. `torch-floating-point` is a new package because it is the PyTorch emulator that spans both axes. It combines parametric layouts with the block-scaled recipes those layouts are used with.

# Software design

A package that spans a parametric layout and block-scaled reconstruction cannot be a frozen dtype enum. `FloatingPoint` is a value object for a user-specified numeric layout, including floating-point and integer grids. Named formats live in documentation and tests, not as package exports, so a variant is a layout change rather than an API fork. The cost of that choice is that users write layouts explicitly. The gain is that an encoding that is not yet a native PyTorch dtype does not require a library release. The round onto that object has to stay in autograd.

Nearest rounding is a staircase with a null derivative almost everywhere, so `Round` is a `torch.autograd.Function` whose backward uses a surrogate. The default forward clamps to the layout range and nearest-rounds onto that codebook with CPU or CUDA kernels.

$$
y = \mathrm{Round}(x).
$$

The default backward is a clipped STE, the identity through the staircase [@Bengio2013STE] and zero outside $[\min, \max]$. Clipping is the usual QAT surrogate, so saturation is not an interior point [@Esser2020LSQ]. Users subclass `Round` to change the round, the surrogate gradient, or both. Keeping the kernel means a new autograd `Function` plus a `Round` subclass that calls it. `BlockRound` takes that subclass as `rounder=`. The kernels implement one forward round. They are not a fused matrix-multiply path. Element-wise rounding is not enough once a block shares a scale.

`BlockRound` is a second path. A `BlockFormat` pairs an element layout with a scale layout, a block geometry, and optional extra scales.

$$
e = \mathrm{Round}\!\left(\frac{x}{S} + z\right), \qquad y = (e - z)\,S,
$$

where \(S\) is a product of shared scales. The implementation uses a per-block scale and an optional tensor scale. Then \(S = s\) recovers MX. Published recipes including MX, NVFP4, integer groups, and later hardware choose the element codebook, the scale codebook, block shape, and \(z\). They do not change this reconstruct. Absmax scales detach so gradients train the payload through the element rounder. Callers may pass learnable scales when they want scale gradients. Python decode is checked against NVIDIA CUDA decode headers and documented encodings, not against silicon throughput.

# Research impact statement

NVIDIA `__nv_fp4` / `__nv_fp8` tables pin the Python decode path. Round kernels are checked against torch dtype casts and committed self-goldens when those tests are collected. The codec check collected in GitHub Actions is the explorer Python and C++ self-check, not the NVIDIA per-code suite. GitHub Actions tests the CPU extension when those tests are collected. CUDA kernels are in the source tree and are not claimed as CI-validated. The library is on PyPI as `torch-floating-point`. The published wheel is a CPU Linux wheel. Other platforms compile the extension from source. Documentation and an interactive format explorer are on Read the Docs. The author uses the package to emulate layouts that are not native PyTorch dtypes. The repository ships an OSI MIT license, a `CITATION.cff` file, contributing guidelines, and a public test suite. External publications that depend on this specific library are not yet the evidence. The near-term significance is a reproducible emulator, goldens, and documentation that other groups can rerun rather than an accuracy claim against FP32 or against production FP8 stacks.

# AI usage disclosure

Generative AI (Cursor) was used only for documentation and the software docs. It was not used to design or implement `FloatingPoint`, `Round`, or `BlockRound`. The author specified those numerical designs and verified all correctness claims through tests.

# Acknowledgements

I thank Wilfried Gansterer for discussions and for the research environment in which this work was initiated.

# References
