<div align="center">

<h1> Torch Floating Point</h1>
<img src="https://raw.githubusercontent.com/SamirMoustafa/torch-floating-point/refs/heads/main/assets/torch-floating-point-logo.svg" width="20%" alt="Torch Floating Point"/>

![python-3.10](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch-1.13.1](https://img.shields.io/badge/torch-2.4.1%2B-orange)
![release-version](https://img.shields.io/badge/release-0.0.21-green)
![license](https://img.shields.io/badge/license-MIT-green)
</div>

A PyTorch library for custom floating point quantization with autograd support. This library provides efficient implementations of custom floating point formats with automatic differentiation capabilities.

## Features

- **Custom Floating Point Formats**: Support for arbitrary floating point configurations (sign bits, exponent bits, mantissa bits, bias)
- **Autograd Support**: Full PyTorch autograd integration for training with quantized weights
- **CUDA Support**: GPU acceleration for both forward and backward passes
- **Straight-Through Estimator**: Gradient-friendly quantization for training

## Installation

Install the PyTorch build you will run first (CPU or CUDA). This package compiles an extension against that torch.

### From PyPI (Recommended)

```bash
pip install torch-floating-point --no-build-isolation
```

A C++ compiler is required. For CUDA kernels, also install a CUDA toolkit that matches `torch.version.cuda` (pip's torch wheel does not include `nvcc`). Set `FORCE_CPU=1` to skip CUDA, or `FORCE_CUDA=1` to require it.

### From Source

```bash
git clone https://github.com/SamirMoustafa/torch-floating-point.git
cd torch-floating-point
pip install --no-build-isolation -e .
```

## Quick Start

```python
import torch
from floating_point import FloatingPoint, Round

# Define a custom 8-bit floating point format (1 sign, 4 exponent, 3 mantissa bits)
fp8 = FloatingPoint(sign_bits=1, exponent_bits=4, mantissa_bits=3, bias=7, bits=8)

# Create a rounding function
rounder = Round(fp8)

# Create a tensor with gradients
x = torch.randn(10, requires_grad=True)

# Quantize the tensor
quantized = rounder(x)

# Use in training (gradients flow through)
loss = quantized.sum()
loss.backward()

print(f"Original: {x}")
print(f"Quantized: {quantized}")
print(f"Gradients: {x.grad}")
```

## Training with Custom Floating Point Weights

```python
import torch
import torch.nn as nn
from floating_point import FloatingPoint, Round


class FloatPointLinear(nn.Module):
    def __init__(self, in_features, out_features, fp_config):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.rounder = Round(fp_config)

    def forward(self, x):
        quantized_weight = self.rounder(self.weight)
        return torch.nn.functional.linear(x, quantized_weight, self.bias)


# Define custom floating point format
fp8 = FloatingPoint(sign_bits=1, exponent_bits=4, mantissa_bits=3, bias=7, bits=8)

# Create model with quantized weights
model = FloatPointLinear(10, 5, fp8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Create simple data
x = torch.randn(32, 10)
y = torch.randn(32, 5)

# Training loop
for epoch in range(5):
    optimizer.zero_grad()

    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")
```

## Common layouts (OFP8 / MX)

E4M3 and E5M2 are [OCP OFP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) encodings ([Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433)). E2M1 and UE8M0 come from [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) ([Rouhani et al., 2023](https://arxiv.org/abs/2310.10537)). CUDA `__nv_*` comments below are aliases; decode goldens match `cuda_fp4.h` / `cuda_fp8.h`. AMD MI300 FP8 is HIP **FNUZ**, not OCP. For block-scaled `y = (e - z) * s * s_global`, use `BlockRound`. Constructors for FP6, E1M2, UE4M3, FNUZ, INT, and BFP mag are in the docs (not package exports).

```python
from floating_point import FloatingPoint

# E2M1  (__nv_fp4_e2m1; reserved_exponent=False)
fp4_e2m1 = FloatingPoint(sign_bits=1, exponent_bits=2, mantissa_bits=1, bias=1, bits=4, reserved_exponent=False)

# E4M3-FN (__nv_fp8_e4m3): max finite ±448; codes 127/255 are NaN
fp8_e4m3fn = FloatingPoint(
    sign_bits=1,
    exponent_bits=4,
    mantissa_bits=3,
    bias=7,
    bits=8,
    max_mantissa_at_max_exponent=6,
    reserved_exponent=False,
)

# E5M2 (__nv_fp8_e5m2)
fp8_e5m2 = FloatingPoint(sign_bits=1, exponent_bits=5, mantissa_bits=2, bias=15, bits=8, reserved_exponent=True)

# UE8M0 (__nv_fp8_e8m0): codes 0..254 → 2^(E-127); 255 → NaN
fp8_e8m0 = FloatingPoint(sign_bits=0, exponent_bits=8, mantissa_bits=0, bias=127, bits=8, reserved_exponent=True)
```

### Block-scaled Round (NVFP4 / MX)

Shared per-block scale: `y = (e - z) * s * s_global` with `e = Round_elem(x / (s * s_global) + z)`. Absmax mode detaches `s` (STE on `x` only); pass `scales=` for learnable QAT scales with gradients. `s_global=` is the optional second-level tensor scale (NVFP4).

OCP MX (MXFP8 / MXFP4) uses UE8M0 scales and `block_size=32`. **NVFP4** is NVIDIA-only: E2M1 + E4M3 (UE4M3) scales, `block_size=16`, plus FP32 `s_global` ([NVIDIA, 2025](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)). NVIDIA block UE8M0 uses `ue8m0_ceil`; the OCP sample is `ocp_floor`; AWS Trainium3 is `ocp_floor_x2`. Element-wise `Round(fp8_e8m0)` remains nearest. Recipe tables with source URLs: the docs.

```python
from floating_point import BlockFormat, BlockRound

nvfp4 = BlockFormat(fp4_e2m1, fp8_e4m3fn, 16, 6.0, "nearest")
mxfp8 = BlockFormat(fp8_e4m3fn, fp8_e8m0, 32, 448.0, "ue8m0_ceil")
mxfp8_ocp = BlockFormat(fp8_e4m3fn, fp8_e8m0, 32, 448.0, "ocp_floor")

y = BlockRound(nvfp4)(x)  # absmax scales, STE on x only
y = BlockRound(nvfp4)(x, scales=learnable_s)  # grad into scales
y = BlockRound(nvfp4)(x, s_global=tensor_scale)
y = BlockRound(mxfp8)(x)
y = BlockRound(nvfp4, rounder=MyRound)(x)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests, and pull-request expectations. Short version:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`make env`)
4. Make your changes
5. Run tests (`make test`)
6. Run linting (`make lint`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{moustafa2026torchfloatingpoint,
  title={Torch Floating Point: a PyTorch library for custom floating-point formats with automatic differentiation},
  author={Samir Moustafa},
  year={2026},
  version={0.0.21},
  url={https://github.com/SamirMoustafa/torch-floating-point}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/SamirMoustafa/torch-floating-point/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SamirMoustafa/torch-floating-point/discussions)
- **Email**: samir.moustafa.97@gmail.com
