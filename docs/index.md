---
hide:
  - toc
---

<div class="hero" markdown>

![Torch Floating Point](assets/banner.svg){ class="hero-logo" }

# Custom floating point, in PyTorch

<p class="lede">Quantize tensors to arbitrary FP layouts — including OFP8 / MX FP4 and FP8 — with CUDA kernels and autograd.</p>

</div>

<div class="grid cards" markdown>

-   **Any layout**

    Sign, exponent, mantissa, bias, and reserved-NaN behavior are yours to set.

-   **Trainable**

    `Round` uses a straight-through estimator so gradients flow through quantization.

-   **Block-scaled**

    NVFP4 and MXFP8 via shared per-block scales.

</div>

## Install

=== "PyPI"

    ```bash
    pip install torch-floating-point
    ```

=== "From source"

    ```bash
    git clone https://github.com/SamirMoustafa/torch-floating-point.git
    cd torch-floating-point
    pip install -e .
    ```

Requires **Python 3.10+** and **PyTorch 2.4+**. CUDA is used when the install machine has a GPU; otherwise a CPU extension is built.

## Quick start

```python
import torch
from floating_point import FloatingPoint, Round

fp8 = FloatingPoint(sign_bits=1, exponent_bits=4, mantissa_bits=3, bias=7, bits=8)
x = torch.randn(8, requires_grad=True)

y = Round(fp8)(x)
y.sum().backward()
```

Train with quantized weights using the same rounder:

```python
import torch.nn as nn
from floating_point import FloatingPoint, Round

class FloatPointLinear(nn.Module):
    def __init__(self, inn, out, fp):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, inn))
        self.bias = nn.Parameter(torch.randn(out))
        self.rounder = Round(fp)

    def forward(self, x):
        return nn.functional.linear(x, self.rounder(self.weight), self.bias)
```

Next: [Formats](formats.md) · [block scaling](block.md) · [Autograd](autograd.md) · [API](api.md)

## Cite

```bibtex
@software{moustafa2025torchfloatingpoint,
  title={Torch Floating Point: A PyTorch library for custom floating point quantization},
  author={Samir Moustafa},
  year={2025},
  url={https://github.com/SamirMoustafa/torch-floating-point}
}
```
