<div align="center">

<h1> Torch Floating Point</h1>
<img src="assets/torch-floating-point-logo.png"/>

![python-3.10](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch-1.13.1](https://img.shields.io/badge/torch-2.4.1%2B-orange)
![release-version](https://img.shields.io/badge/release-0.1-green)
![license](https://img.shields.io/badge/license-GPL%202-red)
</div>

A PyTorch library for custom floating point quantization with autograd support. This library provides efficient implementations of custom floating point formats with automatic differentiation capabilities.

## Features

- **Custom Floating Point Formats**: Support for arbitrary floating point configurations (sign bits, exponent bits, mantissa bits, bias)
- **Autograd Support**: Full PyTorch autograd integration for training with quantized weights
- **CUDA Support**: GPU acceleration for both forward and backward passes
- **Straight-Through Estimator**: Gradient-friendly quantization for training

## Installation

### From PyPI (Recommended)

```bash
pip install torch-floating-point
```

### From Source

```bash
git clone https://github.com/SamirMoustafa/torch-floating-point.git
cd torch-floating-point
pip install -e .
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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`make setup-dev`)
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
@software{moustafa2025torchfloatingpoint,
  title={Torch Floating Point: A PyTorch library for custom floating point quantization},
  author={Samir Moustafa},
  year={2025},
  url={https://github.com/SamirMoustafa/torch-floating-point}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/SamirMoustafa/torch-floating-point/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SamirMoustafa/torch-floating-point/discussions)
- **Email**: samir.moustafa.97@gmail.com
