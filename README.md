# Torch Floating Point

A PyTorch library for custom floating point quantization with autograd support. This library provides efficient implementations of custom floating point formats with automatic differentiation capabilities.

## Features

- **Custom Floating Point Formats**: Support for arbitrary floating point configurations (sign bits, exponent bits, mantissa bits, bias)
- **Autograd Support**: Full PyTorch autograd integration for training with quantized weights
- **CUDA Support**: GPU acceleration for both forward and backward passes
- **Multiple Precision**: Support for various bit widths (4-bit, 8-bit, 16-bit, 32-bit)
- **Straight-Through Estimator**: Gradient-friendly quantization for training
- **Comprehensive Testing**: Extensive test suite covering differentiability and accuracy

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

### Development Installation

```bash
git clone https://github.com/SamirMoustafa/torch-floating-point.git
cd torch-floating-point
pip install -e ".[dev,test]"
pre-commit install
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

## Usage Examples

### Custom Floating Point Configuration

```python
from floating_point import FloatingPoint

# 4-bit floating point (1 sign, 2 exponent, 1 mantissa)
fp4 = FloatingPoint(sign_bits=1, exponent_bits=2, mantissa_bits=1, bias=1, bits=4)

# 8-bit floating point with custom max mantissa
fp8_custom = FloatingPoint(
    sign_bits=1, 
    exponent_bits=4, 
    mantissa_bits=3, 
    bias=7, 
    bits=8,
    max_mantissa_at_max_exponent=6,  # Custom max mantissa
    reserved_exponent=False  # No reserved exponent for inf/nan
)

# 16-bit floating point (standard)
fp16 = FloatingPoint(sign_bits=1, exponent_bits=5, mantissa_bits=10, bias=15, bits=16)
```

### Training with Quantized Weights

```python
import torch
import torch.nn as nn
from floating_point import FloatingPoint, Round

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, fp_config):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.rounder = Round(fp_config)
    
    def forward(self, x):
        quantized_weight = self.rounder(self.weight)
        return torch.nn.functional.linear(x, quantized_weight)

# Define quantization format
fp8 = FloatingPoint(sign_bits=1, exponent_bits=4, mantissa_bits=3, bias=7, bits=8)

# Create model with quantized weights
model = QuantizedLinear(784, 10, fp8)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    # ... your training code ...
    loss.backward()
    optimizer.step()
```

### Direct Function Usage

```python
import torch
from floating_point import autograd

# Direct quantization function
x = torch.randn(100, requires_grad=True)
quantized = autograd(x, exponent_bits=4, mantissa_bits=3, bias=7)

# Gradients work automatically
loss = quantized.sum()
loss.backward()
```

## Supported Formats

The library supports various floating point formats:

| Format | Sign Bits | Exponent Bits | Mantissa Bits | Bias | Total Bits |
|--------|-----------|---------------|---------------|------|------------|
| FP4    | 1         | 2             | 1             | 1    | 4          |
| FP8    | 1         | 4             | 3             | 7    | 8          |
| FP16   | 1         | 5             | 10            | 15   | 16         |
| BF16   | 1         | 8             | 7             | 127  | 16         |
| FP32   | 1         | 8             | 23            | 127  | 32         |

## Development

### Testing

The project includes two testing approaches:

1. **CI/CD Tests** (GitHub Actions): Fast, lightweight tests that verify core functionality without heavy numerical computations
2. **Full Test Suite**: Complete test coverage including all numerical precision tests (run locally or via manual workflow trigger)

To run the full test suite locally:
```bash
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__file__)")/lib:$LD_LIBRARY_PATH
python -m pytest test/round.py test/data_types.py -v
```

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
python -m pytest test/round.py -v
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Run all checks
make full-check
```

### Building

```bash
# Build extension
cd floating_point && python setup.py build_ext --inplace

# Build package
make build

# Clean build artifacts
make clean
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
@software{torch_floating_point,
  title={Torch Floating Point: A PyTorch library for custom floating point quantization},
  author={Samir Moustafa},
  year={2024},
  url={https://github.com/SamirMoustafa/torch-floating-point}
}
```

## Acknowledgments

- PyTorch team for the excellent autograd system
- The PyTorch C++ extension community for guidance on extension development
- Contributors and users of this library

## Support

- **Issues**: [GitHub Issues](https://github.com/SamirMoustafa/torch-floating-point/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SamirMoustafa/torch-floating-point/discussions)
- **Email**: samir.moustafa.97@gmail.com
