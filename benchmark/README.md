# Float Round CUDA Kernel Benchmarking

Simple guide for profiling the `float_round_cuda` kernel with Nsight Compute.

## Directory Structure

```
benchmark/
├── profile.py      # Main profiling script
└── README.md       # This file
```

## Quick Start

### Basic Profiling
```bash
# Run with default settings (1M elements, 4 exp bits, 3 mant bits)
ncu --set full --kernel-name "float_round_kernel_inplace" python profile.py
```

### Custom Configuration
```bash
# Profile with different parameters
ncu --set full --kernel-name "float_round_kernel_inplace" \
    python profile.py --size 2000000 --exp-bits 5 --mant-bits 4 --data-type mixed
```