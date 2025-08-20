#!/usr/bin/env python3
"""
Simple profiling script for float_round_cuda kernel with Nsight Compute
Usage: ncu --kernel-name "float_round_kernel_inplace" python profile.py [options]
"""

import torch
import argparse

try:
    from floating_point import inplace as float_round_cuda_inplace
except ImportError:
    print("Error: CUDA extension not found. Run: python setup.py build_ext --inplace")
    exit(1)

def generate_data(data_type: str, size: int) -> torch.Tensor:
    """Generate test data"""
    if data_type == 'uniform':
        return torch.rand(size, device='cuda') * 200.0 - 100.0
    elif data_type == 'normal':
        return torch.randn(size, device='cuda') * 10.0
    elif data_type == 'small':
        return torch.rand(size, device='cuda') * 1e-6
    elif data_type == 'large':
        return torch.rand(size, device='cuda') * 1e6 + 1e6
    elif data_type == 'mixed':
        data = torch.randn(size, device='cuda')
        data[torch.rand(size, device='cuda') < 0.3] *= 1e-6
        data[torch.rand(size, device='cuda') < 0.2] *= 1e6
        return data
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def main():
    parser = argparse.ArgumentParser(description='Profile float_round_cuda kernel')
    parser.add_argument('--size', type=int, default=1_000_000_001, help='Input size')
    parser.add_argument('--exp-bits', type=int, default=4, help='Exponent bits')
    parser.add_argument('--mant-bits', type=int, default=3, help='Mantissa bits')
    parser.add_argument('--bias', type=int, default=4, help='Bias')
    parser.add_argument('--data-type', type=str, default='uniform',
                       choices=['uniform', 'normal', 'small', 'large', 'mixed'])
    parser.add_argument('--warmup', type=int, default=0, help='Warmup runs')
    parser.add_argument('--runs', type=int, default=3, help='Profiling runs')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Size: {args.size:,}, Exp: {args.exp_bits}, Mant: {args.mant_bits}, Bias: {args.bias}")
    print(f"Data: {args.data_type}, Memory: {args.size * 4 / 1024 / 1024:.1f} MB")
    
    data = generate_data(args.data_type, args.size)
    
    # Warmup
    for _ in range(args.warmup):
        test_data = data.clone()
        float_round_cuda_inplace(test_data, args.exp_bits, args.mant_bits, args.bias)
    
    torch.cuda.synchronize()
    print("Warmup complete. Starting profiling...")
    
    # Profiling runs
    for i in range(args.runs):
        test_data = data.clone()
        float_round_cuda_inplace(test_data, args.exp_bits, args.mant_bits, args.bias)
        if (i + 1) % 10 == 0:
            print(f"Run {i + 1}/{args.runs}")
    
    print("Profiling complete!")

if __name__ == "__main__":
    main()
