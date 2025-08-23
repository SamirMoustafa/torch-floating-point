#!/usr/bin/env python3
"""
Setup script for torch-floating-point
"""

import os
import platform
import sys
from os import environ, path

# Set environment variables for fastest possible builds
environ.setdefault("MAX_JOBS", str(os.cpu_count()))  # Parallel compilation
environ.setdefault("TORCH_EXTENSION_NAME", "floating_point")
environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0;9.0+PTX")  # Default for RTX 4090

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from setuptools import find_packages, setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from wheel.bdist_wheel import bdist_wheel

from version import __version__

__HERE__ = path.dirname(path.abspath(__file__))

with open(path.join(__HERE__, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Force the ABI to match actual PyTorch libraries (always use legacy ABI for compatibility)
extra_compile_args = {
    "cxx": [
        "-fopenmp",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-O3",  # Maximum optimization level
        "-march=native",  # Use CPU-specific optimizations
        "-mtune=native",
        "-ffast-math",  # Fast math operations
        "-funroll-loops",  # Loop unrolling
        "-fomit-frame-pointer",  # Omit frame pointer for better performance
        "-DNDEBUG",  # Disable debug assertions
    ]
    if platform.system() != "Windows"
    else ["/openmp", "-D_GLIBCXX_USE_CXX11_ABI=0", "/O2", "/GL"]
}

extra_link_args = (
    [
        "-fopenmp",
        "-O3",  # Link-time optimization
    ]
    if platform.system() != "Windows"
    else []
)

# Set PyTorch library path for runtime linking
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in environ:
    environ["LD_LIBRARY_PATH"] = torch_lib_path
else:
    environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{environ['LD_LIBRARY_PATH']}"

# Base sources
sources = ["floating_point/float_round.cpp"]
define_macros = []


# Custom wheel builder to fix platform tag
class CustomWheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        # Use manylinux_2_28_x86_64 for Linux wheels
        if plat.startswith("linux"):
            plat = "manylinux_2_28_x86_64"
        return python, abi, plat


# Conditionally add CUDA support
if cuda.is_available():
    print("CUDA detected, building with CUDA support.")
    extension_class = CUDAExtension
    sources.append("floating_point/float_round_cuda.cu")
    define_macros.append(("WITH_CUDA", None))
    # Aggressive CUDA compilation optimizations for maximum speed
    extra_compile_args["nvcc"] = [
        "-O3",  # Maximum optimization level (faster than -O2)
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-use_fast_math",  # Fast math operations (significant speedup)
        "-maxrregcount=32",  # Limit register usage for better occupancy
        "-Xcompiler",
        "-fPIC",
        "--expt-relaxed-constexpr",  # Relaxed constexpr evaluation
        "--expt-extended-lambda",  # Extended lambda support
        "-gencode=arch=compute_90,code=compute_90",  # RTX 4090 specific
        "-gencode=arch=compute_90,code=sm_90",
        "-std=c++17",
        "-DNDEBUG",  # Disable debug assertions (significant speedup)
        "--ptxas-options=-v",  # Verbose PTX assembly
        "-lineinfo",  # Include line information
        "--threads",
        "0",  # Use all available threads for compilation
        "-Xcompiler",
        "-O3",  # Maximum optimization for host compiler
        "-Xcompiler",
        "-march=native",  # CPU-specific optimizations
        "-Xcompiler",
        "-mtune=native",  # CPU tuning
        "-Xcompiler",
        "-ffast-math",  # Fast math for host compiler
        "-Xcompiler",
        "-funroll-loops",  # Loop unrolling
        "-Xcompiler",
        "-fomit-frame-pointer",  # Omit frame pointer
    ]
else:
    print("No CUDA detected, building without CUDA support.")
    extension_class = CppExtension

# Create extension module
ext_module = extension_class(
    name="floating_point.floating_point",
    sources=sources,
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="torch-floating-point",
    version=__version__,
    description="A PyTorch library for custom floating point quantization with autograd support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Samir Moustafa",
    author_email="samir.moustafa.97@gmail.com",
    url="https://github.com/SamirMoustafa/torch-floating-point",
    install_requires=["torch>=2.4.0"],
    packages=find_packages(),
    ext_modules=[ext_module],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": CustomWheel},
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: Pytest",
    ],
    keywords=["pytorch", "floating-point", "quantization", "autograd", "machine-learning", "deep-learning"],
)

# CUDA compilation speedup environment variables
environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Non-blocking CUDA launches
environ.setdefault("CUDA_CACHE_DISABLE", "0")  # Keep CUDA cache enabled
environ.setdefault("CUDA_FORCE_PTX_JIT", "0")  # Use pre-compiled PTX when possible
environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # Consistent device ordering
environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Use first GPU for compilation

# NVCC specific optimizations
environ.setdefault("NVCC_FLAGS", "--threads 0 --parallel-threads 0")
environ.setdefault("NVCC_ARCH_FLAGS", "-arch=sm_90 -code=sm_90")

if cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in environ:
    arch_list = []
    for i in range(cuda.device_count()):
        capability = cuda.get_device_capability(i)
        arch = f"{capability[0]}.{capability[1]}"
        arch_list.append(arch)

    # Add PTX for the highest architecture for forward compatibility
    if arch_list:
        highest_arch = arch_list[-1]
        arch_list.append(f"{highest_arch}+PTX")

    environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
