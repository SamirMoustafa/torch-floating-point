#!/usr/bin/env python3
"""
Setup script for torch-floating-point
"""

# Import version from the project root
import os
import platform
import sys
from os import environ, path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setuptools import find_packages, setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from wheel.bdist_wheel import bdist_wheel

from version import __version__

__HERE__ = path.dirname(path.abspath(__file__))

# Get the long description from the README file
try:
    with open(path.join(__HERE__, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A PyTorch library for custom floating point quantization with autograd support."

# Automatically detect and set CUDA architectures
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
    print(f"Setting TORCH_CUDA_ARCH_LIST={environ['TORCH_CUDA_ARCH_LIST']}")

extra_compile_args = {"cxx": ["-fopenmp" if platform.system() != "Windows" else "/openmp"]}
extra_link_args = ["-fopenmp"] if platform.system() != "Windows" else []

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
    extra_compile_args["nvcc"] = ["-O2"]
else:
    print("No CUDA detected, building without CUDA support.")
    extension_class = CppExtension

setup(
    name="torch-floating-point",
    version=__version__,
    description="A PyTorch library for custom floating point quantization with autograd support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Samir Moustafa",
    author_email="samir.moustafa.97@gmail.com",
    url="https://github.com/SamirMoustafa/torch-floating-point",
    install_requires=["torch>=2.4.0,<2.6.0"],
    packages=find_packages(),
    ext_modules=[
        extension_class(
            name="floating_point.floating_point",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
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
