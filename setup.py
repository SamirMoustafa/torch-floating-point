#!/usr/bin/env python3
"""
Setup script for torch-floating-point
"""

import os
import platform
import subprocess
import sys
from os import environ, path

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


def detect_cpu_flags():
    """Auto-detect CPU-specific optimization flags"""
    flags = []

    try:
        # Check CPU architecture
        if platform.machine() == "x86_64":
            # Check for specific CPU features
            result = subprocess.run(
                ["grep", "-m1", "flags", "/proc/cpuinfo"], check=False, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cpu_flags = result.stdout.lower()

                # Add architecture-specific optimizations
                flags.extend(["-march=native", "-mtune=native"])

                # Check for AVX support
                if "avx512" in cpu_flags:
                    flags.extend(["-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl"])
                elif "avx2" in cpu_flags:
                    flags.extend(["-mavx2", "-mfma"])
                elif "avx" in cpu_flags:
                    flags.append("-mavx")

                # Check for SSE support
                if "sse4.2" in cpu_flags:
                    flags.append("-msse4.2")
                elif "sse4.1" in cpu_flags:
                    flags.append("-msse4.1")

                # Check for FMA support
                if "fma" in cpu_flags:
                    flags.append("-mfma")

                print(f"CPU optimization flags detected: {flags}")
            else:
                # Fallback to safe optimizations
                flags.extend(["-march=native", "-mtune=native"])
                print("Using fallback CPU optimization flags")
        else:
            # Non-x86 architecture, use safe defaults
            print(f"Non-x86 architecture detected: {platform.machine()}, using safe optimizations")

    except Exception as e:
        print(f"CPU detection failed: {e}, using safe defaults")
        flags = []

    return flags


def detect_cuda_flags():
    """Auto-detect CUDA-specific optimization flags"""
    flags = []

    if not cuda.is_available():
        return flags

    try:
        # Get CUDA device count and capabilities
        device_count = cuda.device_count()
        if device_count > 0:
            # Get the first device for compilation optimization
            device = 0
            capability = cuda.get_device_capability(device)
            compute_cap = f"{capability[0]}.{capability[1]}"

            # Add compute capability specific flags
            flags.extend(
                [
                    f"-gencode=arch=compute_{capability[0]}{capability[1]},code=compute_{capability[0]}{capability[1]}",
                    f"-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}",
                ]
            )

            # Note: PTX forward compatibility is handled by TORCH_CUDA_ARCH_LIST

            # Add optimization flags based on compute capability
            if capability[0] >= 8:  # Ampere and newer
                flags.extend(
                    [
                        "-O3",  # Maximum optimization
                        "-use_fast_math",  # Fast math operations
                        "-maxrregcount=32",  # Limit register usage
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                    ]
                )
            elif capability[0] >= 7:  # Volta and Turing
                flags.extend(["-O3", "-use_fast_math", "-maxrregcount=32"])
            else:  # Older architectures
                flags.extend(["-O2", "-use_fast_math"])

            print(f"CUDA optimization flags for compute capability {compute_cap}: {flags}")

    except Exception as e:
        print(f"CUDA detection failed: {e}, using safe defaults")
        flags = ["-O2", "-use_fast_math"]

    return flags


def detect_system_flags():
    """Auto-detect system-wide optimization flags"""
    flags = []

    try:
        # Check available memory for optimization decisions
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / (1024 * 1024)

                            # If we have enough memory, we can be more aggressive
                            if mem_gb >= 16:
                                flags.append("-DNDEBUG")  # Disable debug assertions
                                print(
                                    f"High memory system detected ({mem_gb:.1f}GB), enabling aggressive optimizations"
                                )
                            break
            except:
                pass

        # Check number of CPU cores for parallel compilation
        cpu_count = os.cpu_count()
        if cpu_count:
            environ.setdefault("MAX_JOBS", str(cpu_count))
            print(f"Parallel compilation enabled with {cpu_count} cores")

        # Check for ccache availability
        try:
            result = subprocess.run(["which", "ccache"], check=False, capture_output=True, timeout=2)
            if result.returncode == 0:
                environ.setdefault("CC", "ccache gcc")
                environ.setdefault("CXX", "ccache g++")
                print("ccache detected and enabled for faster rebuilds")
        except:
            pass

    except Exception as e:
        print(f"System detection failed: {e}")

    return flags


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

# Auto-detect hardware-specific optimization flags
print("Auto-detecting hardware-specific optimization flags...")
cpu_flags = detect_cpu_flags()
cuda_flags = detect_cuda_flags()
system_flags = detect_system_flags()

# Force the ABI to match actual PyTorch libraries (always use legacy ABI for compatibility)
base_cxx_flags = ["-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0"]
if platform.system() != "Windows":
    base_cxx_flags.extend(cpu_flags)
    base_cxx_flags.extend(system_flags)

extra_compile_args = {
    "cxx": base_cxx_flags if platform.system() != "Windows" else ["/openmp", "-D_GLIBCXX_USE_CXX11_ABI=0"]
}

extra_link_args = ["-fopenmp"] if platform.system() != "Windows" else []

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

    # Use auto-detected CUDA flags
    if cuda_flags:
        extra_compile_args["nvcc"] = cuda_flags
    else:
        extra_compile_args["nvcc"] = ["-O2", "-D_GLIBCXX_USE_CXX11_ABI=0"]
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
    install_requires=["torch>=2.4.0"],
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
