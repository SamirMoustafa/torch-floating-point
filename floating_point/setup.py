# setup.py build_ext --inplace
import platform
from os import environ, path

from setuptools import find_packages, setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

__version__ = "0.0.1"
__HERE__ = path.dirname(path.abspath(__file__))

# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), "..", "README.md"), encoding="utf-8") as f:
    long_description = f.read()

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
sources = [path.join(__HERE__, "float_round.cpp")]
define_macros = []

# Conditionally add CUDA support
if cuda.is_available():
    print("CUDA detected, building with CUDA support.")
    extension_class = CUDAExtension
    sources.append(path.join(__HERE__, "float_round_cuda.cu"))
    define_macros.append(("WITH_CUDA", None))
    extra_compile_args["nvcc"] = ["-O2"]
else:
    print("No CUDA detected, building without CUDA support.")
    extension_class = CppExtension

setup(
    name="floating_point",
    version=__version__,
    description="Floating Point Quantization Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Samir Moustafa",
    author_email="samir.moustafa.97@gmail.com",
    url="https://github.com/SamirMoustafa/torch-floating-point",
    install_requires=["torch>=2.4.0"],
    packages=find_packages(),
    ext_modules=[
        extension_class(
            name="floating_point",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
