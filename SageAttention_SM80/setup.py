"""
Minimal SageAttention setup for SM80 (Ampere) architecture only.
Based on SageAttention but stripped down to essential components.
"""

import os
import sys
import subprocess
from packaging.version import parse, Version

from setuptools import setup, find_packages

# Skip CUDA build in CI or when explicitly requested
SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    # Compiler flags for SM80
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
    NVCC_FLAGS = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math",
        "--threads=8",
        "-Xptxas=-v",
        "-diag-suppress=174",
    ]

    # Append flags from env if provided
    cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
    if cxx_append:
        CXX_FLAGS += cxx_append.split()
    nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
    if nvcc_append:
        NVCC_FLAGS += nvcc_append.split()

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc."""
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)

    # Validate the NVCC CUDA version
    if nvcc_cuda_version < Version("12.0"):
        raise RuntimeError("CUDA 12.0 or higher is required to build the package.")

    # Add SM80 architecture (Ampere: A100, A6000, RTX 3090, etc.)
    NVCC_FLAGS += ["-gencode", "arch=compute_80,code=sm_80"]
    NVCC_FLAGS += ["-gencode", "arch=compute_80,code=compute_80"]

    # Also support SM86 (RTX 3080, 3090, etc.) if available
    arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if "8.6" in arch_list_env or not arch_list_env:
        NVCC_FLAGS += ["-gencode", "arch=compute_86,code=sm_86"]

    print(f"Building for SM80 (Ampere) architecture with CUDA {nvcc_cuda_version}")

    # SM80 QAttn kernel
    ext_modules.append(
        CUDAExtension(
            name="sageattention._qattn_sm80",
            sources=[
                "csrc/qattn/pybind_sm80.cpp",
                "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
            ],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        )
    )

    # Fused kernels
    ext_modules.append(
        CUDAExtension(
            name="sageattention._fused",
            sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        )
    )

    cmdclass = {"build_ext": BuildExtension}

setup(
    name='sageattention_sm80',
    version='1.0.0',
    author='SageAttention team (SM80 minimal version)',
    license='Apache 2.0 License',
    description='Minimal SM80-only version of SageAttention for Ampere GPUs',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
