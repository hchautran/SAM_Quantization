# setup_optimized.py for Optimized W8A8 CUDA Extension
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils import cpp_extension
import torch
import os

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")

# Get CUDA compute capability
device_props = torch.cuda.get_device_properties(0)
compute_capability = f"{device_props.major}{device_props.minor}"

# Define target architectures (common ones)
arch_flags = [
    f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    # Add support for other common architectures
    "-gencode=arch=compute_70,code=sm_70",   # V100
    "-gencode=arch=compute_75,code=sm_75",   # T4, RTX 20xx
    "-gencode=arch=compute_80,code=sm_80",   # A100
    "-gencode=arch=compute_86,code=sm_86",   # RTX 30xx
    "-gencode=arch=compute_89,code=sm_89",   # RTX 40xx
]

# Additional compilation flags for optimization
extra_compile_args = {
    'cxx': [
        '-O3',
        '-std=c++17',
        '-fPIC',
        '-Wall',
        '-Wextra'
    ],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-Xptxas=-O3',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '--maxrregcount=64',
        '--ptxas-options=-v',
        '-std=c++17'
    ] + arch_flags
}

# Extension module
ext_modules = [
    cpp_extension.CUDAExtension(
        'w8a8_matmul_optimized_ext',
        [
            'w8a8_optimized.cpp',
            'w8a8_optimized.cu',
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=cpp_extension.include_paths(),
        library_dirs=cpp_extension.library_paths(),
        libraries=['cudart', 'cublas', 'cublasLt'],
        define_macros=[
            ('TORCH_EXTENSION_NAME', 'w8a8_matmul_optimized_ext'),
        ],
    ),
]

setup(
    name='w8a8_matmul_optimized_ext',
    version='1.0.0',
    description='Optimized W8A8 Matrix Multiplication CUDA Extension',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "pybind11>=2.6.0",
    ],
    author='SAM Quantization Team',
    author_email='',
    url='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
) 