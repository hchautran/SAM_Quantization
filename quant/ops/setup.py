# setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils import cpp_extension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")

# Get CUDA compute capability
device_props = torch.cuda.get_device_properties(0)
compute_capability = f"{device_props.major}{device_props.minor}"
arch_flags = [f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}"]

# Additional compilation flags
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-Xptxas=-O3',
        '--expt-relaxed-constexpr'
    ] + arch_flags
}

# Extension module
ext_modules = [
    cpp_extension.CUDAExtension(
        'w8a8_matmul_ext',
        [
            'w8a8.cpp',
            'w8a8.cu',
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=cpp_extension.include_paths(),
        library_dirs=cpp_extension.library_paths(),
        libraries=['cudart', 'cublas'],
    ),
]

setup(
    name='w8a8_matmul_ext',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.5.0",
    ],
)