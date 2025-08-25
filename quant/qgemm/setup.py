import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as torch_cpp_ext
import os
import pathlib


this_dir = Path(__file__).parent.resolve()
HERE = pathlib.Path(__file__).absolute().parent
def third_party_cmake():
    import subprocess, sys, shutil
    
    cmake = shutil.which('cmake')
    if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

    retcode = subprocess.call([cmake, HERE])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)

    # install fast hadamard transform
    hadamard_dir = os.path.join(HERE, 'third-party/fast-hadamard-transform')
    pip = shutil.which('pip')
    retcode = subprocess.call([pip, 'install', '-e', hadamard_dir])




def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    return [
        '-gencode', 'arch=compute_75,code=sm_75',  # Turing
        '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
        '-gencode', 'arch=compute_86,code=sm_86',  # Ampere
    ]


def make_extension():
    include_dirs = [
        str(this_dir / "kernels" / "include"),
        # CUTLASS typical include layout is <repo>/include/cutlass/...
        str(this_dir / "third-party" / "cutlass" / "include"),
        # Some forks place headers directly under cutlass/
        str(this_dir / "third-party" / "cutlass"),
    ]

    # Allow overriding CUTLASS path via environment variable if bundled path is empty
    cutlass_env = os.environ.get("CUTLASS_PATH")
    if cutlass_env:
        include_dirs.extend([
            str(Path(cutlass_env) / "include"),
            str(Path(cutlass_env)),
        ])

    sources = [
        str(this_dir / "kernels" / "bindings.cpp"),
        str(this_dir / "kernels" / "gemm.cu"),
        str(this_dir / "kernels" / "quant.cu"),
    ]

    extra_compile_args = {
        "cxx": [],
        "nvcc": get_cuda_arch_flags(),
    }

    return CUDAExtension(
        name="qgemm",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
    )


if __name__ == '__main__':
    third_party_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name="qgemm",
        version="0.1.0",
        description="Quantized GEMM CUDA extension (int4/int8/fp16)",
        ext_modules=[make_extension()],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )


