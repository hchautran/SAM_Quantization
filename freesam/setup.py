from torch.utils import cpp_extension
from setuptools import setup
import os
import torch

os.environ['TORCH_CUDA_ARCH_LIST']='8.0'

# Get PyTorch library path
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

# Get absolute paths
here = os.path.abspath(os.path.dirname(__file__))
cutlass_include = os.path.join(here, 'third-party', 'cutlass', 'include')
csrc_path = os.path.join(here, 'csrc')

setup(
    name="freesam",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "freesam",
            [
                "binding.cpp",
                "csrc/gemm.cu",
                "csrc/gemm_v1.cu",
                "csrc/gemm_v2.cu",
                "csrc/quant.cu",
                "csrc/flash_attn.cu",
                "csrc/flash_attn_rel.cu",
            ],
            # For debugging: use -g, -G, -O0
            # For production: use -O3
            extra_compile_args={'nvcc': ['-g', '-G', '-O0']},
            extra_link_args=[f'-Wl,-rpath,{torch_lib_path}'],
            include_dirs=[
                csrc_path,
                cutlass_include,
            ]
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
