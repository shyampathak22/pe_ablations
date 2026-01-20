"""Setup script for pe_ablations.

Note: The FPoPE CUDA kernels are now OPTIONAL (legacy).
The new architecture uses standard attention and doesn't require custom kernels.

To build legacy CUDA kernels (optional):
    FPOPE_BUILD_CUDA=1 uv pip install -e .
"""

import os
from setuptools import setup, find_packages

# Only build CUDA extension if explicitly requested
BUILD_CUDA = os.environ.get("FPOPE_BUILD_CUDA", "0") == "1"

ext_modules = []
cmdclass = {}

if BUILD_CUDA:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Force CUDA 12.8 path for Blackwell GPUs
    CUDA_12_8_PATH = "/usr/local/cuda-12.8"
    if os.path.exists(os.path.join(CUDA_12_8_PATH, "bin", "nvcc")):
        os.environ["CUDA_HOME"] = CUDA_12_8_PATH
        os.environ["CUDA_PATH"] = CUDA_12_8_PATH
        os.environ["PATH"] = os.path.join(CUDA_12_8_PATH, "bin") + ":" + os.environ.get("PATH", "")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSRC_DIR = os.path.join(ROOT_DIR, "src", "model", "kernels", "csrc")

    if os.path.exists(CSRC_DIR):
        nvcc_flags = ["-O3", "--use_fast_math", "-lineinfo"]
        rel_csrc_dir = os.path.relpath(CSRC_DIR, ROOT_DIR)

        ext_modules = [
            CUDAExtension(
                name="fpope_cuda",
                sources=[
                    os.path.join(rel_csrc_dir, "fpope_cuda.cpp"),
                    os.path.join(rel_csrc_dir, "fpope_cuda_kernel.cu"),
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_flags},
                include_dirs=[CSRC_DIR],
            )
        ]
        cmdclass = {"build_ext": BuildExtension}

setup(
    name="pe_ablations",
    version="0.1.0",
    packages=find_packages(where="."),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
