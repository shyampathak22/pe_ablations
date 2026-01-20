"""Setup script for building FPoPE CUDA extension.

Build the CUDA extension with uv:
    uv pip install -e .

Or build only the extension in-place:
    uv run python setup.py build_ext --inplace

The extension will be built as 'fpope_cuda.so' in the project root.
"""

import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(ROOT_DIR, "src", "model", "kernels", "csrc")


def find_cuda_home():
    """Find a compatible CUDA installation."""
    # Check if CUDA_HOME is already set
    if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "bin", "nvcc")):
        return CUDA_HOME

    # Look for CUDA in uv cache (PyTorch bundles CUDA)
    uv_cuda_paths = glob.glob(os.path.expanduser("~/.cache/uv/archive-v0/*/nvidia/cu*/"))
    for path in uv_cuda_paths:
        nvcc_path = os.path.join(path, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            return path

    # Look in standard locations
    for cuda_path in ["/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-13.0"]:
        if os.path.exists(os.path.join(cuda_path, "bin", "nvcc")):
            return cuda_path

    return None


def get_cuda_extension():
    """Build the FPoPE CUDA extension."""
    # Find CUDA and set environment
    cuda_home = find_cuda_home()
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = os.path.join(cuda_home, "bin") + ":" + os.environ.get("PATH", "")

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]

    # Set TORCH_CUDA_ARCH_LIST to let PyTorch auto-detect if not set
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            import torch
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                arch = f"{cap[0]}.{cap[1]}"
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        except Exception:
            pass

    return CUDAExtension(
        name="fpope_cuda",
        sources=[
            os.path.join(CSRC_DIR, "fpope_cuda.cpp"),
            os.path.join(CSRC_DIR, "fpope_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": nvcc_flags,
        },
        include_dirs=[CSRC_DIR],
    )


setup(
    ext_modules=[get_cuda_extension()],
    cmdclass={"build_ext": BuildExtension},
)
