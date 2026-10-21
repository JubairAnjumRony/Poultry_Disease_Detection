import ctypes
import os
import sys
import sysconfig

_CUDA_LIBS = (
    "libcudart.so.11.0",
    "libcublas.so.11",
    "libcublasLt.so.11",
    "libcufft.so.10",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.11",
    "libcudnn.so.8",
)

_loaded = False


def enable_cuda_libs() -> None:
    """Preload pip-installed CUDA 11.8 libraries for TensorFlow 2.12.

    The `nvidia-*-cu11` wheels live in site-packages/nvidia/*/lib, which is not
    on the dynamic loader search path unless the venv activation script exports
    LD_LIBRARY_PATH. Preloading them here makes the GPU available even when the
    project is started as `.venv/bin/python app.py` without activation.
    """
    global _loaded
    if _loaded or sys.platform != "linux":
        return
    _loaded = True

    site_packages = sysconfig.get_paths().get("purelib")
    if not site_packages:
        return

    nvidia_root = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia_root):
        return

    lib_dirs = [
        os.path.join(nvidia_root, name, "lib")
        for name in sorted(os.listdir(nvidia_root))
    ]
    lib_dirs = [lib_dir for lib_dir in lib_dirs if os.path.isdir(lib_dir)]

    for lib_name in _CUDA_LIBS:
        for lib_dir in lib_dirs:
            lib_path = os.path.join(lib_dir, lib_name)
            if not os.path.exists(lib_path):
                continue
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
