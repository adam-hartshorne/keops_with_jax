#!/usr/bin/env python
"""
PyKeOps setup.py - builds JAX extension during install
"""
# print("[KeOps setup.py] *** SETUP.PY IS BEING EXECUTED ***")

import sys

# print(f"[KeOps setup.py] Python: {sys.executable}")
# print(f"[KeOps setup.py] Args: {sys.argv}")


# =============================================================================
# Development mode detection - warn about --no-deps
# =============================================================================

def check_development_mode():
    """Check if keopscore is installed in editable/development mode."""
    try:
        import keopscore
        keopscore_path = keopscore.__file__

        # Check if it's an editable install (path contains site-packages with .egg-link
        # or is outside site-packages entirely)
        import site
        site_packages = site.getsitepackages() + [site.getusersitepackages()]

        is_editable = True
        for sp in site_packages:
            if sp and keopscore_path and keopscore_path.startswith(sp):
                # Check if it's a .egg-link (editable) or regular install
                import os
                egg_link = os.path.join(sp, 'keopscore.egg-link')
                if not os.path.exists(egg_link):
                    is_editable = False
                break

        if is_editable:
            # Check if --no-deps was passed
            if '--no-deps' not in sys.argv:
                print("\n" + "=" * 70)
                print("[KeOps WARNING] Detected keopscore in DEVELOPMENT/EDITABLE mode!")
                print("=" * 70)
                print(f"  keopscore location: {keopscore_path}")
                print("")
                print("  If you're developing keopscore, use --no-deps to avoid replacing it:")
                print("")
                print("    pip install . --no-build-isolation --no-deps")
                print("")
                print("  Or for editable pykeops install:")
                print("")
                print("    pip install -e . --no-build-isolation --no-deps")
                print("=" * 70 + "\n")
    except ImportError:
        pass  # keopscore not installed yet, that's fine


# Only check during install commands
if any(cmd in sys.argv for cmd in ['install', 'bdist_wheel', 'develop']):
    check_development_mode()

from codecs import open
import os
from os import path
import subprocess
import shutil
import glob

from setuptools import setup, Distribution
from setuptools.command.build import build
from setuptools.command.egg_info import egg_info

here = path.abspath(path.dirname(__file__))

# get keops version
with open(os.path.join(here, "pykeops", "keops_version"), encoding="utf-8") as v:
    current_version = v.read().rstrip()

# Get the long description from the README file
with open(path.join(here, "pykeops", "readme.md"), encoding="utf-8") as f:
    long_description = f.read()


# =============================================================================
# JAX Extension Builder
# =============================================================================

_jax_extension_built = False

def build_jax_extension(install_dir=None):
    """Build the JAX extension using CMake."""
    global _jax_extension_built

    if _jax_extension_built:
        print("[KeOps] JAX extension already built, skipping")
        return True

    print("[KeOps] build_jax_extension() called")

    jax_binder_source = os.path.join(here, "pykeops", "jax", "binders")

    if not os.path.isdir(jax_binder_source):
        print(f"[KeOps] JAX binder source not found at {jax_binder_source}, skipping")
        return False

    # Check dependencies
    try:
        import jax
        import jaxlib
        print(f"[KeOps] Found JAX {jax.__version__}")
    except ImportError:
        print("[KeOps] JAX not found - skipping JAX extension build")
        return False

    try:
        import nanobind
        nanobind_cmake_dir = nanobind.cmake_dir()
        print(f"[KeOps] Found nanobind, cmake_dir={nanobind_cmake_dir}")
    except ImportError:
        print("[KeOps] nanobind not found - skipping JAX extension build")
        return False

    # The extension is dlopen'ed into the JAX process, so its CUDA runtime must match the
    # major version JAX itself uses. CMake would otherwise take whatever `nvcc` is first on
    # PATH, which on a box with several toolkits installed can be a different major.
    jax_cuda_major = detect_jax_cuda_major()
    nvcc_path = find_nvcc(jax_cuda_major)
    if not nvcc_path:
        print("[KeOps] CUDA nvcc not found - skipping JAX extension build")
        return False
    if jax_cuda_major is None:
        print(f"[KeOps] Found nvcc: {nvcc_path} (could not tell JAX's CUDA major; using it as is)")
    elif nvcc_major(nvcc_path) == jax_cuda_major:
        print(f"[KeOps] Found nvcc: {nvcc_path} (CUDA {jax_cuda_major}, matching JAX)")
    else:
        print(f"[KeOps] WARNING: no CUDA {jax_cuda_major} nvcc found to match JAX; building the "
              f"extension with {nvcc_path} (CUDA {nvcc_major(nvcc_path)}) instead")

    cmake_path = shutil.which("cmake")
    if not cmake_path:
        print("[KeOps] CMake not found - skipping JAX extension build")
        return False
    print(f"[KeOps] Found cmake: {cmake_path}")

    # Determine output directory
    if install_dir is None:
        install_dir = os.path.join(here, "pykeops", "jax")

    os.makedirs(install_dir, exist_ok=True)

    # Build directory - use a temp location outside source
    build_dir = os.path.join(here, "build", "jax_ext_build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    # Detect CUDA architecture
    cuda_arch = detect_cuda_arch()

    cmake_args = [
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={install_dir}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dnanobind_DIR={nanobind_cmake_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if not os.environ.get("CUDACXX"):
        # An explicit CUDACXX from the caller wins; otherwise pin CMake to the nvcc chosen above.
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")
    if cuda_arch:
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

    print(f"\n{'=' * 60}")
    print(f"[KeOps] Building JAX extension")
    print(f"  Source: {jax_binder_source}")
    print(f"  Build:  {build_dir}")
    print(f"  Output: {install_dir}")
    print(f"  CUDA:   {cuda_arch}")
    print(f"  CMake args: {cmake_args}")
    print(f"{'=' * 60}\n")

    try:
        subprocess.check_call(["cmake", jax_binder_source] + cmake_args, cwd=build_dir)

        build_args = ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count() or 1)]
        subprocess.check_call(build_args, cwd=build_dir)

        so_files = glob.glob(os.path.join(install_dir, "keops_jax_ext*.so"))
        print(f"\n[KeOps] SUCCESS! Built: {so_files}\n")
        _jax_extension_built = True
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[KeOps] BUILD FAILED: {e}\n")
        return False


def nvcc_major(nvcc_path):
    """Major CUDA version an nvcc binary reports, or None."""
    import re
    try:
        out = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"release (\d+)\.", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def detect_jax_cuda_major():
    """CUDA major version of the installed JAX GPU plugin (jax-cuda13-plugin -> 13), or None."""
    import importlib.metadata as md
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name.startswith("jax-cuda") and name.endswith("-plugin"):
            digits = name[len("jax-cuda"):-len("-plugin")]
            if digits.isdigit():
                return int(digits)
    return None


def find_nvcc(cuda_major):
    """First nvcc whose major version matches `cuda_major`, searching CUDA_PATH / CUDA_HOME,
    PATH, /usr/local/cuda and the versioned /usr/local/cuda-* toolkits. Falls back to the PATH
    nvcc (or None) when nothing matches or the major is unknown."""
    candidates = []
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            candidates.append(os.path.join(os.environ[var], "bin", "nvcc"))
    on_path = shutil.which("nvcc")
    if on_path:
        candidates.append(on_path)
    candidates.append("/usr/local/cuda/bin/nvcc")
    if cuda_major is not None:
        candidates += sorted(glob.glob(f"/usr/local/cuda-{cuda_major}*/bin/nvcc"), reverse=True)
    candidates = [c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)]
    if cuda_major is not None:
        for c in candidates:
            if nvcc_major(c) == cuda_major:
                return c
    return on_path


def detect_cuda_arch():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            archs = [cap.strip().replace(".", "") for cap in result.stdout.strip().split('\n') if cap.strip()]
            if archs:
                return ";".join(dict.fromkeys(archs))  # unique, preserve order
    except Exception:
        pass
    return "70;75;80;86;89;90"


# =============================================================================
# Custom Commands
# =============================================================================

class CustomBuild(build):
    def run(self):
        print("[KeOps] CustomBuild.run() called")
        build_jax_extension(os.path.join(here, "pykeops", "jax"))
        build.run(self)


class CustomEggInfo(egg_info):
    def run(self):
        print("[KeOps] CustomEggInfo.run() called")
        # Build extension before creating egg-info so .so is included
        build_jax_extension(os.path.join(here, "pykeops", "jax"))
        egg_info.run(self)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# =============================================================================
# BUILD NOW if we're doing an install/build/bdist_wheel
# =============================================================================

# Check if this is an install or build command
if any(cmd in sys.argv for cmd in ['install', 'build', 'bdist_wheel', 'develop', 'egg_info']):
    print(f"[KeOps] Detected build command in {sys.argv}, building JAX extension NOW")
    build_jax_extension(os.path.join(here, "pykeops", "jax"))

# =============================================================================
# Setup
# =============================================================================

setup(
    name="pykeops",
    version=current_version,
    description="Python bindings of KeOps: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.kernel-operations.io/",
    project_urls={
        "Bug Reports": "https://github.com/getkeops/keops/issues",
        "Source": "https://github.com/getkeops/keops",
    },
    author="B. Charlier, J. Feydy, J. Glaunes",
    author_email="benjamin.charlier@umontpellier.fr, jean.feydy@gmail.com, alexis.glaunes@parisdescartes.fr",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="kernels gpu autodiff",
    packages=[
        "pykeops",
        "pykeops.common",
        "pykeops.common.keops_io",
        "pykeops.numpy",
        "pykeops.numpy.cluster",
        "pykeops.numpy.generic",
        "pykeops.numpy.lazytensor",
        "pykeops.test",
        "pykeops.torch",
        "pykeops.torch.cluster",
        "pykeops.torch.generic",
        "pykeops.torch.lazytensor",
        "pykeops.jax",
        "pykeops.jax.generic",
        "pykeops.jax.lazytensor",
        "pykeops.jax.test",
    ],
    package_data={
        "pykeops": [
            "readme.md",
            "licence.txt",
            "keops_version",
            "common/keops_io/pykeops_nvrtc.cpp",
        ],
        "pykeops.jax": [
            "*.so",
            "binders/CMakeLists.txt",
            "binders/keops_jax.cpp",
        ],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass={
        "build": CustomBuild,
        "egg_info": CustomEggInfo,
    },
    install_requires=["numpy", "pybind11", "keopscore"],
    extras_require={
        "jax": ["jax>=0.4.20", "jaxlib>=0.4.20", "nanobind>=1.0", "cmake>=3.18"],
        "test": ["pytest", "numpy", "torch"],
    },
)