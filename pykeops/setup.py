#!/usr/bin/env python
"""
PyKeOps setup.py - builds JAX extension during install
"""
print("[KeOps setup.py] *** SETUP.PY IS BEING EXECUTED ***")

import sys

print(f"[KeOps setup.py] Python: {sys.executable}")
print(f"[KeOps setup.py] Args: {sys.argv}")


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

def build_jax_extension(install_dir=None):
    """Build the JAX extension using CMake."""
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

    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        print("[KeOps] CUDA nvcc not found - skipping JAX extension build")
        return False
    print(f"[KeOps] Found nvcc: {nvcc_path}")

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
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[KeOps] BUILD FAILED: {e}\n")
        return False


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