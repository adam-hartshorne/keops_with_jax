#!/usr/bin/env python3
"""
Pytest configuration for KeOps JAX tests.
"""

import pytest
import sys


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "multigpu: marks tests that require multiple GPUs"
    )
    config.addinivalue_line(
        "markers", "pytorch: marks tests that require PyTorch"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available hardware."""
    import jax
    
    # Check GPU availability
    try:
        devices = jax.devices('gpu')
        n_gpus = len(devices)
    except:
        n_gpus = 0
    
    # Check PyTorch availability
    try:
        import torch
        pytorch_available = torch.cuda.is_available()
    except ImportError:
        pytorch_available = False
    
    skip_gpu = pytest.mark.skip(reason="No GPU available")
    skip_multigpu = pytest.mark.skip(reason="Multiple GPUs required")
    skip_pytorch = pytest.mark.skip(reason="PyTorch not available")
    
    for item in items:
        if "gpu" in item.keywords and n_gpus == 0:
            item.add_marker(skip_gpu)
        if "multigpu" in item.keywords and n_gpus < 2:
            item.add_marker(skip_multigpu)
        if "pytorch" in item.keywords and not pytorch_available:
            item.add_marker(skip_pytorch)


@pytest.fixture(scope="session")
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture(scope="session")
def jax_device():
    """Get the default JAX device."""
    import jax
    return jax.devices()[0]


@pytest.fixture(scope="session")
def torch_device():
    """Get the default PyTorch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')
    except ImportError:
        return None
