# Basic install (JAX extension built if jax/nanobind/nvcc are available)
pip install .

# With JAX dependencies
pip install ".[jax]"

# Editable install for development
pip install -e . --no-build-isolation

# Force rebuild
pip install -e . --no-build-isolation --force-reinstall