# klujax justfile

# List all available commands
list:
    just --list

# Set up development environment (clones dependencies first)
dev: maybe-deps
    uv venv --python 3.13 --clear
    uv sync --all-extras
    uv run python setup.py build_ext --inplace

# Build distribution
dist:
    uv run python setup.py build sdist bdist_wheel

# (Re-)initialize dependencies
deps: suitesparse xla pybind11

# Initialize missing dependencies only
maybe-deps:
    @if [ ! -d "suitesparse" ]; then just suitesparse; fi
    @if [ ! -d "xla" ]; then just xla; fi
    @if [ ! -d "pybind11" ]; then just pybind11; fi

# Build extension in place
inplace:
    uv run python setup.py build_ext --inplace

# Run tests
test:
    uv run pytest tests.py

# Clone SuiteSparse
suitesparse:
    rm -rf suitesparse
    git clone --depth 1 --branch v7.5.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
    cd suitesparse && rm -rf .git

# Clone XLA
xla:
    rm -rf xla
    git clone https://github.com/openxla/xla xla
    cd xla && git checkout 05f004e8368c955b872126b1c978c60e33bbc5c8 && rm -rf .git

# Clone pybind11
pybind11:
    rm -rf pybind11
    git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11 pybind11
    cd pybind11 && rm -rf .git

# Clean build artifacts
clean:
    rm -rf .venv
    find . -name "dist" | xargs rm -rf
    find . -name "build" | xargs rm -rf
    find . -name "builds" | xargs rm -rf
    find . -name "__pycache__" | xargs rm -rf
    find . -name "*.so" | xargs rm -rf
    find . -name "*.egg-info" | xargs rm -rf
    find . -name ".ipynb_checkpoints" | xargs rm -rf
    find . -name ".pytest_cache" | xargs rm -rf

# Clean everything
clean-all: clean
    rm -rf suitesparse
    rm -rf xla
    rm -rf pybind11
    rm uv.lock
