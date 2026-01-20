dist:
	uv run python setup.py build sdist bdist_wheel

dev: suitesparse xla pybind11
	uv venv --python 3.13 --clear
	uv sync --all-extras
	uv run python setup.py build_ext --inplace

inplace:
	uv run python setup.py build_ext --inplace

test:
	uv run pytest tests.py

suitesparse:
	rm -rf suitesparse
	git clone --depth 1 --branch v7.5.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
	cd suitesparse && rm -rf .git

xla:
	rm -rf xla
	git clone https://github.com/openxla/xla xla
	cd xla && git checkout 05f004e8368c955b872126b1c978c60e33bbc5c8 && rm -rf .git

pybind11:
	rm -rf pybind11
	git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11 pybind11
	cd pybind11 && rm -rf .git

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

