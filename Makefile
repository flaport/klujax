dist:
	python setup.py build sdist bdist_wheel

.PHONY: build
dev:
	uv venv --python 3.12
	uv sync --all-extras
	python setup.py build_ext --inplace

inplace:
	python setup.py build_ext --inplace

test:
	pytest tests.py

.PHONY: suitesparse
suitesparse:
	rm -rf suitesparse
	git clone --depth 1 --branch v7.9.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
	cd suitesparse && rm -rf .git

.PHONY: xla
xla:
	rm -rf xla
	git clone --depth 1 --branch main https://github.com/openxla/xla xla
	cd xla && rm -rf .git

.PHONY: pybind11
pybind11:
	rm -rf pybind11
	git clone --depth 1 --branch stable https://github.com/pybind/pybind11 pybind11
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

