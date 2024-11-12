dist:
	python setup.py build sdist bdist_wheel

test:
	pytest tests.py

.PHONY: build
build:
	python setup.py build_ext --inplace

.PHONY: suitesparse
suitesparse:
	rm -rf suitesparse
	git clone --depth 1 --branch v7.2.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
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
	find . -name "dist" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf

env:
	@echo export CPLUS_INCLUDE_PATH='/home/flaport/Projects/klujax/xla:/home/flaport/.anaconda/include/python3.12:/home/flaport/.anaconda/lib/python3.12/site-packages/pybind11/include:/home/flaport/Projects/klujax/suitesparse/SuiteSparse_config:/home/flaport/Projects/klujax/suitesparse/AMD/Include:/home/flaport/Projects/klujax/suitesparse/COLAMD/Include:/home/flaport/Projects/klujax/suitesparse/BTF/Include:/home/flaport/Projects/klujax/suitesparse/KLU/Include'
