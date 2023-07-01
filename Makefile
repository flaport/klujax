dist:
	python setup.py build sdist bdist_wheel

.PHONY: build
build:
	python setup.py build_ext --inplace

test:
	pytest tests.py

klu:
	git clone --depth 1 --branch stable git@github.com:DrTimothyAldenDavis/SuiteSparse suitesparse || true
	export CMAKE_OPTIONS="-DNSTATIC=0" && cd suitesparse && sed -i 's/.*GraphBLAS \&\&.*//g' Makefile && make local

clean:
	find . -not -path "./suitesparse*" -name "dist" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "build" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "builds" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "__pycache__" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.so" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.egg-info" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".ipynb_checkpoints" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".pytest_cache" | xargs rm -rf

