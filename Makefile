dist:
	python setup.py build sdist bdist_wheel

.PHONY: build
build:
	python setup.py build_ext --inplace

test:
	pytest tests.py

suitesparse:
	git clone --depth 1 --branch stable https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse || true
	cd suitesparse && rm -rf .git

clean:
	find . -not -path "./suitesparse*" -name "dist" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "build" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "builds" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "__pycache__" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.so" | xargs rm -rf
	find . -not -path "./suitesparse*" -name "*.egg-info" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".ipynb_checkpoints" | xargs rm -rf
	find . -not -path "./suitesparse*" -name ".pytest_cache" | xargs rm -rf

