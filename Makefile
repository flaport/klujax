build:
	python setup.py build_ext --inplace

clean:
	find . -name "build" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf

