SHELL= /bin/bash
PYTHON ?= python

inplace:
	$(PYTHON) setup.py build_ext --inplace --with-cython

install:
	$(PYTHON) setup.py build_ext --with-cython install

test: inplace
	py.test geopandas

clean:
	rm -f geopandas/vectorized.c geopandas/array.c geopandas/*.so geopandas/*.o
	rm -rf build/ geopandas/__pycache__/ geopandas/*/__pycache__/
