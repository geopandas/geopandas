SHELL= /bin/bash
PYTHON ?= python

inplace:
	$(PYTHON) setup.py build_ext --inplace --with-cython -l geos_c

test: inplace
	py.test geopandas

clean:
	rm -f geopandas/*.c geopandas/*.so
	rm -rf build/ geopandas/__pycache__/ geopandas/*/__pycache__/
