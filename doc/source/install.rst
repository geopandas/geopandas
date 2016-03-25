Installation
============

Installing GeoPandas
---------------------

To install the released version, you can use pip::

    pip install geopandas

or you can install the conda package from the IOOS channel::

    conda install -c ioos geopandas

You may install the latest development version by cloning the
`GitHub` repository and using the setup script::

    git clone https://github.com/geopandas/geopandas.git
    cd geopandas
    pip install .

It is also possible to install the latest development version
available on PyPI with `pip` by adding the ``--pre`` flag for pip 1.4
and later, or to use `pip` to install directly from the GitHub
repository with::

    pip install git+git://github.com/geopandas/geopandas.git


Dependencies
------------

GeoPandas supports Python versions 2.6, 2.7, and 3.3+. The required
dependencies are:

- `numpy`
- `pandas` (version 0.13 or later)
- `shapely`
- `fiona`
- `six`
- `pyproj`

Further, optional dependencies are:

- `geopy` 0.99 (optional; for geocoding)
- `psycopg2` (optional; for PostGIS connection)

For plotting, these additional packages may be used:

- `matplotlib`
- `descartes`
- `pysal`

Further, `rtree` is an optional dependency. ``rtree`` requires the C library
`libspatialindex`. If using brew, you can install using
``brew install Spatialindex``.


Troubleshooting
---------------

~~~~~~




.. toctree::
   :maxdepth: 2
