Installation
============

To install the released version, you can use pip::

    pip install geopandas

or you can install the conda package from the IOOS channel::

    conda install -c ioos geopandas

You may install the latest development version by cloning the
`GitHub`_ repository and using the setup script::

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

- `numpy`_
- `pandas`_ (version 0.13 or later)
- `shapely`_
- `fiona`_
- `six`_
- `pyproj`_

Further, optional dependencies are:

- `geopy`_ 0.99 (optional; for geocoding)
- `psycopg2`_ (optional; for PostGIS connection)

For plotting, these additional packages may be used:

- `matplotlib`_
- `descartes`_
- `pysal`_

Further, `rtree`_ is an optional dependency. ``rtree`` requires the C library
`libspatialindex`_. If using brew, you can install using
``brew install Spatialindex``.

Testing
-------

To run the current set of tests from the source directory, run::

    nosetests -v

from a command line.

Tests are automatically run on all commits on the GitHub repository,
including pull requests, on `Travis CI`_.

.. _PyPI: https://pypi.python.org/pypi/geopandas
.. _GitHub: https://github.com/geopandas/geopandas
.. _numpy: http://www.numpy.org
.. _pandas: http://pandas.pydata.org
.. _shapely: http://toblerity.github.io/shapely
.. _fiona: http://toblerity.github.io/fiona
.. _Descartes: https://pypi.python.org/pypi/descartes
.. _matplotlib: http://matplotlib.org
.. _geopy: https://github.com/geopy/geopy
.. _six: https://pythonhosted.org/six
.. _psycopg2: https://pypi.python.org/pypi/psycopg2
.. _pysal: http://pysal.org
.. _pyproj: https://github.com/jswhit/pyproj
.. _rtree: https://github.com/Toblerity/rtree
.. _libspatialindex: https://github.com/libspatialindex/libspatialindex
.. _Travis CI: https://travis-ci.org/geopandas/geopandas


.. toctree::
   :maxdepth: 2
