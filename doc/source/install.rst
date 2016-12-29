Installation
============

Installing GeoPandas
---------------------

To install the released version, you can use pip::

    pip install geopandas

or you can install the conda package from the conda-forge channel::

    conda install -c conda-forge geopandas

You may install the latest development version by cloning the
`GitHub` repository and using the setup script::

    git clone https://github.com/geopandas/geopandas.git
    cd geopandas
    pip install .

It is also possible to install the latest development version
directly from the GitHub repository with::

    pip install git+git://github.com/geopandas/geopandas.git

Dependencies
--------------

Installation via `conda` should also install all dependencies, but a complete list is as follows:

- `numpy`_
- `pandas`_ (version 0.15.2 or later)
- `shapely`_
- `fiona`_
- `six`_
- `pyproj`_

Further, optional dependencies are:

- `geopy`_ 0.99 (optional; for geocoding)
- `psycopg2`_ (optional; for PostGIS connection)
- `rtree`_ (optional; spatial index to improve performance)

For plotting, these additional packages may be used:

- `matplotlib`_
- `descartes`_
- `pysal`_

These can be installed independently via the following set of commands::

    conda install -c conda-forge fiona shapely pyproj rtree
    conda install pandas


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
