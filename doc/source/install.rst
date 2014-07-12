Installation
============

The released version of GeoPandas is 0.1.  To install the released
version, use ``pip install geopandas``.

You may install the latest development version by cloning the
`GitHub`_ repository and using the setup script::

    git clone https://github.com/geopandas/geopandas.git
    cd geopandas
    python setup.py install

It is also possible to install the latest development version
available on PyPI with `pip` by adding the ``--pre`` flag for pip 1.4
and later, or to use `pip` to install directly from the GitHub
repository with::

    pip install git+git://github.com/geopandas/geopandas.git


Dependencies
------------

Supports Python versions 2.6, 2.7, and 3.2+.

- `numpy`_
- `pandas`_ (version 0.13 or later)
- `shapely`_
- `fiona`_
- `six`_
- `geopy`_ 0.96.3 (optional; for geocoding)
- `psycopg2`_ (optional; for PostGIS connection)

For plotting, these additional packages may be used:

- `matplotlib`_
- `descartes`_
- `pysal`_

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
.. _Travis CI: https://travis-ci.org/geopandas/geopandas

.. toctree::
   :maxdepth: 2

