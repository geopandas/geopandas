Installation
============

GeoPandas is continuous-release software.  You may install the latest
source from `GitHub`_ and use the setup script::

    python setup.py install

GeoPandas is also available on `PyPI`_, so ``pip install geopandas``
should work as well. You will have to add the ``--pre`` flag
for pip 1.4 and later.

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
.. _GitHub: https://github.com/kjordahl/geopandas
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
.. _Travis CI: https://travis-ci.org/kjordahl/geopandas

.. toctree::
   :maxdepth: 2

