Installation
============

GeoPandas is pre-alpha software.  Please install the latest source
from `GitHub`_ and use the setup script::

    python setup.py install

GeoPandas is also available on `PyPI`_, so ``pip install --pre geopandas``
should work as well.

Dependencies
------------

Requires Python version 2.6 or 2.7 (Python 3.x support is planned
soon).

- `numpy`_
- `pandas`_
- `shapely`_
- `fiona`_
- `descartes`_
- `matplotlib`_
- `geopy`_ (optional; for geocoding)

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
.. _Travis CI: https://travis-ci.org/kjordahl/geopandas

.. toctree::
   :maxdepth: 2

