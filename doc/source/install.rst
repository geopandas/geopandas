Installation
============

GeoPandas depends for its spatial functionality on a large geospatial, open
source stack of libraries (`GEOS`_, `GDAL`_, `PROJ`_). See the
:ref:`dependencies` section below for more details. Those base C
libraries can sometimes be a challenge to install. Therefore, we advise you
to closely follow the recommendations below to avoid installation problems.

.. _install-conda:

Installing with Anaconda / conda
--------------------------------

To install GeoPandas and all its dependencies, we recommend to use the `conda`_
package manager. This can be obtained by installing the
`Anaconda Distribution`_ (a free Python distribution for data science), or
through `miniconda`_ (minimal distribution only containing Python and the
`conda`_ package manager). See also the `installation docs
<https://conda.io/docs/user-guide/install/download.html>`__ for more information
on how to install Anaconda or miniconda locally.

The advantage of using the `conda`_ package manager is that it provides
pre-built binaries for all the required and optional dependencies of GeoPandas
for all platforms (Windows, Mac, Linux).

To install the latest version of GeoPandas, you can then do::

    conda install geopandas


Using the conda-forge channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`conda-forge`_ is a community effort that provides conda packages for a wide
range of software. It provides the *conda-forge* package channel for conda from
which packages can be installed, in addition to the "*defaults*" channel
provided by Anaconda.
Depending on what other packages you are working with, the *defaults* channel
or *conda-forge* channel may be better for your needs (e.g. some packages are
available on *conda-forge* and not on *defaults*).

GeoPandas and all its dependencies are available on the *conda-forge*
channel, and can be installed as::

    conda install --channel conda-forge geopandas

.. note::

    We strongly recommend to either install everything from the *defaults*
    channel, or everything from the *conda-forge* channel. Ending up with a
    mixture of packages from both channels for the dependencies of GeoPandas
    can lead to import problems.
    See the `conda-forge section on using multiple channels
    <http://conda-forge.org/docs/user/tipsandtricks.html#using-multiple-channels>`__
    for more details.


Creating a new environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating a new environment is not strictly necessary, but given that installing
other geospatial packages from different channels may cause dependency conflicts
(as mentioned in the note above), it can be good practice to install the geospatial
stack in a clean environment starting fresh. 

The following commands create a new environment with the name ``geo_env``,
configures it to install packages always from conda-forge, and installs
GeoPandas in it::

    conda create -n geo_env
    conda activate geo_env
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install python=3 geopandas


.. _install-pip:

Installing with pip
-------------------

GeoPandas can also be installed with pip, if all dependencies can be installed
as well::

    pip install geopandas

.. _install-deps:

.. warning::

    When using pip to install GeoPandas, you need to make sure that all dependencies are
    installed correctly.

    - `shapely`_ and `fiona`_ provide binary wheels with the
      dependencies included for Mac and Linux, but not for Windows.
    - `pyproj`_ provides binary wheels with depencies included
      for Mac, Linux, and Windows.
    - `rtree`_ does not provide wheels.
    - Windows wheels for `shapely`, `fiona`, `pyproj` and `rtree`
      can be found at `Christopher Gohlke's website 
      <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

    So depending on your platform, you might need to compile and install their
    C dependencies manually. We refer to the individual packages for more
    details on installing those.
    Using conda (see above) avoids the need to compile the dependencies yourself.

Installing from source
----------------------

You may install the latest development version by cloning the
`GitHub` repository and using pip to install from the local directory::

    git clone https://github.com/geopandas/geopandas.git
    cd geopandas
    pip install .

It is also possible to install the latest development version
directly from the GitHub repository with::

    pip install git+git://github.com/geopandas/geopandas.git

For installing GeoPandas from source, the same :ref:`note <install-deps>` on
the need to have all dependencies correctly installed applies. But, those
dependencies can also be installed independently with conda before installing
GeoPandas from source::

    conda install pandas fiona shapely pyproj rtree

See the :ref:`section on conda <install-conda>` above for more details on
getting running with Anaconda.

.. _dependencies:

Dependencies
------------

Required dependencies:

- `numpy`_
- `pandas`_ (version 0.23.4 or later)
- `shapely`_ (interface to `GEOS`_)
- `fiona`_ (interface to `GDAL`_)
- `pyproj`_ (interface to `PROJ`_; version 2.2.0 or later)
- `six`_

Further, optional dependencies are:

- `rtree`_ (optional; spatial index to improve performance and required for
  overlay operations; interface to `libspatialindex`_)
- `psycopg2`_ (optional; for PostGIS connection)
- `geopy`_ (optional; for geocoding)

For plotting, these additional packages may be used:

- `matplotlib`_ (>= 2.0.1)
- `descartes`_
- `mapclassify`_


.. _PyPI: https://pypi.python.org/pypi/geopandas

.. _GitHub: https://github.com/geopandas/geopandas

.. _numpy: http://www.numpy.org

.. _pandas: http://pandas.pydata.org

.. _shapely: https://shapely.readthedocs.io

.. _fiona: https://fiona.readthedocs.io

.. _Descartes: https://pypi.python.org/pypi/descartes

.. _matplotlib: http://matplotlib.org

.. _geopy: https://github.com/geopy/geopy

.. _six: https://pythonhosted.org/six

.. _psycopg2: https://pypi.python.org/pypi/psycopg2

.. _mapclassify: http://pysal.org/mapclassify

.. _pyproj: https://github.com/pyproj4/pyproj

.. _rtree: https://github.com/Toblerity/rtree

.. _libspatialindex: https://github.com/libspatialindex/libspatialindex

.. _Travis CI: https://travis-ci.org/geopandas/geopandas

.. _conda: https://conda.io/en/latest/

.. _Anaconda distribution: https://www.anaconda.com/distribution/

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. _conda-forge: https://conda-forge.org/

.. _GDAL: https://www.gdal.org/

.. _GEOS: https://geos.osgeo.org

.. _PROJ: https://proj.org/
