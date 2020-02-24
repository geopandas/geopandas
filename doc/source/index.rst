GeoPandas |version|
===================

GeoPandas is an open source project to make working with geospatial
data in python easier.  GeoPandas extends the datatypes used by
`pandas`_ to allow spatial operations on geometric types.  Geometric
operations are performed by `shapely`_.  Geopandas further depends on
`fiona`_ for file access and `descartes`_ and `matplotlib`_ for plotting.

.. _pandas: http://pandas.pydata.org
.. _shapely: https://shapely.readthedocs.io
.. _fiona: https://fiona.readthedocs.io
.. _Descartes: https://pypi.python.org/pypi/descartes
.. _matplotlib: http://matplotlib.org

Description
-----------

The goal of GeoPandas is to make working with geospatial data in
python easier.  It combines the capabilities of pandas and shapely,
providing geospatial operations in pandas and a high-level interface
to multiple geometries to shapely.  GeoPandas enables you to easily do
operations in python that would otherwise require a spatial database
such as PostGIS.

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  Installation <install>
  Examples Gallery <gallery/index>

.. toctree::
  :maxdepth: 1
  :caption: User Guide

  Data Structures <data_structures>
  Reading and Writing Files <io>
  Indexing and Selecting Data <indexing>
  Making Maps <mapping>
  Managing Projections <projections>
  Geometric Manipulations <geometric_manipulations>
  Set Operations with overlay <set_operations>
  Aggregation with dissolve <aggregation_with_dissolve>
  Merging Data <mergingdata>
  Geocoding <geocoding>
  missing_empty

.. toctree::
  :maxdepth: 1
  :caption: Reference Guide

  Reference to All Attributes and Methods <reference>
  Changelog <changelog>


.. toctree::
  :maxdepth: 1
  :caption: Developer

  Contributing to GeoPandas <contributing>


Get in touch
------------

- Ask usage questions ("How do I?") on `StackOverflow`_ or `GIS StackExchange`_.
- Report bugs, suggest features or view the source code `on GitHub`_.
- For a quick question about a bug report or feature request, or Pull Request,
  head over to the `gitter channel`_.
- For less well defined questions or ideas, or to announce other projects of
  interest to GeoPandas users, ... use the `mailing list`_.

.. _StackOverflow: https://stackoverflow.com/questions/tagged/geopandas
.. _GIS StackExchange: https://gis.stackexchange.com/questions/tagged/geopandas
.. _on GitHub: https://github.com/geopandas/geopandas
.. _gitter channel: https://gitter.im/geopandas/geopandas
.. _mailing list: https://groups.google.com/forum/#!forum/geopandas


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
