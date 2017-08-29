GeoPandas |version|
===================

GeoPandas is an open source project to make working with geospatial
data in python easier.  GeoPandas extends the datatypes used by
`pandas`_ to allow spatial operations on geometric types.  Geometric
operations are performed by `shapely`_.  Geopandas further depends on
`fiona`_ for file access and `descartes`_ and `matplotlib`_ for plotting.

.. _pandas: http://pandas.pydata.org
.. _shapely: http://toblerity.github.io/shapely
.. _fiona: http://toblerity.github.io/fiona
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
  :maxdepth: 2

  Installation <install>
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
  Reference to All Attributes and Methods <reference>
  Contributing to GeoPandas <contributing>
  About <about>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
