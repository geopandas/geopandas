About GeoPandas
---------------

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
===========

The goal of GeoPandas is to make working with geospatial data in
python easier.  It combines the capabilities of pandas and shapely,
providing geospatial operations in pandas and a high-level interface
to multiple geometries to shapely.  GeoPandas enables you to easily do
operations in python that would otherwise require a spatial database
such as PostGIS.
