.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd


Indexing and Selecting Data
===========================

GeoPandas inherits the standard ``pandas`` methods for indexing/selecting data. This includes label based indexing with ``.loc`` and integer position based indexing with ``.iloc``, which apply to both ``GeoSeries`` and ``GeoDataFrame`` objects. For more information on indexing/selecting, see the pandas_ documentation.

.. _pandas: http://pandas.pydata.org/pandas-docs/stable/indexing.html

In addition to the standard ``pandas`` methods, GeoPandas also provides
coordinate based indexing with the ``cx`` indexer, which slices using a bounding
box. Geometries in the ``GeoSeries`` or ``GeoDataFrame`` that intersect the
bounding box will be returned.

Using the ``world`` dataset, we can use this functionality to quickly select all
countries whose boundaries extend into the southern hemisphere.

.. ipython:: python

   world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
   southern_world = world.cx[:, :0]
   @savefig world_southern.png width=5in
   southern_world.plot();

