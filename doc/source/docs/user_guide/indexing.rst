.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Indexing and Selecting Data
===========================

GeoPandas inherits the standard pandas_ methods for indexing/selecting data. This includes label based indexing with :attr:`~pandas.DataFrame.loc` and integer position based indexing with :attr:`~pandas.DataFrame.iloc`, which apply to both :class:`GeoSeries` and :class:`GeoDataFrame` objects. For more information on indexing/selecting, see the pandas_ documentation.

.. _pandas: http://pandas.pydata.org/pandas-docs/stable/indexing.html

In addition to the standard pandas_ methods, GeoPandas also provides
coordinate based indexing with the :attr:`~GeoDataFrame.cx` indexer, which slices using a bounding
box. Geometries in the :class:`GeoSeries` or :class:`GeoDataFrame` that intersect the
bounding box will be returned.

Using the ``world`` dataset, we can use this functionality to quickly select all
countries whose boundaries extend into the southern hemisphere.

.. ipython:: python

   world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
   southern_world = world.cx[:, :0]
   @savefig world_southern.png
   southern_world.plot(figsize=(10, 3));
