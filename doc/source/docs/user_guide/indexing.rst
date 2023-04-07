.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Indexing and selecting data
===========================

GeoPandas inherits the standard pandas_ methods for indexing/selecting data. This includes label based indexing with :attr:`~pandas.DataFrame.loc` and integer position based indexing with :attr:`~pandas.DataFrame.iloc`, which apply to both :class:`GeoSeries` and :class:`GeoDataFrame` objects. For more information on indexing/selecting, see the pandas_ documentation.

.. _pandas: http://pandas.pydata.org/pandas-docs/stable/indexing.html

In addition to the standard pandas_ methods, GeoPandas also provides
coordinate based indexing with the :attr:`~GeoDataFrame.cx` indexer, which slices using a bounding
box. Geometries in the :class:`GeoSeries` or :class:`GeoDataFrame` that intersect the
bounding box will be returned.

Using the ``geoda.chile_labor`` dataset, you can use this functionality to quickly select parts
of Chile whose boundaries extend south of the -50 degrees latitude. You can first check the original GeoDataFrame.

.. ipython:: python

   import geodatasets

   chile = geopandas.read_file(geodatasets.get_path('geoda.chile_labor'))
   @savefig chile.png
   chile.plot(figsize=(8, 8));

And then select only the southern part of the country.

.. ipython:: python

   southern_chile = chile.cx[:, :-50]
   @savefig chile_southern.png
   southern_chile.plot(figsize=(8, 8));
