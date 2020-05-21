.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Merging Data
=========================================

There are two ways to combine datasets in *geopandas* -- attribute joins and spatial joins.

In an attribute join, a ``GeoSeries`` or ``GeoDataFrame`` is combined with a regular *pandas* ``Series`` or ``DataFrame`` based on a common variable. This is analogous to normal merging or joining in *pandas*.

In a Spatial Join, observations from two ``GeoSeries`` or ``GeoDataFrames`` are combined based on their spatial relationship to one another.

In the following examples, we use these datasets:

.. ipython:: python

   world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
   cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))

   # For attribute join
   country_shapes = world[['geometry', 'iso_a3']]
   country_names = world[['name', 'iso_a3']]

   # For spatial join
   countries = world[['geometry', 'name']]
   countries = countries.rename(columns={'name':'country'})


Appending
---------

Appending GeoDataFrames and GeoSeries uses pandas ``append`` methods. Keep in mind, that appended geometry columns needs to have the same CRS.

.. ipython:: python

    # Appending GeoSeries
    joined = world.geometry.append(cities.geometry)

    # Appending GeoDataFrames
    europe = world[world.continent == 'Europe']
    asia = world[world.continent == 'Asia']
    eurasia = europe.append(asia)


Attribute Joins
----------------

Attribute joins are accomplished using the ``merge`` method. In general, it is recommended to use the ``merge`` method called from the spatial dataset. With that said, the stand-alone ``merge`` function will work if the GeoDataFrame is in the ``left`` argument; if a DataFrame is in the ``left`` argument and a GeoDataFrame is in the ``right`` position, the result will no longer be a GeoDataFrame.


For example, consider the following merge that adds full names to a ``GeoDataFrame`` that initially has only ISO codes for each country by merging it with a *pandas* ``DataFrame``.

.. ipython:: python

   # `country_shapes` is GeoDataFrame with country shapes and iso codes
   country_shapes.head()

   # `country_names` is DataFrame with country names and iso codes
   country_names.head()

   # Merge with `merge` method on shared variable (iso codes):
   country_shapes = country_shapes.merge(country_names, on='iso_a3')
   country_shapes.head()



Spatial Joins
----------------

In a Spatial Join, two geometry objects are merged based on their spatial relationship to one another.

.. ipython:: python


   # One GeoDataFrame of countries, one of Cities.
   # Want to merge so we can get each city's country.
   countries.head()
   cities.head()

   # Execute spatial join

   cities_with_country = geopandas.sjoin(cities, countries, how="inner", op='intersects')
   cities_with_country.head()


Sjoin Arguments
~~~~~~~~~~~~~~~~

``sjoin()`` has two core arguments: ``how`` and ``op``.

**op**

The ``op`` argument specifies how ``geopandas`` decides whether or not to join the attributes of one object to another. There are three different join options as follows:

* `intersects`: The attributes will be joined if the boundary and interior of the object intersect in any way with the boundary and/or interior of the other object.
* `within`: The attributes will be joined if the object’s boundary and interior intersect *only* with the interior of the other object (not its boundary or exterior).
* `contains`: The attributes will be joined if the object’s interior contains the boundary and interior of the other object and their boundaries do not touch at all.

You can read more about each join type in the `Shapely documentation <http://shapely.readthedocs.io/en/latest/manual.html#binary-predicates>`__.

**how**

The `how` argument specifies the type of join that will occur and which geometry is retained in the resultant geodataframe. It accepts the following options:

* ``left``: use the index from the first (or `left_df`) geodataframe that you provide to ``sjoin``; retain only the `left_df` geometry column
* ``right``: use index from second (or `right_df`); retain only the `right_df` geometry column
* ``inner``: use intersection of index values from both geodataframes; retain only the `left_df` geometry column

Note more complicated spatial relationships can be studied by combining geometric operations with spatial join. To find all polygons within a given distance of a point, for example, one can first use the ``buffer`` method to expand each point into a circle of appropriate radius, then intersect those buffered circles with the polygons in question.


Sjoin Performance
~~~~~~~~~~~~~~~~~~

Existing spatial indexes on either `left_df` or `right_df` will be reused when performing an ``sjoin``. If neither df has a spatial index, a spatial index will be generated for the longer df. If both have a spatial index, the `right_df`'s index will be used preferentially. Performance of multiple sjoins in a row involving a common GeoDataFrame may be improved by pre-generating the spatial index of the common GeoDataFrame prior to performing sjoins using ``df1.sindex``.

.. code-block:: python

    df1 = # a GeoDataFrame with data
    df2 = # a second GeoDataFrame
    df3 = # a third GeoDataFrame

    # pre-generate sindex on df1 if it doesn't already exist
    df1.sindex

    sjoin(df1, df2, ...)
    # sindex for df1 is reused
    sjoin(df1, df3, ...)
    # sindex for df1 is reused again
