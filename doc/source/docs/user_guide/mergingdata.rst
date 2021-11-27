.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


Merging Data
=========================================

There are two ways to combine datasets in *geopandas* -- attribute joins and spatial joins.

In an attribute join, a :class:`GeoSeries` or :class:`GeoDataFrame` is
combined with a regular :class:`pandas.Series` or :class:`pandas.DataFrame` based on a
common variable. This is analogous to normal merging or joining in *pandas*.

In a Spatial Join, observations from two :class:`GeoSeries` or :class:`GeoDataFrame`
are combined based on their spatial relationship to one another.

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

Appending :class:`GeoDataFrame` and :class:`GeoSeries` uses pandas :meth:`~pandas.DataFrame.append` methods.
Keep in mind, that appended geometry columns needs to have the same CRS.

.. ipython:: python

    # Appending GeoSeries
    joined = world.geometry.append(cities.geometry)

    # Appending GeoDataFrames
    europe = world[world.continent == 'Europe']
    asia = world[world.continent == 'Asia']
    eurasia = europe.append(asia)


Attribute Joins
----------------

Attribute joins are accomplished using the :meth:`~pandas.DataFrame.merge` method. In general, it is recommended
to use the ``merge()`` method called from the spatial dataset. With that said, the stand-alone
:func:`pandas.merge` function will work if the :class:`GeoDataFrame` is in the ``left`` argument;
if a :class:`~pandas.DataFrame` is in the ``left`` argument and a :class:`GeoDataFrame`
is in the ``right`` position, the result will no longer be a :class:`GeoDataFrame`.

For example, consider the following merge that adds full names to a :class:`GeoDataFrame`
that initially has only ISO codes for each country by merging it with a :class:`~pandas.DataFrame`.

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

   cities_with_country = cities.sjoin(countries, how="inner", predicate='intersects')
   cities_with_country.head()


GeoPandas provides two spatial-join functions:

- :meth:`GeoDataFrame.sjoin`: joins based on binary predicates (intersects, contains, etc.)
- :meth:`GeoDataFrame.sjoin_nearest`: joins based on proximity, with the ability to set a maximum search radius.

.. note::
   For historical reasons, both methods are also available as top-level functions :func:`sjoin` and :func:`sjoin_nearest`.
   It is recommended to use methods as the functions may be deprecated in the future.

Binary Predicate Joins
~~~~~~~~~~~~~~~~~~~~~~

Binary predicate joins are available via :meth:`GeoDataFrame.sjoin`.

:meth:`GeoDataFrame.sjoin` has two core arguments: ``how`` and ``predicate``.

**predicate**

The ``predicate`` argument specifies how ``geopandas`` decides whether or not to join the attributes of one
object to another, based on their geometric relationship.

The values for ``predicate`` correspond to the names of geometric binary predicates and depend on the spatial
index implementation.

The default spatial index in ``geopandas`` currently supports the following values for ``predicate`` which are
defined in the
`Shapely documentation <http://shapely.readthedocs.io/en/latest/manual.html#binary-predicates>`__:

* `intersects`
* `contains`
* `within`
* `touches`
* `crosses`
* `overlaps`

**how**

The `how` argument specifies the type of join that will occur and which geometry is retained in the resultant
:class:`GeoDataFrame`. It accepts the following options:

* ``left``: use the index from the first (or `left_df`) :class:`GeoDataFrame` that you provide
  to :meth:`GeoDataFrame.sjoin`; retain only the `left_df` geometry column
* ``right``: use index from second (or `right_df`); retain only the `right_df` geometry column
* ``inner``: use intersection of index values from both :class:`GeoDataFrame`; retain only the `left_df` geometry column

Note more complicated spatial relationships can be studied by combining geometric operations with spatial join.
To find all polygons within a given distance of a point, for example, one can first use the :meth:`~geopandas.GeoSeries.buffer` method to expand each
point into a circle of appropriate radius, then intersect those buffered circles with the polygons in question.

Nearest Joins
~~~~~~~~~~~~~

Proximity-based joins can be done via :meth:`GeoDataFrame.sjoin_nearest`.

:meth:`GeoDataFrame.sjoin_nearest` shares the ``how`` argument with :meth:`GeoDataFrame.sjoin`, and
includes two additional arguments: ``max_distance`` and ``distance_col``.

**max_distance**

The ``max_distance`` argument specifies a maximum search radius for matching geometries. This can have a considerable performance impact in some cases.
If you can, it is highly recommended that you use this parameter.

**distance_col**

If set, the resultant GeoDataFrame will include a column with this name containing the computed distances between an input geometry and the nearest geometry.
