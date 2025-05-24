.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas
   import pandas as pd


Merging data
=========================================

There are two ways to combine datasets in GeoPandas -- attribute joins and spatial joins.

In an attribute join, a :class:`GeoSeries` or :class:`GeoDataFrame` is
combined with a regular :class:`pandas.Series` or :class:`pandas.DataFrame` based on a
common variable. This is analogous to normal merging or joining in *pandas*.

In a spatial join, observations from two :class:`GeoSeries` or :class:`GeoDataFrame`
are combined based on their spatial relationship to one another.

In the following examples, these datasets are used:

.. ipython:: python

   import geodatasets

   chicago = geopandas.read_file(geodatasets.get_path("geoda.chicago_commpop"))
   groceries = geopandas.read_file(geodatasets.get_path("geoda.groceries"))

   # For attribute join
   chicago_shapes = chicago[['geometry', 'NID']]
   chicago_names = chicago[['community', 'NID']]

   # For spatial join
   chicago = chicago[['geometry', 'community']].to_crs(groceries.crs)


Appending
---------

Appending :class:`GeoDataFrame` and :class:`GeoSeries` uses pandas :func:`~pandas.concat` function.
Keep in mind, that appended geometry columns needs to have the same CRS.

.. ipython:: python

    # Appending GeoSeries
    joined = pd.concat([chicago.geometry, groceries.geometry])

    # Appending GeoDataFrames
    douglas = chicago[chicago.community == 'DOUGLAS']
    oakland = chicago[chicago.community == 'OAKLAND']
    douglas_oakland = pd.concat([douglas, oakland])


Attribute joins
----------------

Attribute joins are accomplished using the :meth:`~pandas.DataFrame.merge` method. In general, it is recommended
to use the ``merge()`` method called from the spatial dataset. With that said, the stand-alone
:func:`pandas.merge` function will work if the :class:`GeoDataFrame` is in the ``left`` argument;
if a :class:`~pandas.DataFrame` is in the ``left`` argument and a :class:`GeoDataFrame`
is in the ``right`` position, the result will no longer be a :class:`GeoDataFrame`.

For example, consider the following merge that adds full names to a :class:`GeoDataFrame`
that initially has only area ID for each geometry by merging it with a :class:`~pandas.DataFrame`.

.. ipython:: python

   # `chicago_shapes` is GeoDataFrame with community shapes and area IDs
   chicago_shapes.head()

   # `chicago_names` is DataFrame with community names and area ID
   chicago_names.head()

   # Merge with `merge` method on shared variable (area ID):
   chicago_shapes = chicago_shapes.merge(chicago_names, on='NID')
   chicago_shapes.head()

.. _mergingdata.spatial-joins:

Spatial joins
----------------

In a spatial join, two geometry objects are merged based on their spatial relationship to one another.

.. ipython:: python


   # One GeoDataFrame of communities, one of grocery stores.
   # Want to merge to get each grocery's community.
   chicago.head()
   groceries.head()

   # Execute spatial join

   groceries_with_community = groceries.sjoin(chicago, how="inner", predicate='intersects')
   groceries_with_community.head()


GeoPandas provides two spatial-join functions:

- :meth:`GeoDataFrame.sjoin`: joins based on binary predicates (intersects, contains, etc.)
- :meth:`GeoDataFrame.sjoin_nearest`: joins based on proximity, with the ability to set a maximum search radius.

.. note::
   For historical reasons, both methods are also available as top-level functions :func:`sjoin` and :func:`sjoin_nearest`.
   It is recommended to use methods as the functions may be deprecated in the future.

Binary predicate joins
~~~~~~~~~~~~~~~~~~~~~~

Binary predicate joins are available via :meth:`GeoDataFrame.sjoin`.

:meth:`GeoDataFrame.sjoin` has two core arguments: ``how`` and ``predicate``.

**predicate**

The ``predicate`` argument specifies how GeoPandas decides whether or not to join the attributes of one
object to another, based on their geometric relationship.

The values for ``predicate`` correspond to the names of geometric binary predicates and depend on the spatial
index implementation.

The default spatial index in GeoPandas currently supports the following values for ``predicate`` which are
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

Nearest joins
~~~~~~~~~~~~~

Proximity-based joins can be done via :meth:`GeoDataFrame.sjoin_nearest`.

:meth:`GeoDataFrame.sjoin_nearest` shares the ``how`` argument with :meth:`GeoDataFrame.sjoin`, and
includes two additional arguments: ``max_distance`` and ``distance_col``.

**max_distance**

The ``max_distance`` argument specifies a maximum search radius for matching geometries. This can have a considerable performance impact in some cases.
If you can, it is highly recommended that you use this parameter.

**distance_col**

If set, the resultant GeoDataFrame will include a column with this name containing the computed distances between an input geometry and the nearest geometry.
