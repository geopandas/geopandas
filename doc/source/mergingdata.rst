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

Most of the computation is done within the spatial index queries. The spatial index is always queried as follows:

.. code-block:: python

   right_df.sindex.query_bulk(left_df.geometry, op)

This translates to:

.. code-block:: python

   geom_in_left_df.op(geom_in_right_df)

For ``sjoin``, there are two concrete choices which can be made:

**Order of left_df and right_df**

For predicates where the order does not matter (ex: ``intersects``) simply flipping the GeoDataFrames and the ``how`` parameter can yield great performance improvements.

.. code-block:: python

   # changing
   sjoin(left_df=df1, right_df=df2, how="left", op='intersects')
   # to
   sjoin(left_df=df2, right_df=df1, how="right", op='intersects')
   # yields the same result but possibly different performance

There are many factors that may influence the performance of the query, some of which are:
* Complexity of geometries. Some operations may handle complex geometries better than others. For example, ``polygon.intersects(point)`` will perform differently than ``point.intersects(polygon)``. Swapping ``left_df`` and ``right_df`` will change which way the comparison is made.
* Partitioning of the data by the spatial index. Before doing predicate comparisons, the query uses a leaf-based spatial index to find geometries with intersecting bounds in logarithmic time. Since ``right_df`` is always the GeoDataFrame used as the spatial index, making ``right_df`` the longer GeoDataFrame may improve the performance of the query.

**Choice of op**

Since some operations have inverses (ex: ``within`` is the inverse of ``contains``), you can try flipping these operations along with the GeoDataFrames to speed up your spatial join.

.. code-block:: python

   # changing
   sjoin(left_df=df1, right_df=df2, how="left", op='within')
   # to
   sjoin(left_df=df2, right_df=df1, how="right", op='contains')
   # yields the same result but possibly different performance

For a more in-depth discussion, see the section below regarding spatial index performance in general.

Spatial Index Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

The performance of several ``GeoPandas`` functions (namely ``sjoin``, ``overlay`` and ``clip``) is largely determined by the spatial index query.
As discussed above, many factors may influence the performance of the query.
To offer some guidance, we have benchmarked queries across different spatial index implementations (``pygeos`` and ``rtree``) as well as different geometry types and predicates.
Keep in mind that these benchmarks are highly variable and may behave differently than your data. It is highly recommended that you do your own testing for best performance.


+--------------------------------------------------------+--------------+---------------+
| test(predicate, input\_geom\_type, tree\_geom\_type)   | rtree        | pygeos        |
+========================================================+==============+===============+
| query\_bulk('contains', 'mixed', 'mixed')              | 130±9ms      | 68.1±0.7ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'mixed', 'points')             | 43.1±0.7ms   | 557±20μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'mixed', 'polygons')           | 112±2ms      | 69.4±3ms      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'points', 'mixed')             | 41.5±6ms     | 21.3±0.5ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'points', 'points')            | 23.4±0.5ms   | 408±4μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'points', 'polygons')          | 27.0±2ms     | 20.9±0.5ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'polygons', 'mixed')           | 110±2ms      | 48.2±0.8ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'polygons', 'points')          | 30.2±2ms     | 150±7μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('contains', 'polygons', 'polygons')        | 96.6±2ms     | 49.5±1ms      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'mixed', 'mixed')               | 122±4ms      | 159±3ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'mixed', 'points')              | 42.2±4ms     | 19.7±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'mixed', 'polygons')            | 106±2ms      | 140±9ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'points', 'mixed')              | 39.1±0.8ms   | 21.2±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'points', 'points')             | 22.7±1ms     | 388±7μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'points', 'polygons')           | 30.3±2ms     | 20.6±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'polygons', 'mixed')            | 103±3ms      | 137±2ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'polygons', 'points')           | 25.2±0.5ms   | 19.2±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('crosses', 'polygons', 'polygons')         | 94.4±2ms     | 119±4ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'mixed', 'mixed')            | 49.3±1ms     | 59.1±3ms      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'mixed', 'points')           | 35.8±1ms     | 1.72±0.02ms   |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'mixed', 'polygons')         | 39.9±1ms     | 55.8±0.7ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'points', 'mixed')           | 32.0±1ms     | 664±20μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'points', 'points')          | 18.1±0.8ms   | 407±3μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'points', 'polygons')        | 24.9±0.8ms   | 200±20μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'polygons', 'mixed')         | 39.2±4ms     | 56.8±1ms      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'polygons', 'points')        | 22.3±0.2ms   | 1.32±0.04ms   |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('intersects', 'polygons', 'polygons')      | 27.9±0.8ms   | 55.4±0.5ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'mixed', 'mixed')              | 246±10ms     | 160±3ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'mixed', 'points')             | 59.9±1ms     | 20.8±1ms      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'mixed', 'polygons')           | 209±5ms      | 141±2ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'points', 'mixed')             | 66.2±5ms     | 21.1±0.2ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'points', 'points')            | 21.6±0.4ms   | 394±8μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'points', 'polygons')          | 52.6±2ms     | 20.5±0.6ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'polygons', 'mixed')           | 202±7ms      | 137±0.9ms     |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'polygons', 'points')          | 50.6±3ms     | 19.2±0.4ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('overlaps', 'polygons', 'polygons')        | 165±3ms      | 122±3ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'mixed', 'mixed')               | 249±8ms      | 157±2ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'mixed', 'points')              | 62.3±3ms     | 19.4±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'mixed', 'polygons')            | 212±4ms      | 139±2ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'points', 'mixed')              | 61.9±2ms     | 21.0±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'points', 'points')             | 22.1±0.3ms   | 397±20μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'points', 'polygons')           | 51.3±1ms     | 20.6±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'polygons', 'mixed')            | 199±4ms      | 137±2ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'polygons', 'points')           | 54.6±4ms     | 19.4±0.7ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('touches', 'polygons', 'polygons')         | 169±6ms      | 120±1ms       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'mixed', 'mixed')                | 78.1±1ms     | 12.6±0.2ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'mixed', 'points')               | 42.6±1ms     | 1.44±0.02ms   |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'mixed', 'polygons')             | 66.8±2ms     | 12.3±0.09ms   |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'points', 'mixed')               | 37.5±0.8ms   | 948±80μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'points', 'points')              | 22.3±0.7ms   | 114±1μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'points', 'polygons')            | 28.3±0.6ms   | 791±20μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'polygons', 'mixed')             | 61.6±3ms     | 11.7±0.3ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'polygons', 'points')            | 28.2±0.5ms   | 1.33±0.03ms   |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk('within', 'polygons', 'polygons')          | 48.6±0.5ms   | 11.4±0.7ms    |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'mixed', 'mixed')                    | 249±10ms     | 634±30μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'mixed', 'points')                   | 61.7±2ms     | 193±8μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'mixed', 'polygons')                 | 207±4ms      | 393±20μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'points', 'mixed')                   | 64.2±3ms     | 286±10μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'points', 'points')                  | 21.5±0.3ms   | 81.1±4μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'points', 'polygons')                | 50.9±0.6ms   | 177±10μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'polygons', 'mixed')                 | 197±4ms      | 392±2μs       |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'polygons', 'points')                | 48.6±4ms     | 133±10μs      |
+--------------------------------------------------------+--------------+---------------+
| query\_bulk(None, 'polygons', 'polygons')              | 165±2ms      | 238±20μs      |
+--------------------------------------------------------+--------------+---------------+

