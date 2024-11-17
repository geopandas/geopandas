.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas


.. _missing-empty:

Missing and empty geometries
============================

GeoPandas supports, just like in pandas, the concept of missing values (NA
or null values). But for geometry values, there is an additional concept of
empty geometries:

- **Empty geometries** are actual geometry objects but that have no coordinates
  (and thus also no area, for example). They can for example originate from
  taking the intersection of two polygons that have no overlap.
  The scalar object (when accessing a single element of a GeoSeries) is still
  a Shapely geometry object.
- **Missing geometries** are unknown values in a GeoSeries. They will typically
  be propagated in operations (for example in calculations of the area or of
  the intersection), or ignored in reductions such as :meth:`~GeoSeries.union_all`.
  The scalar object (when accessing a single element of a GeoSeries) is the
  Python ``None`` object.

.. warning::

    Starting from GeoPandas v0.6.0, those two concepts are more consistently
    separated. See :ref:`below <missing-empty.changes-0.6.0>` for more details
    on what changed compared to earlier versions.


Consider the following example GeoSeries with one polygon, one missing value
and one empty polygon:

.. ipython:: python

    from shapely.geometry import Polygon
    s = geopandas.GeoSeries([Polygon([(0, 0), (1, 1), (0, 1)]), None, Polygon([])])
    s

In spatial operations, missing geometries will typically propagate (be missing
in the result as well), while empty geometries are treated as a geometry
and the result will depend on the operation:

.. ipython:: python

    s.area
    s.union(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))
    s.intersection(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))

The :meth:`GeoSeries.isna` method will only check for missing values and not
for empty geometries:

.. ipython:: python
    :okwarning:

    s.isna()

On the other hand, if you want to know which values are empty geometries,
you can use the :attr:`GeoSeries.is_empty` attribute:

.. ipython:: python

    s.is_empty

To get only the actual geometry objects that are neither missing nor empty,
you can use a combination of both:

.. ipython:: python
    :okwarning:

    s.is_empty | s.isna()
    s[~(s.is_empty | s.isna())]


.. _missing-empty.changes-0.6.0:

Changes since GeoPandas v0.6.0
------------------------------

In GeoPandas v0.6.0, the missing data handling was refactored and made more
consistent across the library.

Historically, missing ("NA") values in a GeoSeries could be represented by empty
geometric objects, in addition to standard representations such as ``None`` and
``np.nan``. At least, this was the case in :meth:`GeoSeries.isna` or when a
GeoSeries got aligned in geospatial operations. But, other methods like
:meth:`~GeoSeries.dropna` and :meth:`~GeoSeries.fillna` did not follow this
approach and did not consider empty geometries as missing.

In GeoPandas v0.6.0, the most important change is :meth:`GeoSeries.isna` no
longer treating empty as missing:

* Using the small example from above, the old behaviour treated both the
  empty as missing geometry as "missing":

  .. code-block:: python

    >>> s
    0    POLYGON ((0 0, 1 1, 0 1, 0 0))
    1                              None
    2          GEOMETRYCOLLECTION EMPTY
    dtype: object

    >>> s.isna()
    0    False
    1     True
    2     True
    dtype: bool

* Starting from GeoPandas v0.6.0, it will now only see actual missing values
  as missing:

  .. ipython:: python
    :okwarning:

    s.isna()

  For now, when ``isna()`` is called on a GeoSeries with empty geometries,
  a warning is raised to alert the user of the changed behaviour with an
  indication how to solve this.

Additionally, the behaviour of :meth:`GeoSeries.align` changed to use
missing values instead of empty geometries to fill non-matching indexes.
Consider the following small toy example:

.. ipython:: python

    from shapely.geometry import Point
    s1 = geopandas.GeoSeries([Point(0, 0), Point(1, 1)], index=[0, 1])
    s2 = geopandas.GeoSeries([Point(1, 1), Point(2, 2)], index=[1, 2])
    s1
    s2

* Previously, the ``align`` method would use empty geometries to fill
  values:

  .. code-block:: python

    >>> s1_aligned, s2_aligned = s1.align(s2)

    >>> s1_aligned
    0                 POINT (0 0)
    1                 POINT (1 1)
    2    GEOMETRYCOLLECTION EMPTY
    dtype: object

    >>> s2_aligned
    0    GEOMETRYCOLLECTION EMPTY
    1                 POINT (1 1)
    2                 POINT (2 2)
    dtype: object

  This method is used under the hood when performing spatial operations on
  mis-aligned GeoSeries objects:

  .. code-block:: python

    >>> s1.intersection(s2)
    0    GEOMETRYCOLLECTION EMPTY
    1                 POINT (1 1)
    2    GEOMETRYCOLLECTION EMPTY
    dtype: object

* Starting from GeoPandas v0.6.0, :meth:`GeoSeries.align` will use missing
  values to fill in the non-aligned indices, to be consistent with the
  behaviour in pandas:

  .. ipython:: python

    s1_aligned, s2_aligned = s1.align(s2)
    s1_aligned
    s2_aligned

  This has the consequence that spatial operations will also use missing
  values instead of empty geometries, which can have a different behaviour
  depending on the spatial operation:

  .. ipython:: python
    :okwarning:

    s1.intersection(s2)
