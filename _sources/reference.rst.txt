

Reference
===========================

The following Shapely methods and attributes are available on
``GeoSeries`` objects:

.. attribute:: GeoSeries.area

  Returns a ``Series`` containing the area of each geometry in the ``GeoSeries``.

.. attribute:: GeoSeries.bounds

  Returns a ``DataFrame`` with columns ``minx``, ``miny``, ``maxx``,
  ``maxy`` values containing the bounds for each geometry.
  (see ``GeoSeries.total_bounds`` for the limits of the entire series).

.. attribute:: GeoSeries.length

  Returns a ``Series`` containing the length of each geometry.

.. attribute:: GeoSeries.geom_type

  Returns a ``Series`` of strings specifying the `Geometry Type` of
  each object.

.. method:: GeoSeries.distance(other)

  Returns a ``Series`` containing the minimum distance to the `other`
  ``GeoSeries`` (elementwise) or geometric object.

.. method:: GeoSeries.representative_point()

  Returns a ``GeoSeries`` of (cheaply computed) points that are
  guaranteed to be within each geometry.

.. attribute:: GeoSeries.exterior

  Returns a ``GeoSeries`` of LinearRings representing the outer
  boundary of each polygon in the GeoSeries.  (Applies to GeoSeries
  containing only Polygons).

.. attribute:: GeoSeries.interiors

  Returns a ``GeoSeries`` of InteriorRingSequences representing the
  inner rings of each polygon in the GeoSeries.  (Applies to GeoSeries
  containing only Polygons).

`Unary Predicates`

.. attribute:: GeoSeries.is_empty

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
  empty geometries.

.. attribute:: GeoSeries.is_ring

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
  features that are closed.

.. attribute:: GeoSeries.is_simple

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
  geometries that do not cross themselves (meaningful only for
  `LineStrings` and `LinearRings`).

.. attribute:: GeoSeries.is_valid

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` for
  geometries that are valid.

`Binary Predicates`

.. method:: GeoSeries.almost_equals(other[, decimal=6])

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  each object is approximately equal to the `other` at all
  points to specified `decimal` place precision.  (See also :meth:`equals`)

.. method:: GeoSeries.contains(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  each object's `interior` contains the `boundary` and
  `interior` of the other object and their boundaries do not touch at all.

.. method:: GeoSeries.crosses(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  the `interior` of each object intersects the `interior` of
  the other but does not contain it, and the dimension of the intersection is
  less than the dimension of the one or the other.

.. method:: GeoSeries.disjoint(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  the `boundary` and `interior` of each object does not
  intersect at all with those of the other.

.. method:: GeoSeries.equals(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  if the set-theoretic `boundary`, `interior`, and `exterior`
  of each object coincides with those of the other.

.. method:: GeoSeries.intersects(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  if the `boundary` and `interior` of each object intersects in
  any way with those of the other.

.. method:: GeoSeries.touches(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  the objects have at least one point in common and their
  interiors do not intersect with any part of the other.

.. method:: GeoSeries.within(other)

  Returns a ``Series`` of ``dtype('bool')`` with value ``True`` if
  each object's `boundary` and `interior` intersect only
  with the `interior` of the other (not its `boundary` or `exterior`).
  (Inverse of :meth:`contains`)

`Set-theoretic Methods`

.. method:: GeoSeries.difference(other)

  Returns a ``GeoSeries`` of the points in each geometry that
  are not in the *other* object.

.. method:: GeoSeries.intersection(other)

  Returns a ``GeoSeries`` of the intersection of each object with the `other`
  geometric object.

.. method:: GeoSeries.symmetric_difference(other)

  Returns a ``GeoSeries`` of the points in each object not in the `other`
  geometric object, and the points in the `other` not in this object.

.. method:: GeoSeries.union(other)

  Returns a ``GeoSeries`` of the union of points from each object and the
  `other` geometric object.

`Constructive Methods`

.. method:: GeoSeries.buffer(distance, resolution=16)

  Returns a ``GeoSeries`` of geometries representing all points within a given `distance`
  of each geometric object.

.. attribute:: GeoSeries.boundary

  Returns a ``GeoSeries`` of lower dimensional objects representing
  each geometries's set-theoretic `boundary`.

.. attribute:: GeoSeries.centroid

  Returns a ``GeoSeries`` of points for each geometric centroid.

.. attribute:: GeoSeries.convex_hull

  Returns a ``GeoSeries`` of geometries representing the smallest
  convex `Polygon` containing all the points in each object unless the
  number of points in the object is less than three. For two points,
  the convex hull collapses to a `LineString`; for 1, a `Point`.

.. attribute:: GeoSeries.envelope

  Returns a ``GeoSeries`` of geometries representing the point or
  smallest rectangular polygon (with sides parallel to the coordinate
  axes) that contains each object.

.. method:: GeoSeries.simplify(tolerance, preserve_topology=True)

  Returns a ``GeoSeries`` containing a simplified representation of
  each object.

`Affine transformations`

.. method:: GeoSeries.rotate(self, angle, origin='center', use_radians=False)

  Rotate the coordinates of the GeoSeries.

.. method:: GeoSeries.scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin='center')

 Scale the geometries of the GeoSeries along each (x, y, z) dimensio.

.. method:: GeoSeries.skew(self, angle, origin='center', use_radians=False)

  Shear/Skew the geometries of the GeoSeries by angles along x and y dimensions.

.. method:: GeoSeries.translate(self, angle, origin='center', use_radians=False)

  Shift the coordinates of the GeoSeries.

`Aggregating methods`

.. attribute:: GeoSeries.unary_union

  Return a geometry containing the union of all geometries in the ``GeoSeries``.

Additionally, the following methods are implemented:

.. method:: GeoSeries.from_file()

  Load a ``GeoSeries`` from a file from any format recognized by
  `fiona`_.

.. method:: GeoSeries.to_crs(crs=None, epsg=None)

  Transform all geometries in a GeoSeries to a different coordinate
  reference system.  The ``crs`` attribute on the current GeoSeries
  must be set.  Either ``crs`` in dictionary form or an EPSG code may
  be specified for output.

  This method will transform all points in all objects.  It has no
  notion or projecting entire geometries.  All segments joining points
  are assumed to be lines in the current projection, not geodesics.
  Objects crossing the dateline (or other projection boundary) will
  have undesirable behavior.

.. method:: GeoSeries.plot(colormap='Set1', alpha=0.5, axes=None)

  Generate a plot of the geometries in the ``GeoSeries``.
  ``colormap`` can be any recognized by matplotlib, but discrete
  colormaps such as ``Accent``, ``Dark2``, ``Paired``, ``Pastel1``,
  ``Pastel2``, ``Set1``, ``Set2``, or ``Set3`` are recommended.
  Wraps the ``plot_series()`` function.

.. attribute:: GeoSeries.total_bounds

  Returns a tuple containing ``minx``, ``miny``, ``maxx``,
  ``maxy`` values for the bounds of the series as a whole.
  See ``GeoSeries.bounds`` for the bounds of the geometries contained
  in the series.

.. attribute:: GeoSeries.__geo_interface__

  Implements the `geo_interface`_. Returns a python data structure
  to represent the ``GeoSeries`` as a GeoJSON-like ``FeatureCollection``. 
  Note that the features will have an empty ``properties`` dict as they don't
  have associated attributes (geometry only).

Methods of pandas ``Series`` objects are also available, although not
all are applicable to geometric objects and some may return a
``Series`` rather than a ``GeoSeries`` result.  The methods
``copy()``, ``align()``, ``isnull()`` and ``fillna()`` have been
implemented specifically for ``GeoSeries`` and are expected to work
correctly.

GeoDataFrame
------------

A ``GeoDataFrame`` is a tablular data structure that contains a column
called ``geometry`` which contains a `GeoSeries``.

Currently, the following methods are implemented for a ``GeoDataFrame``:

.. classmethod:: GeoDataFrame.from_file(filename, **kwargs)

  Load a ``GeoDataFrame`` from a file from any format recognized by
  `fiona`_.  See ``read_file()``.

.. classmethod:: GeoDataFrame.from_postgis(sql, con, geom_col='geom', crs=None, index_col=None, coerce_float=True, params=None)

  Load a ``GeoDataFrame`` from a file from a PostGIS database.
  See ``read_postgis()``.

.. method:: GeoSeries.to_crs(crs=None, epsg=None, inplace=False)

  Transform all geometries in the ``geometry`` column of a
  GeoDataFrame to a different coordinate reference system.  The
  ``crs`` attribute on the current GeoSeries must be set.  Either
  ``crs`` in dictionary form or an EPSG code may be specified for
  output.  If ``inplace=True`` the geometry column will be replaced in
  the current dataframe, otherwise a new GeoDataFrame will be returned.

  This method will transform all points in all objects.  It has no
  notion or projecting entire geometries.  All segments joining points
  are assumed to be lines in the current projection, not geodesics.
  Objects crossing the dateline (or other projection boundary) will
  have undesirable behavior.

.. method:: GeoSeries.to_file(filename, driver="ESRI Shapefile", **kwargs)

  Write the ``GeoDataFrame`` to a file.  By default, an ESRI shapefile
  is written, but any OGR data source supported by Fiona can be
  written.  ``**kwargs`` are passed to the Fiona driver.

.. method:: GeoSeries.to_json(**kwargs)

  Returns a GeoJSON representation of the ``GeoDataFrame`` as a string.

.. method:: GeoDataFrame.plot(column=None, colormap=None, alpha=0.5, categorical=False, legend=False, axes=None)

  Generate a plot of the geometries in the ``GeoDataFrame``.  If the
  ``column`` parameter is given, colors plot according to values in
  that column, otherwise calls ``GeoSeries.plot()`` on the
  ``geometry`` column.  Wraps the ``plot_dataframe()`` function.

.. attribute:: GeoDataFrame.__geo_interface__

  Implements the `geo_interface`_. Returns a python data structure
  to represent the ``GeoDataFrame`` as a GeoJSON-like ``FeatureCollection``.

All pandas ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``geometry`` column and may not
return a ``GeoDataFrame`` result even when it would be appropriate to
do so.
