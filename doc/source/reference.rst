.. _reference:

Reference
===========================

GeoSeries
---------

The following Shapely methods and attributes are available on
``GeoSeries`` objects:

.. autoattribute:: geopandas.GeoSeries.area

.. autoattribute:: geopandas.GeoSeries.bounds

.. autoattribute:: geopandas.GeoSeries.length

.. autoattribute:: geopandas.GeoSeries.geom_type

.. automethod:: geopandas.GeoSeries.distance

.. automethod:: geopandas.GeoSeries.representative_point

.. autoattribute:: geopandas.GeoSeries.exterior

.. autoattribute:: geopandas.GeoSeries.interiors

.. autoattribute:: geopandas.GeoSeries.x

.. autoattribute:: geopandas.GeoSeries.y

`Unary Predicates`

.. autoattribute:: geopandas.GeoSeries.is_empty

.. autoattribute:: geopandas.GeoSeries.is_ring

.. autoattribute:: geopandas.GeoSeries.is_simple

.. autoattribute:: geopandas.GeoSeries.is_valid

`Binary Predicates`

.. automethod:: geopandas.GeoSeries.geom_almost_equals

.. automethod:: geopandas.GeoSeries.contains

.. automethod:: geopandas.GeoSeries.crosses

.. automethod:: geopandas.GeoSeries.disjoint

.. automethod:: geopandas.GeoSeries.geom_equals

.. automethod:: geopandas.GeoSeries.intersects

.. automethod:: geopandas.GeoSeries.overlaps

.. automethod:: geopandas.GeoSeries.touches

.. automethod:: geopandas.GeoSeries.within

`Set-theoretic Methods`

.. automethod:: geopandas.GeoSeries.difference

.. automethod:: geopandas.GeoSeries.intersection

.. automethod:: geopandas.GeoSeries.symmetric_difference

.. automethod:: geopandas.GeoSeries.union

`Constructive Methods`

.. automethod:: geopandas.GeoSeries.buffer

.. autoattribute:: geopandas.GeoSeries.boundary

.. autoattribute:: geopandas.GeoSeries.centroid

.. autoattribute:: geopandas.GeoSeries.convex_hull

.. autoattribute:: geopandas.GeoSeries.envelope

.. automethod:: geopandas.GeoSeries.simplify

`Affine transformations`

.. automethod:: geopandas.GeoSeries.affine_transform

.. automethod:: geopandas.GeoSeries.rotate

.. automethod:: geopandas.GeoSeries.scale

.. automethod:: geopandas.GeoSeries.skew

.. automethod:: geopandas.GeoSeries.translate

`Aggregating methods`

.. autoattribute:: geopandas.GeoSeries.unary_union

Additionally, the following attributes and methods are implemented:

.. automethod:: geopandas.GeoSeries.from_file

.. automethod:: geopandas.GeoSeries.to_file

.. automethod:: geopandas.GeoSeries.to_json

.. autoattribute:: geopandas.GeoSeries.crs

.. automethod:: geopandas.GeoSeries.to_crs

.. automethod:: geopandas.GeoSeries.plot

.. autoattribute:: geopandas.GeoSeries.total_bounds

.. autoattribute:: geopandas.GeoSeries.__geo_interface__

.. automethod:: geopandas.GeoSeries.isna

.. automethod:: geopandas.GeoSeries.notna

.. automethod:: geopandas.GeoSeries.fillna


Methods of pandas ``Series`` objects are also available, although not
all are applicable to geometric objects and some may return a
``Series`` rather than a ``GeoSeries`` result.  The methods
``isna()`` and ``fillna()`` have been
implemented specifically for ``GeoSeries`` and are expected to work
correctly.

GeoDataFrame
------------

A ``GeoDataFrame`` is a tabular data structure that contains a column
called ``geometry`` which contains a `GeoSeries``.

Currently, the following methods/attributes are implemented for a ``GeoDataFrame``:

.. autoattribute:: geopandas.GeoDataFrame.crs

.. automethod:: geopandas.GeoDataFrame.to_crs

.. automethod:: geopandas.GeoDataFrame.from_file

.. automethod:: geopandas.GeoDataFrame.from_features

.. automethod:: geopandas.GeoDataFrame.from_postgis

.. automethod:: geopandas.GeoDataFrame.to_crs

.. automethod:: geopandas.GeoDataFrame.to_file

.. automethod:: geopandas.GeoDataFrame.to_json

.. automethod:: geopandas.GeoDataFrame.to_parquet

.. automethod:: geopandas.GeoDataFrame.to_postgis

.. automethod:: geopandas.GeoDataFrame.plot

.. automethod:: geopandas.GeoDataFrame.rename_geometry

.. automethod:: geopandas.GeoDataFrame.set_geometry

.. automethod:: geopandas.GeoDataFrame.explode

.. automethod:: geopandas.GeoDataFrame.dissolve

.. autoattribute:: geopandas.GeoDataFrame.__geo_interface__

All pandas ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``geometry`` column and may not
return a ``GeoDataFrame`` result even when it would be appropriate to
do so.

Testing
-------

GeoPandas includes specific functions to test its objects.

.. autofunction:: geopandas.testing.geom_equals

.. autofunction:: geopandas.testing.geom_almost_equals

.. autofunction:: geopandas.testing.assert_geoseries_equal

.. autofunction:: geopandas.testing.assert_geodataframe_equal


Top-level Functions
-------------------

.. currentmodule:: geopandas
.. autosummary::
  :template: autosummary.rst
  :toctree: reference/

  GeoDataFrame
  GeoSeries
  read_file
  read_parquet
  read_postgis
  sjoin
  overlay
  clip
  tools.geocode
  tools.collect
  points_from_xy
  datasets.get_path
