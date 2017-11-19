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

.. automethod:: geopandas.GeoSeries.rotate

.. automethod:: geopandas.GeoSeries.scale

.. automethod:: geopandas.GeoSeries.skew

.. automethod:: geopandas.GeoSeries.translate

`Aggregating methods`

.. autoattribute:: geopandas.GeoSeries.unary_union

Additionally, the following methods are implemented:

.. automethod:: geopandas.GeoSeries.from_file

.. automethod:: geopandas.GeoSeries.to_crs

.. automethod:: geopandas.GeoSeries.plot

.. autoattribute:: geopandas.GeoSeries.total_bounds

.. autoattribute:: geopandas.GeoSeries.__geo_interface__

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

.. automethod:: geopandas.GeoDataFrame.from_file

.. automethod:: geopandas.GeoDataFrame.from_postgis

.. automethod:: geopandas.GeoDataFrame.to_crs

.. automethod:: geopandas.GeoDataFrame.to_file

.. automethod:: geopandas.GeoDataFrame.to_json

.. automethod:: geopandas.GeoDataFrame.plot

.. autoattribute:: geopandas.GeoDataFrame.__geo_interface__

All pandas ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``geometry`` column and may not
return a ``GeoDataFrame`` result even when it would be appropriate to
do so.

API Pages
---------

.. currentmodule:: geopandas
.. autosummary::
  :template: autosummary.rst
  :toctree: reference/

  GeoDataFrame
  GeoSeries
  overlay
  read_file
  sjoin
  tools.geocode
  datasets.get_path
