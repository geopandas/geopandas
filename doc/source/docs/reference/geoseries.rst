=========
GeoSeries
=========
.. currentmodule:: geopandas

Constructor
-----------
.. autosummary::
   :toctree: api/

   GeoSeries

General methods and attributes
------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.area
   GeoSeries.bounds
   GeoSeries.total_bounds
   GeoSeries.length
   GeoSeries.geom_type
   GeoSeries.distance
   GeoSeries.representative_point
   GeoSeries.exterior
   GeoSeries.interiors
   GeoSeries.x
   GeoSeries.y

Unary predicates
----------------

.. autosummary::
   :toctree: api/

   GeoSeries.is_empty
   GeoSeries.is_ring
   GeoSeries.is_simple
   GeoSeries.is_valid
   GeoSeries.has_z


Binary Predicates
-----------------

.. autosummary::
   :toctree: api/

   GeoSeries.geom_almost_equals
   GeoSeries.contains
   GeoSeries.crosses
   GeoSeries.disjoint
   GeoSeries.geom_equals
   GeoSeries.intersects
   GeoSeries.overlaps
   GeoSeries.touches
   GeoSeries.within
   GeoSeries.covers
   GeoSeries.covered_by


Set-theoretic Methods
---------------------

.. autosummary::
   :toctree: api/

   GeoSeries.difference
   GeoSeries.intersection
   GeoSeries.symmetric_difference
   GeoSeries.union

Constructive Methods and Attributes
-----------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.buffer
   GeoSeries.boundary
   GeoSeries.centroid
   GeoSeries.convex_hull
   GeoSeries.envelope
   GeoSeries.simplify

Affine transformations
----------------------

.. autosummary::
   :toctree: api/

   GeoSeries.affine_transform
   GeoSeries.rotate
   GeoSeries.scale
   GeoSeries.skew
   GeoSeries.translate

Aggregating and exploding
-------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.unary_union
   GeoSeries.explode

Reading and writing files
-------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.from_file
   GeoSeries.to_file
   GeoSeries.to_json

Projection handling
-------------------

.. autosummary::
   :toctree: api/

   GeoSeries.crs
   GeoSeries.set_crs
   GeoSeries.to_crs
   GeoSeries.estimate_utm_crs

Missing values
--------------

.. autosummary::
   :toctree: api/

   GeoSeries.fillna
   GeoSeries.isna
   GeoSeries.notna

Plotting
--------

.. autosummary::
   :toctree: api/

   GeoSeries.plot


Spatial index
-------------

.. autosummary::
   :toctree: api/

   GeoSeries.sindex
   GeoSeries.has_sindex

Interface
---------

.. autosummary::
   :toctree: api/

   GeoSeries.__geo_interface__


Methods of pandas ``Series`` objects are also available, although not
all are applicable to geometric objects and some may return a
``Series`` rather than a ``GeoSeries`` result when appropriate. The methods
``isna()`` and ``fillna()`` have been
implemented specifically for ``GeoSeries`` and are expected to work
correctly.
