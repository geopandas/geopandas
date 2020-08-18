=========
GeoSeries
=========
.. currentmodule:: geopandas

Constructor
-----------
.. autosummary::
   :toctree: geoseries/

   GeoSeries

General methods and attributes
------------------------------

.. autosummary::
   :toctree: geoseries/

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
   :toctree: geoseries/

   GeoSeries.is_empty
   GeoSeries.is_ring
   GeoSeries.is_simple
   GeoSeries.is_valid
   GeoSeries.has_z


Binary Predicates
-----------------

.. autosummary::
   :toctree: geoseries/

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
   :toctree: geoseries/

   GeoSeries.difference
   GeoSeries.intersection
   GeoSeries.symmetric_difference
   GeoSeries.union

Constructive Methods and Attributes
-----------------------------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.buffer
   GeoSeries.boundary
   GeoSeries.centroid
   GeoSeries.convex_hull
   GeoSeries.envelope
   GeoSeries.simplify

Affine transformations
----------------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.affine_transform
   GeoSeries.rotate
   GeoSeries.scale
   GeoSeries.skew
   GeoSeries.translate

Aggregating methods
-------------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.unary_union

Reading and writing files
-------------------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.from_file
   GeoSeries.to_file
   GeoSeries.to_json

Projection handling
-------------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.crs
   GeoSeries.set_crs
   GeoSeries.to_crs

Missing values
--------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.fillna
   GeoSeries.isna
   GeoSeries.notna

Plotting
--------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.plot


Spatial index
-------------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.sindex

Interface
---------

.. autosummary::
   :toctree: geoseries/

   GeoSeries.__geo_interface__


Methods of pandas ``Series`` objects are also available, although not
all are applicable to geometric objects and some may return a
``Series`` rather than a ``GeoSeries`` result when appropriate. The methods
``isna()`` and ``fillna()`` have been
implemented specifically for ``GeoSeries`` and are expected to work
correctly.
