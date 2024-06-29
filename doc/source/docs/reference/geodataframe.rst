============
GeoDataFrame
============
.. currentmodule:: geopandas

A ``GeoDataFrame`` is a tabular data structure that contains at least
one ``GeoSeries`` column storing geometry.

Constructor
-----------
.. autosummary::
   :toctree: api/

   GeoDataFrame

Serialization / IO / conversion
-------------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.from_file
   GeoDataFrame.from_features
   GeoDataFrame.from_postgis
   GeoDataFrame.from_arrow
   GeoDataFrame.to_file
   GeoDataFrame.to_json
   GeoDataFrame.to_geo_dict
   GeoDataFrame.to_parquet
   GeoDataFrame.to_arrow
   GeoDataFrame.to_feather
   GeoDataFrame.to_postgis
   GeoDataFrame.to_wkb
   GeoDataFrame.to_wkt

Projection handling
-------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.crs
   GeoDataFrame.set_crs
   GeoDataFrame.to_crs
   GeoDataFrame.estimate_utm_crs

Active geometry handling
------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.rename_geometry
   GeoDataFrame.set_geometry
   GeoDataFrame.active_geometry_name

Aggregating and exploding
-------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.dissolve
   GeoDataFrame.explode

Spatial joins
-------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.sjoin
   GeoDataFrame.sjoin_nearest

Overlay operations
------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.clip
   GeoDataFrame.overlay

Plotting
--------

.. autosummary::
   :toctree: api/

   GeoDataFrame.explore


.. autosummary::
   :toctree: api/
   :template: accessor_callable.rst

   GeoDataFrame.plot

Spatial index
-------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.sindex
   GeoDataFrame.has_sindex

Indexing
--------

.. autosummary::
   :toctree: api/

   GeoDataFrame.cx

Interface
---------

.. autosummary::
   :toctree: api/

   GeoDataFrame.__geo_interface__
   GeoDataFrame.iterfeatures

All pandas ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``geometry`` column. All methods
listed in `GeoSeries <geoseries>`__ work directly on an active geometry column of GeoDataFrame.
