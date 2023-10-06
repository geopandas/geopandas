=============
Spatial index
=============
.. currentmodule:: geopandas

GeoPandas will use the STRtree implementation provided by the Shapely
(``shapely.STRtree``)


``GeoSeries.sindex`` creates a spatial index, which can use the methods and
properties documented below.

Constructor
-----------
.. autosummary::
   :toctree: api/

    GeoSeries.sindex

Spatial index object
--------------------

The spatial index object returned from :attr:`GeoSeries.sindex` has the following
methods:

.. currentmodule:: geopandas.sindex.SpatialIndex
.. autosummary::
   :toctree: api/

    intersection
    is_empty
    nearest
    query
    size
    valid_query_predicates

The spatial index offers the full capability of :class:`shapely.STRtree`.