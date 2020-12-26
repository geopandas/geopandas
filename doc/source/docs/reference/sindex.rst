=============
Spatial index
=============
.. currentmodule:: geopandas

GeoPandas offers built-in support of spatial indexing using an R-Tree algorithm. Depending on the ability to import ``pygeos``, GeoPandas will either use ``pygeos.STRtree`` or ``rtree.index.Index``. The main interface for both is the same and follows the ``pygeos`` model. 

``GeoSeries.sindex`` creates a spatial index, which can use the methods and properties documented below.

Constructor
-----------
.. autosummary::
   :toctree: api/

    GeoSeries.sindex

PyGEOS STRtree
--------------
.. currentmodule:: geopandas.sindex.PyGEOSSTRTreeIndex
.. autosummary::
   :toctree: api/

    intersection
    is_empty
    query
    query_bulk
    size
    valid_query_predicates

rtree Rtree
-----------
.. currentmodule:: geopandas.sindex.RTreeIndex
.. autosummary::
   :toctree: api/

    intersection
    is_empty
    query
    query_bulk
    size
    valid_query_predicates

Furthermore, the ``rtree``-based spatial index offers full capability of ``rtree.index.Index`` - see the full API in the `rtree documentation`_.

.. _rtree documentation: https://rtree.readthedocs.io/en/stable/class.html
