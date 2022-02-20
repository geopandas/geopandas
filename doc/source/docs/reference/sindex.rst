=============
Spatial index
=============
.. currentmodule:: geopandas

GeoPandas offers built-in support for spatial indexing using an R-Tree algorithm.
Depending on the ability to import ``pygeos``, GeoPandas will either use
``pygeos.STRtree`` or ``rtree.index.Index``. The main interface for both is the
same and follows the ``pygeos`` model.

``GeoSeries.sindex`` creates a spatial index, which can use the methods and
properties documented below.

Constructor
-----------
.. autosummary::
   :toctree: api/

    GeoSeries.sindex

Spatial Index object
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
    query_bulk
    size
    valid_query_predicates

The concrete implementations currently available are
``geopandas.sindex.PyGEOSSTRTreeIndex`` and ``geopandas.sindex.RTreeIndex``.

In addition to the methods listed above, the ``rtree``-based spatial index
(``geopandas.sindex.RTreeIndex``) offers the full capability of
``rtree.index.Index`` - see the full API in the `rtree documentation`_.

Similarly, the ``pygeos``-based spatial index
(``geopandas.sindex.PyGEOSSTRTreeIndex``) offers the full capability of
``pygeos.STRtree``, including nearest-neighbor queries.
See the full API in the `PyGEOS STRTree documentation`_.

.. _rtree documentation: https://rtree.readthedocs.io/en/stable/class.html
.. _PyGEOS STRTree documentation: https://pygeos.readthedocs.io/en/latest/strtree.html
