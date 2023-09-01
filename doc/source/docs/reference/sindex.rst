=============
Spatial index
=============
.. currentmodule:: geopandas

GeoPandas offers built-in support for spatial indexing using an R-Tree algorithm.
Depending on the ability to import PyGEOS, GeoPandas will either use ``shapely.STRtree``,
``pygeos.STRtree`` or ``rtree.index.Index``. The main interface for both is the
same and follows the Shapely 2.0 model.

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

The concrete implementations currently available are
``geopandas.sindex.PyGEOSSTRTreeIndex`` (when using Shapely 2.0 or PyGEOS) and
``geopandas.sindex.RTreeIndex`` (when using Shapely <2).

In addition to the methods listed above, the ``rtree``-based spatial index
(``geopandas.sindex.RTreeIndex``) offers the full capability of
``rtree.index.Index`` - see the full API in the `rtree documentation`_.

Similarly, the PyGEOS-based spatial index
(``geopandas.sindex.PyGEOSSTRTreeIndex``) offers the full capability of
``pygeos.STRtree/shapely.STRtree``, including nearest-neighbor queries.
See the full API in the `Shapely STRTree documentation`_ or `PyGEOS STRTree documentation`_.

.. _rtree documentation: https://rtree.readthedocs.io/en/stable/class.html
.. _PyGEOS STRTree documentation: https://pygeos.readthedocs.io/en/latest/strtree.html
.. _Shapely STRTree documentation: https://shapely.readthedocs.io/en/stable/strtree.html