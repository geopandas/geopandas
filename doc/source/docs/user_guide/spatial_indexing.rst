.. currentmodule:: geopandas

Spatial indexing
================

When you want to know a spatial relationship (known as a spatial
predicate) between a set of geometries A and a geometry B (or a set of
them), you can compare geometry B against any geometry in a
set A. However, that is not the most performant approach
in most cases. A spatial index is a more efficient method for pre-filtering comparisons of geometries before using more computationally expensive spatial predicates.
GeoPandas exposes the
Sort-Tile-Recursive R-tree from shapely on any GeoDataFrame and
GeoSeries using the :attr:`GeoSeries.sindex` property. This page outlines its options
and common usage patterns.

Note that for many operations where a spatial index provides significant
performance benefits, GeoPandas already uses it automatically (like :meth:`~GeoDataFrame.sjoin`,
:meth:`~GeoDataFrame.overlay`, or :meth:`~GeoDataFrame.clip`). However, more advanced use cases may require
a direct interaction with the index.

.. ipython:: python

    import geopandas
    import matplotlib.pyplot as plt
    import shapely

    from geodatasets import get_path

Load data on New York City subboroughs to illustrate the spatial
indexing.

.. ipython:: python

    nyc = geopandas.read_file(get_path("geoda nyc"))

R-tree principle
----------------

In principle, any R-tree index builds a hierarchical collection of
bounding boxes (envelopes) representing first individual geometries and then their
most efficient combinations (from a spatial query perspective). When
creating one, you can imagine that your geometries are represented by
their envelopes, as illustrated below.

.. ipython:: python

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

    nyc.plot(ax=axs[0], edgecolor="black", linewidth=1)
    @savefig bbox.png
    nyc.envelope.boundary.plot(ax=axs[1], color='black');


The left side of the figure shows the original geometries, while the
right side their bounding boxes, extracted using the :attr:`~GeoSeries.envelope`
property. Typically, the index works on top of those.

Let’s generate two points now, both intersecting at least one bounding
box but only one intersecting the actual geometry.

.. ipython:: python

    point_inside = shapely.Point(950000, 155000)
    point_outside = shapely.Point(1050000, 150000)
    points = geopandas.GeoSeries([point_inside, point_outside], crs=nyc.crs)

You can verify that visually.

.. ipython:: python

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

    nyc.plot(ax=axs[0], edgecolor="black", linewidth=1)
    nyc.envelope.boundary.plot(ax=axs[1], color='black')
    points.plot(ax=axs[0], color="limegreen")
    @savefig points.png
    points.plot(ax=axs[1], color="limegreen");

Querying the index
------------------

Scalar query
------------

You can now use the :attr:`~GeoSeries.sindex` property to query the index. The
:meth:`~sindex.SpatialIndex.query` method, by default, returns positions of all geometries
whose bounding boxes intersect the bounding box of the input geometry.

.. ipython:: python

    bbox_query_inside = nyc.sindex.query(point_inside)
    bbox_query_outside = nyc.sindex.query(point_outside)
    bbox_query_inside, bbox_query_outside

Both the point we know is inside a geometry and the one that is outside
a geometry return one hit as each intersects one bounding box in the
tree.

.. ipython:: python

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

    nyc.plot(ax=axs[0], edgecolor="black", linewidth=1)
    nyc.envelope.boundary.plot(ax=axs[1], color='black')
    points.plot(ax=axs[0], color="limegreen", zorder=3, edgecolor="black", linewidth=.5)
    points.plot(ax=axs[1], color="limegreen", zorder=3, edgecolor="black", linewidth=.5)
    nyc.iloc[bbox_query_inside].plot(ax=axs[0], color='orange')
    nyc.iloc[bbox_query_outside].plot(ax=axs[0], color='orange')
    nyc.envelope.iloc[bbox_query_inside].plot(ax=axs[1], color='orange')
    @savefig box_query.png
    nyc.envelope.iloc[bbox_query_outside].plot(ax=axs[1], color='orange');


The image above provides a clear illustration of what happens. While you
can see on the left image that only one intersects an orange geometry
marked as a *hit*, the hits are quite clear when looking at the bounding
box.

Thankfully, the spatial index allows for further filtering based on the
actual geometry. In this case, the tree is first queried as above but
afterwards, each of the possible hits is checked using a spatial predicate.

.. ipython:: python

    pred_inside = nyc.sindex.query(point_inside, predicate="intersects")
    pred_outside = nyc.sindex.query(point_outside, predicate="intersects")
    pred_inside, pred_outside


When you specify ``predicate="intersects"``, the result is indeed
different and the output of the query using the point that lies outside
of any geometry is empty.

.. ipython:: python

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

    nyc.plot(ax=axs[0], edgecolor="black", linewidth=1)
    nyc.envelope.boundary.plot(ax=axs[1], color='black')
    points.plot(ax=axs[0], color="limegreen", zorder=3, edgecolor="black", linewidth=.5)
    points.plot(ax=axs[1], color="limegreen", zorder=3, edgecolor="black", linewidth=.5)
    nyc.iloc[pred_inside].plot(ax=axs[0], color='orange')
    @savefig predicate.png
    nyc.envelope.iloc[pred_inside].plot(ax=axs[1], color='orange');


You can use any of the predicates available in :attr:`~sindex.SpatialIndex.valid_query_predicates`:

.. ipython:: python

    nyc.sindex.valid_query_predicates

Array query
~~~~~~~~~~~

Checking a single geometry against the tree is nice but not that
efficient if you are interested in many-to-many relationships. The
:meth:`~sindex.SpatialIndex.query` method allows passing any 1-D array of geometries to be
checked against the tree. If you do so, the output structure is slightly
different:

.. ipython:: python

    bbox_array_query = nyc.sindex.query(points)
    bbox_array_query


By default, the method returns a 2-D array of indices where the query
found a hit where the subarrays correspond to the indices of the input
geometries and indices of the tree geometries associated with each. In
the example above, the 0-th geometry in the ``points`` GeoSeries
intersects the bounding box of the geometry at the position 1 from the
``nyc`` GeoDataFrame, while the geometry 1 in the ``points`` matches
geometry 16 in the ``nyc``. You may notice that these are the same
indices as you’ve seen above.

The other option is to return a boolean array with shape
``(len(tree), n)`` with boolean values marking whether the bounding box
of a geometry in the tree intersects a bounding box of a given geometry.
This can be either a dense numpy array, or a sparse scipy array. Keep in
mind that the output will be, in most cases, mostly filled with
``False`` and the array can become really large, so it is recommended to
use the sparse format, if possible.

You can specify each using the ``output_format`` keyword:

.. ipython:: python

    bbox_array_query_dense = nyc.sindex.query(points, output_format="dense")
    bbox_array_query_dense


The dense array above has rows aligned with the rows of ``nyc`` and
columns aligned with the rows of ``points`` and indicates all pairs
where a *hit* was found.

The same array can be represented as a :func:`scipy.sparse.coo_array`:

.. ipython:: python

    bbox_array_query_sparse = nyc.sindex.query(points, output_format="sparse")
    bbox_array_query_sparse

For example, to find the number of neighboring geometries for each subborough, you can
use the spatial index to compare all geometries against each other. Since you are using
``nyc`` on both sides of the query here, the resulting array is square-shaped with
diagonal filled with ``True``.

.. ipython:: python

    neighbors = nyc.sindex.query(nyc.geometry, predicate="intersects", output_format="dense")
    neighbors


Getting the sum along one axis can then give you the answer. Note that
since a geometry always intersects itself, you need to subtract one.

.. ipython:: python

    n_neighbors = neighbors.sum(axis=1) - 1
    n_neighbors

The result is a numpy array you can directly plot on a map.

.. ipython:: python

    nyc.plot(n_neighbors, legend=True);


Nearest geometry query
----------------------

While checking the spatial predicate using the spatial index is indeed extremely useful, GeoPandas
also allows you to use the spatial index to find
the nearest geometry. The API is similar as above:

.. ipython:: python

    nearest_indices = nyc.sindex.nearest(points)
    nearest_indices

You can see that the nearest query returns the indices representation.
If you are interested in how “near” the geometries actually are, the method
can also return distances. In this case, the return format is a tuple of
arrays.

.. ipython:: python

    nearest_indices, distance = nyc.sindex.nearest(points, return_distance=True)
    distance