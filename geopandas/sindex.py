import numpy as np

import shapely
from shapely.geometry.base import BaseGeometry

from . import _compat as compat
from . import array, geoseries

PREDICATES = {p.name for p in shapely.strtree.BinaryPredicate} | {None}

if compat.GEOS_GE_310:
    PREDICATES.update(["dwithin"])


class SpatialIndex:
    """A simple wrapper around Shapely's STRTree.


    Parameters
    ----------
    geometry : np.array of Shapely geometries
        Geometries from which to build the spatial index.
    """

    def __init__(self, geometry):
        # set empty geometries to None to avoid segfault on GEOS <= 3.6
        # see:
        # https://github.com/pygeos/pygeos/issues/146
        # https://github.com/pygeos/pygeos/issues/147
        non_empty = geometry.copy()
        non_empty[shapely.is_empty(non_empty)] = None
        # set empty geometries to None to maintain indexing
        self._tree = shapely.STRtree(non_empty)
        # store geometries, including empty geometries for user access
        self.geometries = geometry.copy()

    @property
    def valid_query_predicates(self):
        """Returns valid predicates for the spatial index.

        Returns
        -------
        set
            Set of valid predicates for this spatial index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
        >>> s.sindex.valid_query_predicates  # doctest: +SKIP
        {None, "contains", "contains_properly", "covered_by", "covers", \
"crosses", "dwithin", "intersects", "overlaps", "touches", "within"}
        """
        return PREDICATES

    def query(
        self, geometry, predicate=None, sort=False, distance=None, output_format="tuple"
    ):
        """
        Return the integer indices of all combinations of each input geometry
        and tree geometries where the bounding box of each input geometry
        intersects the bounding box of a tree geometry.

        If the input geometry is a scalar, this returns an array of shape (n, ) with
        the indices of the matching tree geometries.  If the input geometry is an
        array_like, this returns an array with shape (2,n) where the subarrays
        correspond to the indices of the input geometries and indices of the
        tree geometries associated with each.  To generate an array of pairs of
        input geometry index and tree geometry index, simply transpose the
        result.

        If a predicate is provided, the tree geometries are first queried based
        on the bounding box of the input geometry and then are further filtered
        to those that meet the predicate when comparing the input geometry to
        the tree geometry: ``predicate(geometry, tree_geometry)``.

        The 'dwithin' predicate requires GEOS >= 3.10.

        Bounding boxes are limited to two dimensions and are axis-aligned
        (equivalent to the ``bounds`` property of a geometry); any Z values
        present in input geometries are ignored when querying the tree.

        Any input geometry that is None or empty will never match geometries in
        the tree.

        Parameters
        ----------
        geometry : shapely.Geometry or array-like of geometries \
(numpy.ndarray, GeoSeries, GeometryArray)
            A single shapely geometry or array of geometries to query against
            the spatial index. For array-like, accepts both GeoPandas geometry
            iterables (GeoSeries, GeometryArray) or a numpy array of Shapely
            geometries.
        predicate : {None, "contains", "contains_properly", "covered_by", "covers", \
"crosses", "intersects", "overlaps", "touches", "within", "dwithin"}, optional
            If predicate is provided, the input geometries are tested
            using the predicate function against each item in the tree
            whose extent intersects the envelope of the input geometry:
            ``predicate(input_geometry, tree_geometry)``.
            If possible, prepared geometries are used to help speed up the
            predicate operation.
        sort : bool, default False
            If True, the results will be sorted in ascending order. In case
            of 2D array, the result is sorted lexicographically using the
            geometries' indexes as the primary key and the sindex's indexes
            as the secondary key.
            If False, no additional sorting is applied (results are often
            sorted but there is no guarantee).
        distance : number or array_like, optional
            Distances around each input geometry within which to query the tree for
            the 'dwithin' predicate. If array_like, shape must be broadcastable to shape
            of geometry. Required if ``predicate='dwithin'``.

        Returns
        -------
        ndarray with shape (n,) if geometry is a scalar
            Integer indices for matching geometries from the spatial index
            tree geometries.

        OR

        ndarray with shape (2, n) if geometry is an array_like
            The first subarray contains input geometry integer indices.
            The second subarray contains tree geometry integer indices.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        Querying the tree with a scalar geometry:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="contains")
        array([2])

        Querying the tree with an array of geometries:

        >>> s2 = geopandas.GeoSeries([box(2, 2, 4, 4), box(5, 5, 6, 6)])
        >>> s2
        0    POLYGON ((4 2, 4 4, 2 4, 2 2, 4 2))
        1    POLYGON ((6 5, 6 6, 5 6, 5 5, 6 5))
        dtype: geometry

        >>> s.sindex.query(s2)
        array([[0, 0, 0, 1, 1],
               [2, 3, 4, 5, 6]])

        >>> s.sindex.query(s2, predicate="contains")
        array([[0],
               [3]])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="dwithin", distance=0)
        array([1, 2, 3])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="dwithin", distance=2)
        array([0, 1, 2, 3, 4])

        Notes
        -----
        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries. This
        effectively performs an inner join, where only those combinations of
        geometries that can be joined based on overlapping bounding boxes or
        optional predicate are returned.
        """
        if predicate not in self.valid_query_predicates:
            if predicate == "dwithin":
                raise ValueError("predicate = 'dwithin' requires GEOS >= 3.10.0")

            raise ValueError(
                "Got predicate='{}'; ".format(predicate)
                + "`predicate` must be one of {}".format(self.valid_query_predicates)
            )

        # distance argument requirement of predicate `dwithin`
        # and only valid for predicate `dwithin`
        kwargs = {}
        if predicate == "dwithin":
            if distance is None:
                # the distance parameter is needed
                raise ValueError(
                    "'distance' parameter is required for 'dwithin' predicate"
                )
            # add distance to kwargs
            kwargs["distance"] = distance

        elif distance is not None:
            # distance parameter is invalid
            raise ValueError(
                "'distance' parameter is only supported in combination with "
                "'dwithin' predicate"
            )

        geometry = self._as_geometry_array(geometry)

        indices = self._tree.query(geometry, predicate=predicate, **kwargs)

        if output_format != "tuple":
            sort = True

        if sort:
            if indices.ndim == 1:
                indices = np.sort(indices)
            else:
                # sort by first array (geometry) and then second (tree)
                geo_idx, tree_idx = indices
                sort_indexer = np.lexsort((tree_idx, geo_idx))
                indices = np.vstack((geo_idx[sort_indexer], tree_idx[sort_indexer]))

        if output_format == "sparse":
            from scipy.sparse import coo_array

            return coo_array(
                (np.ones(len(indices[0]), dtype=np.bool_), indices),
                shape=(len(self.geometries), len(geometry)),
                dtype=np.bool_,
            )

        if output_format == "dense":
            dense = np.zeros((len(self.geometries), len(geometry)), dtype=bool)
            dense[indices] = True
            return dense

        if output_format == "tuple":
            return indices

        raise ValueError("Invalid output_format: {}".format(output_format))

    @staticmethod
    def _as_geometry_array(geometry):
        """Convert geometry into a numpy array of Shapely geometries.

        Parameters
        ----------
        geometry
            An array-like of Shapely geometries, a GeoPandas GeoSeries/GeometryArray,
            shapely.geometry or list of shapely geometries.

        Returns
        -------
        np.ndarray
            A numpy array of Shapely geometries.
        """
        if isinstance(geometry, np.ndarray):
            return array.from_shapely(geometry)._data
        elif isinstance(geometry, geoseries.GeoSeries):
            return geometry.values._data
        elif isinstance(geometry, array.GeometryArray):
            return geometry._data
        elif isinstance(geometry, BaseGeometry):
            return geometry
        elif geometry is None:
            return None
        else:
            return np.asarray(geometry)

    def nearest(
        self,
        geometry,
        return_all=True,
        max_distance=None,
        return_distance=False,
        exclusive=False,
    ):
        """
        Return the nearest geometry in the tree for each input geometry in
        ``geometry``.

        If multiple tree geometries have the same distance from an input geometry,
        multiple results will be returned for that input geometry by default.
        Specify ``return_all=False`` to only get a single nearest geometry
        (non-deterministic which nearest is returned).

        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries.
        If ``max_distance`` is not set, this will effectively be a left join
        because every geometry in ``geometry`` will have a nearest geometry in
        the tree. However, if ``max_distance`` is used, this becomes an
        inner join, since some geometries in ``geometry`` may not have a match
        in the tree.

        For performance reasons, it is highly recommended that you set
        the ``max_distance`` parameter.

        Parameters
        ----------
        geometry : {shapely.geometry, GeoSeries, GeometryArray, numpy.array of Shapely \
geometries}
            A single shapely geometry, one of the GeoPandas geometry iterables
            (GeoSeries, GeometryArray), or a numpy array of Shapely geometries to query
            against the spatial index.
        return_all : bool, default True
            If there are multiple equidistant or intersecting nearest
            geometries, return all those geometries instead of a single
            nearest geometry.
        max_distance : float, optional
            Maximum distance within which to query for nearest items in tree.
            Must be greater than 0. By default None, indicating no distance limit.
        return_distance : bool, optional
            If True, will return distances in addition to indexes. By default False
        exclusive : bool, optional
            if True, the nearest geometries that are equal to the input geometry
            will not be returned. By default False.  Requires Shapely >= 2.0.

        Returns
        -------
        Indices or tuple of (indices, distances)
            Indices is an ndarray of shape (2,n) and distances (if present) an
            ndarray of shape (n).
            The first subarray of indices contains input geometry indices.
            The second subarray of indices contains tree geometry indices.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s.head()
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        dtype: geometry

        >>> s.sindex.nearest(Point(1, 1))
        array([[0],
               [1]])

        >>> s.sindex.nearest([box(4.9, 4.9, 5.1, 5.1)])
        array([[0],
               [5]])

        >>> s2 = geopandas.GeoSeries(geopandas.points_from_xy([7.6, 10], [7.6, 10]))
        >>> s2
        0    POINT (7.6 7.6)
        1    POINT (10 10)
        dtype: geometry

        >>> s.sindex.nearest(s2)
        array([[0, 1],
               [8, 9]])
        """
        geometry = self._as_geometry_array(geometry)
        if isinstance(geometry, BaseGeometry) or geometry is None:
            geometry = [geometry]

        result = self._tree.query_nearest(
            geometry,
            max_distance=max_distance,
            return_distance=return_distance,
            all_matches=return_all,
            exclusive=exclusive,
        )
        if return_distance:
            indices, distances = result
        else:
            indices = result

        if return_distance:
            return indices, distances
        else:
            return indices

    def intersection(self, coordinates):
        """Compatibility wrapper for rtree.index.Index.intersection,
        use ``query`` instead.

        Parameters
        ----------
        coordinates : sequence or array
            Sequence of the form (min_x, min_y, max_x, max_y)
            to query a rectangle or (x, y) to query a point.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        >>> s.sindex.intersection(box(1, 1, 3, 3).bounds)
        array([1, 2, 3])

        Alternatively, you can use ``query``:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        """
        # TODO: we should deprecate this
        # convert bounds to geometry
        # the old API uses tuples of bound, but Shapely uses geometries
        try:
            iter(coordinates)
        except TypeError:
            # likely not an iterable
            # this is a check that rtree does, we mimic it
            # to ensure a useful failure message
            raise TypeError(
                "Invalid coordinates, must be iterable in format "
                "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). "
                "Got `coordinates` = {}.".format(coordinates)
            )

        # need to convert tuple of bounds to a geometry object
        if len(coordinates) == 4:
            indexes = self._tree.query(shapely.box(*coordinates))
        elif len(coordinates) == 2:
            indexes = self._tree.query(shapely.points(*coordinates))
        else:
            raise TypeError(
                "Invalid coordinates, must be iterable in format "
                "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). "
                "Got `coordinates` = {}.".format(coordinates)
            )

        return indexes

    @property
    def size(self):
        """Size of the spatial index

        Number of leaves (input geometries) in the index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        >>> s.sindex.size
        10
        """
        return len(self._tree)

    @property
    def is_empty(self):
        """Check if the spatial index is empty

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        >>> s.sindex.is_empty
        False

        >>> s2 = geopandas.GeoSeries()
        >>> s2.sindex.is_empty
        True
        """
        return len(self._tree) == 0

    def __len__(self):
        return len(self._tree)
