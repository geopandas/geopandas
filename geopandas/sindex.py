import warnings

from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np

from . import _compat as compat
from ._decorator import doc


def _get_sindex_class():
    """Dynamically chooses a spatial indexing backend.

    Required to comply with _compat.USE_PYGEOS.
    The selection order goes PyGEOS > RTree > Error.
    """
    if compat.USE_PYGEOS:
        return PyGEOSSTRTreeIndex
    if compat.HAS_RTREE:
        return RTreeIndex
    raise ImportError(
        "Spatial indexes require either `rtree` or `pygeos`. "
        "See installation instructions at https://geopandas.org/install.html"
    )


class BaseSpatialIndex:
    @property
    def valid_query_predicates(self):
        """Returns valid predicates for this spatial index.

        Returns
        -------
        set
            Set of valid predicates for this spatial index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
        >>> s.sindex.valid_query_predicates  # doctest: +SKIP
        {'contains', 'crosses', 'intersects', 'within', 'touches', \
'overlaps', None, 'covers', 'contains_properly'}
        """
        raise NotImplementedError

    def query(self, geometry, predicate=None, sort=False):
        """Return the index of all geometries in the tree with extents that
        intersect the envelope of the input geometry.

        When using the ``rtree`` package, this is not a vectorized function.
        If speed is important, please use PyGEOS.

        Parameters
        ----------
        geometry : shapely geometry
            A single shapely geometry to query against the spatial index.
        predicate : {None, 'intersects', 'within', 'contains', \
'overlaps', 'crosses', 'touches'}, optional
            If predicate is provided, the input geometry is
            tested using the predicate function against each item
            in the tree whose extent intersects the envelope of the
            input geometry: predicate(input_geometry, tree_geometry).
            If possible, prepared geometries are used to help
            speed up the predicate operation.
        sort : bool, default False
            If True, the results will be sorted in ascending order.
            If False, results are often sorted but there is no guarantee.

        Returns
        -------
        matches : ndarray of shape (n_results, )
            Integer indices for matching geometries from the spatial index.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="contains")
        array([2])
        """
        raise NotImplementedError

    def query_bulk(self, geometry, predicate=None, sort=False):
        """
        Returns all combinations of each input geometry and geometries in
        the tree where the envelope of each input geometry intersects with
        the envelope of a tree geometry.

        In the context of a spatial join, input geometries are the “left”
        geometries that determine the order of the results, and tree geometries
        are “right” geometries that are joined against the left geometries.
        This effectively performs an inner join, where only those combinations
        of geometries that can be joined based on envelope overlap or optional
        predicate are returned.

        When using the ``rtree`` package, this is not a vectorized function
        and may be slow. If speed is important, please use PyGEOS.

        Parameters
        ----------
        geometry : {GeoSeries, GeometryArray, numpy.array of PyGEOS geometries}
            Accepts GeoPandas geometry iterables (GeoSeries, GeometryArray)
            or a numpy array of PyGEOS geometries.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', \
'crosses', 'touches'}, optional
            If predicate is provided, the input geometries are tested using
            the predicate function against each item in the tree whose extent
            intersects the envelope of the each input geometry:
            predicate(input_geometry, tree_geometry).  If possible, prepared
            geometries are used to help speed up the predicate operation.
        sort : bool, default False
            If True, results sorted lexicographically using
            geometry's indexes as the primary key and the sindex's indexes as the
            secondary key. If False, no additional sorting is applied.

        Returns
        -------
        ndarray with shape (2, n)
            The first subarray contains input geometry integer indexes.
            The second subarray contains tree geometry integer indexes.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry
        >>> s2 = geopandas.GeoSeries([box(2, 2, 4, 4), box(5, 5, 6, 6)])
        >>> s2
        0    POLYGON ((4.00000 2.00000, 4.00000 4.00000, 2....
        1    POLYGON ((6.00000 5.00000, 6.00000 6.00000, 5....
        dtype: geometry

        >>> s.sindex.query_bulk(s2)
        array([[0, 0, 0, 1, 1],
                [2, 3, 4, 5, 6]])

        >>> s.sindex.query_bulk(s2, predicate="contains")
        array([[0],
                [3]])
        """
        raise NotImplementedError

    def nearest(
        self, geometry, return_all=True, max_distance=None, return_distance=False
    ):
        """
        Return the nearest geometry in the tree for each input geometry in
        ``geometry``.

        .. note::
            ``nearest`` currently only works with PyGEOS >= 0.10.

            Note that if PyGEOS is not available, geopandas will use rtree
            for the spatial index, where nearest has a different
            function signature to temporarily preserve existing
            functionality. See the documentation of
            :meth:`rtree.index.Index.nearest` for the details on the
            ``rtree``-based implementation.

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
        geometry : {shapely.geometry, GeoSeries, GeometryArray, numpy.array of PyGEOS \
geometries}
            A single shapely geometry, one of the GeoPandas geometry iterables
            (GeoSeries, GeometryArray), or a numpy array of PyGEOS geometries to query
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
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        dtype: geometry

        >>> s.sindex.nearest(Point(1, 1))
        array([[0],
               [1]])

        >>> s.sindex.nearest([box(4.9, 4.9, 5.1, 5.1)])
        array([[0],
               [5]])

        >>> s2 = geopandas.GeoSeries(geopandas.points_from_xy([7.6, 10], [7.6, 10]))
        >>> s2
        0    POINT (7.60000 7.60000)
        1    POINT (10.00000 10.00000)
        dtype: geometry

        >>> s.sindex.nearest(s2)
        array([[0, 1],
               [8, 9]])
        """
        raise NotImplementedError

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
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.intersection(box(1, 1, 3, 3).bounds)
        array([1, 2, 3])

        Alternatively, you can use ``query``:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        """
        raise NotImplementedError

    @property
    def size(self):
        """Size of the spatial index

        Number of leaves (input geometries) in the index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.size
        10
        """
        raise NotImplementedError

    @property
    def is_empty(self):
        """Check if the spatial index is empty

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.is_empty
        False

        >>> s2 = geopandas.GeoSeries()
        >>> s2.sindex.is_empty
        True
        """
        raise NotImplementedError


if compat.HAS_RTREE:

    import rtree.index  # noqa
    from rtree.core import RTreeError  # noqa
    from shapely.prepared import prep  # noqa

    class SpatialIndex(rtree.index.Index, BaseSpatialIndex):
        """Original rtree wrapper, kept for backwards compatibility."""

        def __init__(self, *args):
            warnings.warn(
                "Directly using SpatialIndex is deprecated, and the class will be "
                "removed in a future version. Access the spatial index through the "
                "`GeoSeries.sindex` attribute, or use `rtree.index.Index` directly.",
                FutureWarning,
                stacklevel=2,
            )
            super().__init__(*args)

        @doc(BaseSpatialIndex.intersection)
        def intersection(self, coordinates, *args, **kwargs):
            return super().intersection(coordinates, *args, **kwargs)

        @doc(BaseSpatialIndex.nearest)
        def nearest(self, *args, **kwargs):
            return super().nearest(*args, **kwargs)

        @property
        @doc(BaseSpatialIndex.size)
        def size(self):
            return len(self.leaves()[0][1])

        @property
        @doc(BaseSpatialIndex.is_empty)
        def is_empty(self):
            if len(self.leaves()) > 1:
                return False
            return self.size < 1

    class RTreeIndex(rtree.index.Index):
        """A simple wrapper around rtree's RTree Index

        Parameters
        ----------
        geometry : np.array of Shapely geometries
            Geometries from which to build the spatial index.
        """

        def __init__(self, geometry):
            stream = (
                (i, item.bounds, None)
                for i, item in enumerate(geometry)
                if pd.notnull(item) and not item.is_empty
            )
            try:
                super().__init__(stream)
            except RTreeError:
                # What we really want here is an empty generator error, or
                # for the bulk loader to log that the generator was empty
                # and move on.
                # See https://github.com/Toblerity/rtree/issues/20.
                super().__init__()

            # store reference to geometries for predicate queries
            self.geometries = geometry
            # create a prepared geometry cache
            self._prepared_geometries = np.array(
                [None] * self.geometries.size, dtype=object
            )

        @property
        @doc(BaseSpatialIndex.valid_query_predicates)
        def valid_query_predicates(self):
            return {
                None,
                "intersects",
                "within",
                "contains",
                "overlaps",
                "crosses",
                "touches",
                "covers",
                "contains_properly",
            }

        @doc(BaseSpatialIndex.query)
        def query(self, geometry, predicate=None, sort=False):
            # handle invalid predicates
            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`, `predicate` must be one of {}".format(
                        predicate, self.valid_query_predicates
                    )
                )

            # handle empty / invalid geometries
            if geometry is None:
                # return an empty integer array, similar to pygeys.STRtree.query.
                return np.array([], dtype=np.intp)

            if not isinstance(geometry, BaseGeometry):
                raise TypeError(
                    "Got `geometry` of type `{}`, `geometry` must be ".format(
                        type(geometry)
                    )
                    + "a shapely geometry."
                )

            if geometry.is_empty:
                return np.array([], dtype=np.intp)

            # query tree
            bounds = geometry.bounds  # rtree operates on bounds
            tree_idx = list(self.intersection(bounds))

            if not tree_idx:
                return np.array([], dtype=np.intp)

            # Check predicate
            # This is checked as input_geometry.predicate(tree_geometry)
            # When possible, we use prepared geometries.
            # Prepared geometries only support "intersects" and "contains"
            # For the special case of "within", we are able to flip the
            # comparison and check if tree_geometry.contains(input_geometry)
            # to still take advantage of prepared geometries.
            if predicate == "within":
                # To use prepared geometries for within,
                # we compare tree_geom.contains(input_geom)
                # Since we are preparing the tree geometries,
                # we cache them for multiple comparisons.
                res = []
                for index_in_tree in tree_idx:
                    if self._prepared_geometries[index_in_tree] is None:
                        # if not already prepared, prepare and cache
                        self._prepared_geometries[index_in_tree] = prep(
                            self.geometries[index_in_tree]
                        )
                    if self._prepared_geometries[index_in_tree].contains(geometry):
                        res.append(index_in_tree)
                tree_idx = res
            elif predicate is not None:
                # For the remaining predicates,
                # we compare input_geom.predicate(tree_geom)
                if predicate in (
                    "contains",
                    "intersects",
                    "covers",
                    "contains_properly",
                ):
                    # prepare this input geometry
                    geometry = prep(geometry)
                tree_idx = [
                    index_in_tree
                    for index_in_tree in tree_idx
                    if getattr(geometry, predicate)(self.geometries[index_in_tree])
                ]

            # sort if requested
            if sort:
                # sorted
                return np.sort(np.array(tree_idx, dtype=np.intp))

            # unsorted
            return np.array(tree_idx, dtype=np.intp)

        @doc(BaseSpatialIndex.query_bulk)
        def query_bulk(self, geometry, predicate=None, sort=False):
            # Iterates over geometry, applying func.
            tree_index = []
            input_geometry_index = []

            for i, geo in enumerate(geometry):
                res = self.query(geo, predicate=predicate, sort=sort)
                tree_index.extend(res)
                input_geometry_index.extend([i] * len(res))
            return np.vstack([input_geometry_index, tree_index])

        def nearest(self, coordinates, num_results=1, objects=False):
            """
            Returns the nearest object or objects to the given coordinates.

            Requires rtree, and passes parameters directly to
            :meth:`rtree.index.Index.nearest`.

            This behaviour is deprecated and will be updated to be consistent
            with the pygeos PyGEOSSTRTreeIndex in a future release.

            If longer-term compatibility is required, use
            :meth:`rtree.index.Index.nearest` directly instead.

            Examples
            --------
            >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(3), range(3)))
            >>> s
            0    POINT (0.00000 0.00000)
            1    POINT (1.00000 1.00000)
            2    POINT (2.00000 2.00000)
            dtype: geometry

            >>> list(s.sindex.nearest((0, 0)))  # doctest: +SKIP
            [0]

            >>> list(s.sindex.nearest((0.5, 0.5)))  # doctest: +SKIP
            [0, 1]

            >>> list(s.sindex.nearest((3, 3), num_results=2))  # doctest: +SKIP
            [2, 1]

            >>> list(super(type(s.sindex), s.sindex).nearest((0, 0),
            ... num_results=2))  # doctest: +SKIP
            [0, 1]

            Parameters
            ----------
            coordinates : sequence or array
                This may be an object that satisfies the numpy array protocol,
                providing the index’s dimension * 2 coordinate pairs
                representing the mink and maxk coordinates in each dimension
                defining the bounds of the query window.
            num_results : integer
                The number of results to return nearest to the given
                coordinates. If two index entries are equidistant, both are
                returned. This property means that num_results may return more
                items than specified
            objects : True / False / ‘raw’
                If True, the nearest method will return index objects that were
                pickled when they were stored with each index entry, as well as
                the id and bounds of the index entries. If ‘raw’, it will
                return the object as entered into the database without the
                rtree.index.Item wrapper.
            """
            warnings.warn(
                "sindex.nearest using the rtree backend was not previously documented "
                "and this behavior is deprecated in favor of matching the function "
                "signature provided by the pygeos backend (see "
                "PyGEOSSTRTreeIndex.nearest for details). This behavior will be "
                "updated in a future release.",
                FutureWarning,
            )
            return super().nearest(
                coordinates, num_results=num_results, objects=objects
            )

        @doc(BaseSpatialIndex.intersection)
        def intersection(self, coordinates):
            return super().intersection(coordinates, objects=False)

        @property
        @doc(BaseSpatialIndex.size)
        def size(self):
            if hasattr(self, "_size"):
                size = self._size
            else:
                # self.leaves are lists of tuples of (int, lists...)
                # index [0][1] always has an element, even for empty sindex
                # for an empty index, it will be an empty list
                size = len(self.leaves()[0][1])
                self._size = size
            return size

        @property
        @doc(BaseSpatialIndex.is_empty)
        def is_empty(self):
            return self.geometries.size == 0 or self.size == 0

        def __len__(self):
            return self.size


if compat.HAS_PYGEOS:

    from . import geoseries  # noqa
    from . import array  # noqa
    import pygeos  # noqa

    _PYGEOS_PREDICATES = {p.name for p in pygeos.strtree.BinaryPredicate} | set([None])

    class PyGEOSSTRTreeIndex(pygeos.STRtree):
        """A simple wrapper around pygeos's STRTree.


        Parameters
        ----------
        geometry : np.array of PyGEOS geometries
            Geometries from which to build the spatial index.
        """

        def __init__(self, geometry):
            # set empty geometries to None to avoid segfault on GEOS <= 3.6
            # see:
            # https://github.com/pygeos/pygeos/issues/146
            # https://github.com/pygeos/pygeos/issues/147
            non_empty = geometry.copy()
            non_empty[pygeos.is_empty(non_empty)] = None
            # set empty geometries to None to maintain indexing
            super().__init__(non_empty)
            # store geometries, including empty geometries for user access
            self.geometries = geometry.copy()

        @property
        def valid_query_predicates(self):
            """Returns valid predicates for the used spatial index.

            Returns
            -------
            set
                Set of valid predicates for this spatial index.

            Examples
            --------
            >>> from shapely.geometry import Point
            >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
            >>> s.sindex.valid_query_predicates  # doctest: +SKIP
            {'contains', 'crosses', 'covered_by', None, 'intersects', 'within', \
'touches', 'overlaps', 'contains_properly', 'covers'}
            """
            return _PYGEOS_PREDICATES

        @doc(BaseSpatialIndex.query)
        def query(self, geometry, predicate=None, sort=False):
            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`; ".format(predicate)
                    + "`predicate` must be one of {}".format(
                        self.valid_query_predicates
                    )
                )

            if isinstance(geometry, BaseGeometry):
                geometry = array._shapely_to_geom(geometry)

            matches = super().query(geometry=geometry, predicate=predicate)

            if sort:
                return np.sort(matches)

            return matches

        @staticmethod
        def _as_geometry_array(geometry):
            """Convert geometry into a numpy array of PyGEOS geometries.

            Parameters
            ----------
            geometry
                An array-like of PyGEOS geometries, a GeoPandas GeoSeries/GeometryArray,
                shapely.geometry or list of shapely geometries.

            Returns
            -------
            np.ndarray
                A numpy array of pygeos geometries.
            """
            if isinstance(geometry, np.ndarray):
                return geometry
            elif isinstance(geometry, geoseries.GeoSeries):
                return geometry.values.data
            elif isinstance(geometry, array.GeometryArray):
                return geometry.data
            elif isinstance(geometry, BaseGeometry):
                return array._shapely_to_geom(geometry)
            elif isinstance(geometry, list):
                return np.asarray(
                    [
                        array._shapely_to_geom(el)
                        if isinstance(el, BaseGeometry)
                        else el
                        for el in geometry
                    ]
                )
            else:
                return np.asarray(geometry)

        @doc(BaseSpatialIndex.query_bulk)
        def query_bulk(self, geometry, predicate=None, sort=False):
            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`, `predicate` must be one of {}".format(
                        predicate, self.valid_query_predicates
                    )
                )

            geometry = self._as_geometry_array(geometry)

            res = super().query_bulk(geometry, predicate)

            if sort:
                # sort by first array (geometry) and then second (tree)
                geo_res, tree_res = res
                indexing = np.lexsort((tree_res, geo_res))
                return np.vstack((geo_res[indexing], tree_res[indexing]))

            return res

        @doc(BaseSpatialIndex.nearest)
        def nearest(
            self, geometry, return_all=True, max_distance=None, return_distance=False
        ):
            if not compat.PYGEOS_GE_010:
                raise NotImplementedError("sindex.nearest requires pygeos >= 0.10")

            geometry = self._as_geometry_array(geometry)

            if not return_all and max_distance is None and not return_distance:
                return super().nearest(geometry)

            result = super().nearest_all(
                geometry, max_distance=max_distance, return_distance=return_distance
            )
            if return_distance:
                indices, distances = result
            else:
                indices = result

            if not return_all:
                # first subarray of geometry indices is sorted, so we can use this
                # trick to get the first of each index value
                mask = np.diff(indices[0, :]).astype("bool")
                # always select the first element
                mask = np.insert(mask, 0, True)

                indices = indices[:, mask]
                if return_distance:
                    distances = distances[mask]

            if return_distance:
                return indices, distances
            else:
                return indices

        @doc(BaseSpatialIndex.intersection)
        def intersection(self, coordinates):
            # convert bounds to geometry
            # the old API uses tuples of bound, but pygeos uses geometries
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
                indexes = super().query(pygeos.box(*coordinates))
            elif len(coordinates) == 2:
                indexes = super().query(pygeos.points(*coordinates))
            else:
                raise TypeError(
                    "Invalid coordinates, must be iterable in format "
                    "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). "
                    "Got `coordinates` = {}.".format(coordinates)
                )

            return indexes

        @property
        @doc(BaseSpatialIndex.size)
        def size(self):
            return len(self)

        @property
        @doc(BaseSpatialIndex.is_empty)
        def is_empty(self):
            return len(self) == 0
