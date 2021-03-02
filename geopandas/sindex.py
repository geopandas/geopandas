from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np

from . import _compat as compat


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


if compat.HAS_RTREE:

    import rtree.index  # noqa
    from rtree.core import RTreeError  # noqa
    from shapely.prepared import prep  # noqa

    class SpatialIndex(rtree.index.Index):
        """Original rtree wrapper, kept for backwards compatibility."""

        def __init__(self, *args):
            super().__init__(self, *args)

        @property
        def size(self):
            return len(self.leaves()[0][1])

        @property
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

        def query(self, geometry, predicate=None, sort=False):
            """Return the index of all geometries in the tree with extents that
            intersect the envelope of the input geometry.

            Compatibility layer for pygeos-based ``sindex.query``.

            This is not a vectorized function, if speed is important,
            please use PyGEOS.

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

        def query_bulk(self, geometry, predicate=None, sort=False):
            """
            Returns all combinations of each input geometry and geometries in
            the tree where the envelope of each input geometry intersects with
            the envelope of a tree geometry.

            Compatibility layer for pygeos-based ``sindex.query_bulk``.

            Iterates over ``geometry`` and queries index.
            This operation is not vectorized and may be slow.
            Use PyGEOS with ``query_bulk`` for speed.

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
            # Iterates over geometry, applying func.
            tree_index = []
            input_geometry_index = []

            for i, geo in enumerate(geometry):
                res = self.query(geo, predicate=predicate, sort=sort)
                tree_index.extend(res)
                input_geometry_index.extend([i] * len(res))
            return np.vstack([input_geometry_index, tree_index])

        def intersection(self, coordinates):
            """Wrapper for rtree.index.Index.intersection.

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
            return super().intersection(coordinates, objects=False)

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
            return self.geometries.size == 0 or self.size == 0

        def __len__(self):
            return self.size


if compat.HAS_PYGEOS:

    from . import geoseries  # noqa
    from . import array  # noqa
    import pygeos  # noqa

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
            # set empty geometries to None to mantain indexing
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
            return pygeos.strtree.VALID_PREDICATES | set([None])

        def query(self, geometry, predicate=None, sort=False):
            """
            Return the index of all geometries in the tree with extents
            that intersect the envelope of the input geometry.

            Wrapper for pygeos.STRtree.query.

            This also ensures a deterministic (sorted) order for the results.

            Parameters
            ----------
            geometry : single PyGEOS or shapely geometry
            predicate : {None, 'intersects', 'within', 'contains', \
'overlaps', 'crosses', 'touches'}, optional
                If predicate is provided, the input geometry is tested
                using the predicate function against each item in the
                tree whose extent intersects the envelope of the input
                geometry: predicate(input_geometry, tree_geometry).
            sort : bool, default False
                If True, the results will be sorted in ascending order.
                If False, results are often sorted but there is no guarantee.

            Returns
            -------
            matches : ndarray of shape (n_results, )
                Integer indices for matching geometries from the spatial index.

            Notes
            -----
            See PyGEOS.strtree documentation for more information.

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

        def query_bulk(self, geometry, predicate=None, sort=False):
            """
            Returns all combinations of each input geometry and geometries in
            the tree where the envelope of each input geometry intersects with
            the envelope of a tree geometry.

            Wrapper to expose underlaying pygeos objects to pygeos.query_bulk.

            In the context of a spatial join, input geometries are the “left”
            geometries that determine the order of the results, and tree geometries
            are “right” geometries that are joined against the left geometries.
            This effectively performs an inner join, where only those combinations
            of geometries that can be joined based on envelope overlap or optional
            predicate are returned.

            This also allows a deterministic (sorted) order for the results.


            Parameters
            ----------
            geometry : {GeoSeries, GeometryArray, numpy.array of PyGEOS geometries}
                Accepts GeoPandas geometry iterables (GeoSeries, GeometryArray)
                or a numpy array of PyGEOS geometries.
            predicate : {None, 'intersects', 'within', 'contains', \
'overlaps', 'crosses', 'touches'}, optional
                If predicate is provided, the input geometry is tested
                using the predicate function against each item in the
                index whose extent intersects the envelope of the input geometry:
                predicate(input_geometry, tree_geometry).
            sort : bool, default False
                If True, results sorted lexicographically using
                geometry's indexes as the primary key and the sindex's indexes as the
                secondary key. If False, no additional sorting is applied.

            Returns
            -------
            ndarray with shape (2, n)
                The first subarray contains input geometry integer indexes.
                The second subarray contains tree geometry integer indexes.

            Notes
            -----
            See PyGEOS.strtree documentation for more information.

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

            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`, `predicate` must be one of {}".format(
                        predicate, self.valid_query_predicates
                    )
                )
            if isinstance(geometry, geoseries.GeoSeries):
                geometry = geometry.values.data
            elif isinstance(geometry, array.GeometryArray):
                geometry = geometry.data
            elif not isinstance(geometry, np.ndarray):
                geometry = np.asarray(geometry)

            res = super().query_bulk(geometry, predicate)

            if sort:
                # sort by first array (geometry) and then second (tree)
                geo_res, tree_res = res
                indexing = np.lexsort((tree_res, geo_res))
                return np.vstack((geo_res[indexing], tree_res[indexing]))

            return res

        def nearest_all(self, geometry, max_distance=None):
            if isinstance(geometry, geoseries.GeoSeries):
                geometry = geometry.values.data
            elif isinstance(geometry, array.GeometryArray):
                geometry = geometry.data
            elif not isinstance(geometry, np.ndarray):
                geometry = np.asarray(geometry)
            return super().nearest_all(geometry, max_distance=max_distance)

        def intersection(self, coordinates):
            """Wrapper for pygeos.query that uses the RTree API.

            Compatibility wrapper, use ``query`` instead.

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

            >>> s.sindex.query(box(1, 1, 3, 3))
            array([1, 2, 3])

            >>> s.sindex.intersection(box(1, 1, 3, 3).bounds)
            array([1, 2, 3])

            """
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
            return len(self)

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
            return len(self) == 0
