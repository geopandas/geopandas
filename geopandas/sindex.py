from warnings import warn
from collections import namedtuple

from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np

from . import _compat as compat


VALID_QUERY_PREDICATES = {
    None,
    "intersects",
    "within",
    "contains",
    "overlaps",
    "crosses",
    "touches",
}


def has_sindex():
    """Dynamically checks for ability to generate spatial index.
    """
    return get_sindex_class() is not None


def get_sindex_class():
    """Dynamically chooses a spatial indexing backend.

    Required to comply with _compat.USE_PYGEOS.
    The selection order goes PyGEOS > RTree > None.
    """
    if compat.USE_PYGEOS:
        return PyGEOSSTRTreeIndex
    if compat.HAS_RTREE:
        return RTreeIndex
    warn("Spatial indexes require either `rtree` or `pygeos`.")
    return None


if compat.HAS_RTREE:

    import rtree.index  # noqa
    from rtree.core import RTreeError  # noqa
    from shapely.geometry import box  # noqa
    from shapely.prepared import prep  # noqa

    class SpatialIndex(rtree.index.Index):
        """Original rtree wrapper, kept for backwards compatibility.
        """

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
        geometry : GeoSeries
            GeoSeries from which to build the spatial index.
        """

        # set of valid predicates for this spatial index
        # by default, the global set
        valid_query_predicates = VALID_QUERY_PREDICATES

        def __init__(self, geometry):
            stream = (
                (i, item.bounds, idx)
                for i, (idx, item) in enumerate(geometry.iteritems())
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
            self._geometries = geometry.geometry.values
            # create a prepared geometry cache
            self._prepared_geometries = np.array(
                [None] * self._geometries.size, dtype=object
            )

        def query(self, geometry, predicate=None, sort=False):
            """Compatibility layer for pygeos.query.

            This is not a vectorized function, if speed is important,
            please use PyGEOS.

            Parameters
            ----------
            geometry : shapely geometry
                A single shapely geometry to query against the spatial index.
            predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
                If predicate is provided, a prepared version of the input geometry is tested using
                the predicate function against each item in the index whose extent intersects the
                envelope of the input geometry: predicate(geometry, tree_geometry).
            sort : bool, default False
                If True, the results will be sorted in ascending order. If False, results are
                often sorted but there is no guarantee.

            Returns
            -------
            matches : ndarray of shape (n_results, )
                Integer indices for matching geometries from the spatial index.
            """  # noqa: E501
            # handle invalid predicates
            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`, `predicate` must be one of {}".format(
                        predicate, self.valid_query_predicates
                    )
                )
            # handle empty / invalid geometries
            if geometry is None:
                # this is the behavior in pygeos.strtree.query, we mimic it here
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

            # get bounds
            bounds = geometry.bounds  # rtree operates on bounds
            tree_query = list(self.intersection(bounds, objects=False))

            if not tree_query:
                return np.array([], dtype=np.intp)

            # check predicate
            if predicate in ("intersects", "within"):
                # only contains and intersects are supported by
                # prepared geometries, see note below regarding within
                res = []
                if predicate == "within":
                    # since these are inverse, we can flip the operation
                    # and test with prepared predicates from tree
                    predicate = "contains"
                for i in tree_query:
                    if self._prepared_geometries[i] is None:
                        # if not already prepared, prepare and cache
                        self._prepared_geometries[i] = prep(self._geometries[i])
                    if getattr(self._prepared_geometries[i], predicate)(geometry):
                        res.append(i)
                tree_query = res
            elif predicate == "contains" and len(tree_query) > 1:
                # prepare this geometry
                geometry = prep(geometry)
                tree_query = [
                    i
                    for i in tree_query
                    if getattr(geometry, predicate)(self._geometries[i])
                ]
            elif predicate is not None:
                tree_query = [
                    i
                    for i in tree_query
                    if getattr(geometry, predicate)(self._geometries[i])
                ]

            if not sort:
                # unsorted
                return np.array(tree_query, dtype=np.intp)

            # sorted
            return np.sort(np.array(tree_query, dtype=np.intp))

        def query_bulk(self, geometry, predicate=None, sort=False):
            """Compatibility layer for pygeos.query_bulk.

            Iterates over `geometry` and queries index.
            This operation is not vectorized and may be slow.
            Use PyGEOS with `query_bulk` for speed.

            Parameters
            ----------
            geometry : {GeoSeries, GeometryArray, numpy.array of PyGEOS geometries}
                Accepts GeoPandas geometry iterables (GeoSeries, GeometryArray)
                or a numpy array of PyGEOS geometries.
            predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
                If predicate is provided, a prepared version of the input geometry is tested using
                the predicate function against each item in the index whose extent intersects the
                envelope of the input geometry: predicate(geometry, tree_geometry).
            sort : bool, default False
                If True, results sorted lexicographically using
                geometry's indexes as the primary key and the sindex's indexes as the
                secondary key. If False, no additional sorting is applied.

            Returns
            -------
            ndarray with shape (2, n)
                The first subarray contains input geometry indexes.
                The second subarray contains tree geometry indexes.
            """  # noqa: E501
            # Iterates over geometry, applying func.
            tree_index = []
            geo_index = []

            for i, geo in enumerate(geometry):
                res = self.query(geo, predicate=predicate, sort=sort)
                if res.size > 0:
                    # sort results and append
                    tree_index.extend(res)
                    geo_index.extend([i] * len(res))
            return np.vstack([geo_index, tree_index])

        @property
        def size(self):
            return len(self.leaves()[0][1])

        @property
        def is_empty(self):
            return self.size == 0


if compat.HAS_PYGEOS:

    from . import geoseries  # noqa
    from .array import GeometryArray  # noqa
    from pygeos import STRtree, box, points, Geometry, from_shapely, from_wkb  # noqa

    class PyGEOSSTRTreeIndex(STRtree):
        """A simple wrapper around pygeos's STRTree.


        Parameters
        ----------
        geometry : GeoSeries
            GeoSeries from which to build the spatial index.
        """

        # helper for loc/label based indexing in `intersection` method
        with_objects = namedtuple("with_objects", "object id")

        # set of valid predicates for this spatial index
        # by default, the global set
        valid_query_predicates = VALID_QUERY_PREDICATES

        def __init__(self, geometry):
            # for compatibility with old RTree implementation, store ids/indexes
            original_indexes = geometry.index
            non_empty = geometry[~geometry.values.is_empty]
            self.objects = self.ids = original_indexes[~geometry.values.is_empty]
            super().__init__(non_empty.values.data)

        def query_bulk(self, geometry, predicate=None, sort=False):
            """Wrapper to expose underlaying pygeos objects to pygeos.query_bulk.

            This also allows a deterministic (sorted) order for the results.


            Parameters
            ----------
            geometry : {GeoSeries, GeometryArray, numpy.array of PyGEOS geometries}
                Accepts GeoPandas geometry iterables (GeoSeries, GeometryArray)
                or a numpy array of PyGEOS geometries.
            predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
                If predicate is provided, a prepared version of the input geometry is tested using
                the predicate function against each item in the index whose extent intersects the
                envelope of the input geometry: predicate(geometry, tree_geometry).
            sort : bool, default False
                If True, results sorted lexicographically using
                geometry's indexes as the primary key and the sindex's indexes as the
                secondary key. If False, no additional sorting is applied.

            Returns
            -------
            ndarray with shape (2, n)
                The first subarray contains input geometry indexes.
                The second subarray contains tree geometry indexes.
            See also
            --------
            See PyGEOS.strtree documentation for more information.
            """  # noqa: E501

            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`, `predicate` must be one of {}".format(
                        predicate, self.valid_query_predicates
                    )
                )
            if isinstance(geometry, geoseries.GeoSeries):
                geometry = geometry.values.data
            elif isinstance(geometry, GeometryArray):
                geometry = geometry.data
            elif not isinstance(geometry, np.ndarray):
                geometry = np.asarray(geometry)

            res = super().query_bulk(geometry, predicate)

            if not sort:
                return res
            # sort by first array (geometry) and then second (tree)
            geo_res, tree_res = res
            indexing = np.lexsort((tree_res, geo_res))
            return np.vstack((geo_res[indexing], tree_res[indexing]))

        def query(self, geometry, predicate=None, sort=False):
            """Wrapper for pygeos.query.

            This also ensures a deterministic (sorted) order for the results.

            Parameters
            ----------
            geometry : single PyGEOS geometry
            predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
                If predicate is provided, a prepared version of the input geometry is tested using
                the predicate function against each item in the index whose extent intersects the
                envelope of the input geometry: predicate(geometry, tree_geometry).
            sort : bool, default False
                If True, the results will be sorted in ascending order. If False, results are
                often sorted but there is no guarantee.

            Returns
            -------
            matches : ndarray of shape (n_results, )
                Integer indices for matching geometries from the spatial index.
            See also
            --------
            See PyGEOS.strtree documentation for more information.
            """  # noqa: E501

            if predicate not in self.valid_query_predicates:
                raise ValueError(
                    "Got `predicate` = `{}`; ".format(predicate)
                    + "`predicate` must be one of {}".format(
                        self.valid_query_predicates
                    )
                )

            if isinstance(geometry, BaseGeometry):
                # handle shapely geometries
                if compat.PYGEOS_SHAPELY_COMPAT:
                    geometry = from_shapely(geometry)
                # fallback going through WKB
                elif geometry.is_empty and geometry.geom_type == "Point":
                    # empty point does not roundtrip through WKB
                    geometry = from_wkb("POINT EMPTY")
                else:
                    geometry = from_wkb(geometry.wkb)

            matches = super().query(geometry=geometry, predicate=predicate)

            if sort:
                return np.sort(matches)

            return matches

        def intersection(self, coordinates, objects=False):
            """Wrapper for pygeos.query that uses the RTree API.

            Parameters
            ----------
            coordinates : sequence or array
                Sequence of the form (min_x, min_y, max_x, max_y)
                to query a rectangle or (x, y) to query a point.
            objects : True or False
                If True, return the label based indexes. If False, integer indexes
                are returned.
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
                indexes = super().query(box(*coordinates))
            elif len(coordinates) == 2:
                indexes = super().query(points(*coordinates))
            else:
                raise TypeError(
                    "Invalid coordinates, must be iterable in format "
                    "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). "
                    "Got `coordinates` = {}.".format(coordinates)
                )

            if objects:
                objs = self.objects[indexes].values
                ids = self.ids[indexes]
                return [
                    self.with_objects(id=id, object=obj) for id, obj in zip(ids, objs)
                ]
            else:
                return indexes

        @property
        def size(self):
            return len(self)

        @property
        def is_empty(self):
            return len(self) == 0
