from warnings import warn
from collections import namedtuple

import pandas as pd

from . import _compat as compat


def has_sindex():
    """
    Dynamically checks for ability to generate spatial index.
    """
    return get_sindex_class() is not None


def get_sindex_class():
    """
    Dynamically chooses a spatial indexing backend.
    Required to comply with _compat.USE_PYGEOS.
    The order of preference goes PyGeos > RTree > None.
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

    class SpatialIndex(rtree.index.Index):
        """
        Original rtree wrapper, kept for backwards compatibility.
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
        """
        A simple wrapper around rtree's RTree Index
        """

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

        @property
        def size(self):
            return len(self.leaves()[0][1])

        @property
        def is_empty(self):
            return self.size == 0


if compat.HAS_PYGEOS:

    from pygeos import STRtree, box, points  # noqa

    class PyGEOSSTRTreeIndex(STRtree):
        """
        A simple wrapper around pygeos's STRTree
        """

        with_objects = namedtuple("with_objects", "object id")

        def __init__(self, geometry):
            # for compatibility with old RTree implementation, store ids/indexes
            original_indexes = geometry.index
            non_empty = geometry[~geometry.values.is_empty]
            self.objects = self.ids = original_indexes[~geometry.values.is_empty]
            super().__init__(non_empty.values.data)

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
                # this is a check that rtree does, we mimick it
                # to ensure a useful failure message
                raise TypeError(
                    "Invalid coordinates, must be iterable in format "
                    "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points)."
                )

            # need to convert tuple of bounds to a geometry object
            if len(coordinates) == 4:
                indexes = super().query(box(*coordinates))
            elif len(coordinates) == 2:
                indexes = super().query(points(*coordinates))
            else:
                raise TypeError(
                    "Invalid coordinates, must be iterable in format "
                    "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points)."
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
