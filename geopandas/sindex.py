from geopandas import base

if base.HAS_SINDEX:
    from rtree.index import Index as RTreeIndex


class SpatialIndex(RTreeIndex):
    """
    A simple wrapper around rtree's RTree Index
    """

    def __init__(self, *args):
        if not base.HAS_SINDEX:
            raise ImportError("SpatialIndex needs `rtree`")
        RTreeIndex.__init__(self, *args)

    @property
    def size(self):
        return len(self.leaves()[0][1])

    @property
    def is_empty(self):
        if len(self.leaves()) > 1:
            return False
        return self.size < 1
