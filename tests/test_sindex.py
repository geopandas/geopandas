import shutil
import tempfile
import numpy as np
from numpy.testing import assert_array_equal
from pandas import Series
from shapely.geometry import (Polygon, Point, LineString,
                              MultiPoint, MultiLineString, MultiPolygon)
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries, GeoDataFrame, base
from .util import unittest, geom_equals, geom_almost_equals


@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestSeriesSindex(unittest.TestCase):

    def test_empty_index(self):
        self.assert_(GeoSeries()._sindex is None)

    def test_point(self):
        s = GeoSeries([Point(0, 0)])
        self.assertEqual(s._sindex.size, 1)
        hits = s._sindex.intersection((-1, -1, 1, 1))
        self.assertEqual(len(list(hits)), 1)
        hits = s._sindex.intersection((-2, -2, -1, -1))
        self.assertEqual(len(list(hits)), 0)

    def test_empty_point(self):
        s = GeoSeries([Point()])
        self.assert_(GeoSeries()._sindex is None)

    def test_polygons(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        self.assertEqual(s._sindex.size, 3)

    def test_polygons_append(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        s = GeoSeries([t1, t2, sq])
        t = GeoSeries([t1, t2, sq], [3,4,5])
        s = s.append(t)
        self.assertEqual(len(s), 6)
        self.assertEqual(s._sindex.size, 6)

@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestFrameSindex(unittest.TestCase):

    def test_sindex(self):
        crs = {'init': 'epsg:4326'}
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        df = GeoDataFrame(data, crs=crs, geometry='location')
        self.assertEqual(df._sindex.size, 5)
        hits = list(df._sindex.intersection((2.5, 2.5, 4, 4), objects=True))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].object, 3)

