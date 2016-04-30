from __future__ import absolute_import

from shapely.geometry import Point, MultiPoint, LineString
from geopandas import GeoSeries
from geopandas.tools import collect
from geopandas.tests.util import unittest


class TestTools(unittest.TestCase):
    def setUp(self):
        self.p1 = Point(0, 0)
        self.p2 = Point(1, 1)
        self.p3 = Point(2, 2)
        self.mpc = MultiPoint([self.p1, self.p2, self.p3])

        self.mp1 = MultiPoint([self.p1, self.p2])
        self.line1 = LineString([(3, 3), (4, 4)])

    def test_collect_single(self):
        result = collect(self.p1)
        self.assert_(self.p1.equals(result))

    def test_collect_single_force_multi(self):
        result = collect(self.p1, multi=True)
        expected = MultiPoint([self.p1])
        self.assert_(expected.equals(result))

    def test_collect_multi(self):
        result = collect(self.mp1)
        self.assert_(self.mp1.equals(result))

    def test_collect_multi_force_multi(self):
        result = collect(self.mp1)
        self.assert_(self.mp1.equals(result))

    def test_collect_list(self):
        result = collect([self.p1, self.p2, self.p3])
        self.assert_(self.mpc.equals(result))

    def test_collect_GeoSeries(self):
        s = GeoSeries([self.p1, self.p2, self.p3])
        result = collect(s)
        self.assert_(self.mpc.equals(result))

    def test_collect_mixed_types(self):
        with self.assertRaises(ValueError):
            collect([self.p1, self.line1])

    def test_collect_mixed_multi(self):
        with self.assertRaises(ValueError):
            collect([self.mpc, self.mp1])
