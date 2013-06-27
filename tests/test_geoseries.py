import unittest
import numpy as np
from shapely.geometry import Polygon, Point
from geopandas import GeoSeries


class TestSeries(unittest.TestCase):

    def setUp(self):
        self.p1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g1 = GeoSeries([self.p1, self.p2])
        self.g2 = GeoSeries([self.p2, self.p1])

    def test_area(self):
        assert np.allclose(self.g1.area.values, np.array([0.5, 1.0]))

    def test_bounds(self):
        assert np.allclose(self.g1.bounds.values, np.array([[0, 0, 1, 1],
                                                            [0, 0, 1, 1]]))

    def test_contains(self):
        assert np.alltrue(self.g1.contains(self.p1))
        assert not np.alltrue(self.g1.contains(Point([5, 5])))

    def test_length(self):
        l = np.array([2 + np.sqrt(2), 4])
        assert np.allclose(self.g1.length.values, l)

    def test_is_valid(self):
        assert np.alltrue(self.g1.is_valid)

    def test_is_empty(self):
        assert np.alltrue(np.logical_not(self.g1.is_empty))

    def test_is_ring(self):
        assert np.alltrue(self.g1.is_ring)

    def test_is_simple(self):
        assert np.alltrue(self.g1.is_simple)
