import unittest
import numpy as np
from shapely.geometry import Polygon, Point
from geopandas.series import GeoSeries


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
