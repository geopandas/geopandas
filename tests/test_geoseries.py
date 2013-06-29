import unittest
import numpy as np
from shapely.geometry import Polygon, Point
from geopandas import GeoSeries


class TestSeries(unittest.TestCase):

    def setUp(self):
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g1 = GeoSeries([self.t1, self.sq])
        self.g2 = GeoSeries([self.sq, self.t1])
        self.g3 = GeoSeries([self.t1, self.t2])
        self.g4 = GeoSeries([self.t2, self.t1])

    def test_area(self):
        assert np.allclose(self.g1.area.values, np.array([0.5, 1.0]))

    def test_bounds(self):
        assert np.allclose(self.g1.bounds.values, np.array([[0, 0, 1, 1],
                                                            [0, 0, 1, 1]]))

    def test_contains(self):
        assert np.alltrue(self.g1.contains(self.t1))
        assert not np.alltrue(self.g1.contains(Point([5, 5])))

    def test_length(self):
        l = np.array([2 + np.sqrt(2), 4])
        assert np.allclose(self.g1.length.values, l)

    def test_equals(self):
        assert np.alltrue(self.g1.equals(self.g1))
        assert np.all(self.g1.equals(self.sq).values == np.array([0, 1], dtype=bool))

    def test_almost_equals(self):
        assert np.alltrue(self.g1.equals(self.g1))
        assert np.all(self.g1.equals(self.sq).values == np.array([0, 1], dtype=bool))

    def test_equals_exact(self):
        assert np.alltrue(self.g1.equals(self.g1))
        assert np.all(self.g1.equals(self.sq).values == np.array([0, 1], dtype=bool))

    def test_union_series(self):
        u = self.g1.union(self.g2)
        assert u[0].equals(self.sq)
        assert u[1].equals(self.sq)

    def test_union_polgon(self):
        u = self.g1.union(self.t2)
        assert u[0].equals(self.sq)
        assert u[1].equals(self.sq)

    def test_difference_series(self):
        u = self.g1.difference(self.g2)
        assert u[0].is_empty
        assert u[1].equals(self.t2)

    def test_symmetric_difference_series(self):
        u = self.g3.symmetric_difference(self.g4)
        assert u[0].equals(self.sq)
        assert u[1].equals(self.sq)

    def test_symmetric_difference_poly(self):
        u = self.g3.symmetric_difference(self.t1)
        assert u[0].is_empty
        assert u[1].equals(self.sq)

    def test_difference_poly(self):
        u = self.g1.difference(self.t2)
        assert u[0].equals(self.t1)
        assert u[1].equals(self.t1)

    def test_is_valid(self):
        assert np.alltrue(self.g1.is_valid)

    def test_is_empty(self):
        assert np.alltrue(np.logical_not(self.g1.is_empty))

    def test_is_ring(self):
        assert np.alltrue(self.g1.is_ring)

    def test_is_simple(self):
        assert np.alltrue(self.g1.is_simple)
