import unittest
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from geopandas import GeoSeries


def geom_equals(this, that):
    """
    Test for geometric equality, allowing all empty geometries to be considered equal
    """
    empty = np.logical_and(this.is_empty, that.is_empty)
    eq = this.equals(that)
    return np.all(np.logical_or(eq, empty))


class TestSeries(unittest.TestCase):

    def setUp(self):
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g1 = GeoSeries([self.t1, self.sq])
        self.g2 = GeoSeries([self.sq, self.t1])
        self.g3 = GeoSeries([self.t1, self.t2])
        self.g4 = GeoSeries([self.t2, self.t1])
        self.a1 = self.g1.copy()
        self.a1.index = ['A', 'B']
        self.a2 = self.g2.copy()
        self.a2.index = ['B', 'C']

    def test_area(self):
        assert np.allclose(self.g1.area.values, np.array([0.5, 1.0]))

    def test_in(self):
        assert self.t1 in self.g1
        assert self.sq in self.g1
        assert self.t1 in self.a1
        assert self.t2 in self.g3

    def test_boundary(self):
        l1 = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
        l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        b = self.g1.boundary
        assert b[0].equals(l1)
        assert b[1].equals(l2)

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

    def test_equals_align(self):
        a = self.a1.equals(self.a2)
        assert a['A'] == False
        assert a['B'] == True
        assert a['C'] == False

    def test_align(self):
        a1, a2 = self.a1.align(self.a2)
        assert a2['A'].is_empty
        assert a1['B'].equals(a2['B'])
        assert a1['C'].is_empty

    def test_almost_equals(self):
        assert np.alltrue(self.g1.equals(self.g1))
        assert np.all(self.g1.equals(self.sq).values == np.array([0, 1], dtype=bool))

    def test_equals_exact(self):
        assert np.alltrue(self.g1.equals(self.g1))
        assert np.all(self.g1.equals(self.sq).values == np.array([0, 1], dtype=bool))

    def test_crosses(self):
        # TODO
        pass

    def test_disjoint(self):
        # TODO
        pass

    def test_intersects(self):
        # TODO
        pass

    def test_overlaps(self):
        # TODO
        pass

    def test_touches(self):
        # TODO
        pass

    def test_within(self):
        # TODO
        pass

    def test_intersection(self):
        assert geom_equals(self.g1 & self.g2, self.t1)

    def test_union_series(self):
        u = self.g1.union(self.g2)
        assert u[0].equals(self.sq)
        assert u[1].equals(self.sq)
        assert geom_equals(u, self.g1 | self.g2)

    def test_union_polgon(self):
        u = self.g1.union(self.t2)
        assert u[0].equals(self.sq)
        assert u[1].equals(self.sq)

    def test_symmetric_difference_series(self):
        u = self.g3.symmetric_difference(self.g4)
        assert u[0].equals(self.sq)
        assert u[1].equals(self.sq)
        assert geom_equals(u, self.g3 ^ self.g4)

    def test_symmetric_difference_poly(self):
        u = self.g3.symmetric_difference(self.t1)
        assert u[0].is_empty
        assert u[1].equals(self.sq)

    def test_difference_series(self):
        u = self.g1.difference(self.g2)
        assert u[0].is_empty
        assert u[1].equals(self.t2)
        assert geom_equals(u, self.g1 - self.g2)

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

    def test_envelope(self):
        e = self.g3.envelope
        assert np.alltrue(e.equals(self.sq))

    def test_exterior(self):
        # TODO
        pass

    def test_interiors(self):
        # TODO
        pass

    def test_representative_point(self):
        assert np.alltrue(self.g1.contains(self.g1.representative_point()))
        assert np.alltrue(self.g2.contains(self.g2.representative_point()))
        assert np.alltrue(self.g3.contains(self.g3.representative_point()))
        assert np.alltrue(self.g4.contains(self.g4.representative_point()))
