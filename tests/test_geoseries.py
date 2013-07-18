import unittest
import numpy as np
from numpy.testing import assert_array_equal
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
        self.g3.crs = {'init': 'epsg:4326', 'no_defs': True}
        self.g4 = GeoSeries([self.t2, self.t1])
        self.a1 = self.g1.copy()
        self.a1.index = ['A', 'B']
        self.a2 = self.g2.copy()
        self.a2.index = ['B', 'C']
        self.esb = Point(-73.9847, 40.7484)
        self.sol = Point(-74.0446, 40.6893)
        self.landmarks = GeoSeries([self.esb, self.sol],
                                   crs={'init': 'epsg:4326', 'no_defs': True})

    def test_area(self):
        assert_array_equal(self.g1.area.values, np.array([0.5, 1.0]))

    def test_in(self):
        assert self.t1 in self.g1
        assert self.sq in self.g1
        assert self.t1 in self.a1
        assert self.t2 in self.g3

    def test_boundary(self):
        l1 = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
        l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        b = self.g1.boundary
        self.assertTrue(b[0].equals(l1))
        self.assertTrue(b[1].equals(l2))

    def test_bounds(self):
        assert_array_equal(self.g1.bounds.values, np.array([[0, 0, 1, 1],
                                                            [0, 0, 1, 1]]))

    def test_contains(self):
        self.assertTrue(np.alltrue(self.g1.contains(self.t1)))
        self.assertFalse(np.alltrue(self.g1.contains(Point([5, 5]))))

    def test_length(self):
        l = np.array([2 + np.sqrt(2), 4])
        assert_array_equal(self.g1.length.values, l)

    def test_equals(self):
        self.assertTrue(np.alltrue(self.g1.equals(self.g1)))
        assert_array_equal(self.g1.equals(self.sq), [False, True])

    def test_equals_align(self):
        a = self.a1.equals(self.a2)
        self.assertFalse(a['A'])
        self.assertTrue(a['B'])
        self.assertFalse(a['C'])

    def test_align(self):
        a1, a2 = self.a1.align(self.a2)
        self.assertTrue(a2['A'].is_empty)
        self.assertTrue(a1['B'].equals(a2['B']))
        self.assertTrue(a1['C'].is_empty)

    def test_almost_equals(self):
        # TODO: test decimal parameter
        self.assertTrue(np.alltrue(self.g1.almost_equals(self.g1)))
        assert_array_equal(self.g1.almost_equals(self.sq), [False, True])

    def test_equals_exact(self):
        # TODO: test tolerance parameter
        self.assertTrue(np.alltrue(self.g1.equals_exact(self.g1, 0.001)))
        assert_array_equal(self.g1.equals_exact(self.sq, 0.001), [False, True])

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
        self.assertTrue(geom_equals(self.g1 & self.g2, self.t1))

    def test_union_series(self):
        u = self.g1.union(self.g2)
        self.assertTrue(u[0].equals(self.sq))
        self.assertTrue(u[1].equals(self.sq))
        self.assertTrue(geom_equals(u, self.g1 | self.g2))

    def test_union_polgon(self):
        u = self.g1.union(self.t2)
        self.assertTrue(u[0].equals(self.sq))
        self.assertTrue(u[1].equals(self.sq))

    def test_symmetric_difference_series(self):
        u = self.g3.symmetric_difference(self.g4)
        self.assertTrue(u[0].equals(self.sq))
        self.assertTrue(u[1].equals(self.sq))
        self.assertTrue(geom_equals(u, self.g3 ^ self.g4))
        self.assertEqual(self.g3.crs, u.crs)

    def test_symmetric_difference_poly(self):
        u = self.g3.symmetric_difference(self.t1)
        self.assertTrue(u[0].is_empty)
        self.assertTrue(u[1].equals(self.sq))
        self.assertEqual(self.g3.crs, u.crs)

    def test_difference_series(self):
        u = self.g1.difference(self.g2)
        self.assertTrue(u[0].is_empty)
        self.assertTrue(u[1].equals(self.t2))
        self.assertTrue(geom_equals(u, self.g1 - self.g2))

    def test_difference_poly(self):
        u = self.g1.difference(self.t2)
        self.assertTrue(u[0].equals(self.t1))
        self.assertTrue(u[1].equals(self.t1))

    def test_is_valid(self):
        self.assertTrue(np.alltrue(self.g1.is_valid))

    def test_is_empty(self):
        self.assertTrue(np.alltrue(np.logical_not(self.g1.is_empty)))

    def test_is_ring(self):
        self.assertTrue(np.alltrue(self.g1.is_ring))

    def test_is_simple(self):
        self.assertTrue(np.alltrue(self.g1.is_simple))

    def test_envelope(self):
        e = self.g3.envelope
        self.assertTrue(np.alltrue(e.equals(self.sq)))
        self.assertIsInstance(e, GeoSeries)
        self.assertEqual(self.g3.crs, e.crs)

    def test_exterior(self):
        # TODO
        pass

    def test_interiors(self):
        # TODO
        pass

    def test_representative_point(self):
        self.assertTrue(np.alltrue(self.g1.contains(self.g1.representative_point())))
        self.assertTrue(np.alltrue(self.g2.contains(self.g2.representative_point())))
        self.assertTrue(np.alltrue(self.g3.contains(self.g3.representative_point())))
        self.assertTrue(np.alltrue(self.g4.contains(self.g4.representative_point())))

    def test_transform(self):
        utm18n = self.landmarks.to_crs(epsg=26918)
        lonlat = utm18n.to_crs(epsg=4326)
        self.assertTrue(np.alltrue(self.landmarks.almost_equals(lonlat)))
