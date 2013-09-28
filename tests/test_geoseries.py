import os
import shutil
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from pandas import Series
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries
from pandas import Series


def geom_equals(this, that):
    """
    Test for geometric equality, allowing all empty geometries to be considered equal
    """
    empty = np.logical_and(this.is_empty, that.is_empty)
    eq = this.equals(that)
    return np.all(np.logical_or(eq, empty))

def geom_almost_equals(this, that):
    """
    Test for geometric equality, allowing all empty geometries to be considered almost equal
    """
    empty = np.logical_and(this.is_empty, that.is_empty)
    eq = this.almost_equals(that)
    return np.all(np.logical_or(eq, empty))

class TestSeries(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g1 = GeoSeries([self.t1, self.sq])
        self.g2 = GeoSeries([self.sq, self.t1])
        self.g3 = GeoSeries([self.t1, self.t2])
        self.g3.crs = {'init': 'epsg:4326', 'no_defs': True}
        self.g4 = GeoSeries([self.t2, self.t1])
        self.na = GeoSeries([self.t1, self.t2, Polygon()])
        self.na_none = GeoSeries([self.t1, self.t2, None])
        self.a1 = self.g1.copy()
        self.a1.index = ['A', 'B']
        self.a2 = self.g2.copy()
        self.a2.index = ['B', 'C']
        self.esb = Point(-73.9847, 40.7484)
        self.sol = Point(-74.0446, 40.6893)
        self.landmarks = GeoSeries([self.esb, self.sol],
                                   crs={'init': 'epsg:4326', 'no_defs': True})
        self.l1 = LineString([(0, 0), (0, 1), (1, 1)])
        self.l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g5 = GeoSeries([self.l1, self.l2])

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_area(self):
        self.assertTrue(type(self.g1.area) is Series)
        assert_array_equal(self.g1.area.values, np.array([0.5, 1.0]))

    def test_in(self):
        self.assertTrue(self.t1 in self.g1)
        self.assertTrue(self.sq in self.g1)
        self.assertTrue(self.t1 in self.a1)
        self.assertTrue(self.t2 in self.g3)
        self.assertTrue(self.sq not in self.g3)
        self.assertTrue(5 not in self.g3)

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

    def test_to_file(self):
        """ Test to_file and from_file """
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        self.g3.to_file(tempfilename)
        # Read layer back in?
        s = GeoSeries.from_file(tempfilename)
        self.assertTrue(all(self.g3.equals(s)))
        # TODO: compare crs

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

    def test_fillna(self):
        na = self.na_none.fillna()
        self.assertTrue(isinstance(na[2], BaseGeometry))
        self.assertTrue(na[2].is_empty)
        
    def test_interpolate(self):
        res = self.g5.interpolate(0.75, normalized=True)
        self.assertTrue(geom_equals(res, GeoSeries([Point(0.5, 1.0),
                                                    Point(0.75, 1.0)])))
        res = self.g5.interpolate(1.5)
        self.assertTrue(geom_equals(res, GeoSeries([Point(0.5, 1.0),
                                                    Point(1.0, 0.5)])))
        
    def test_project(self):
        res = self.g5.project(Point(1.0, 0.5))
        assert_array_equal(res, [2.0, 1.5])
        res = self.g5.project(Point(1.0, 0.5), normalized=True)
        assert_array_equal(res, [1.0, 0.5])
        
    def test_translate_tuple(self):
        trans = self.sol.x - self.esb.x, self.sol.y - self.esb.y
        self.assertTrue(self.landmarks.translate(*trans)[0].equals(self.sol))
    
    def test_rotate(self):
        angle = 98
        res = self.g4.rotate(angle, origin=Point(0,0))
        self.assertTrue(geom_almost_equals(self.g4, res.rotate(-angle, 
            origin=Point(0,0))))

    def test_scale(self):
        scale = 2., 1.
        inv = tuple(1./i for i in scale)
        res = self.g4.scale(*scale, origin=Point(0,0))
        self.assertTrue(geom_almost_equals(self.g4, res.scale(*inv, 
            origin=Point(0,0))))
        
    def test_skew(self):
        skew = 45.
        res = self.g4.skew(xs=skew, origin=Point(0,0))
        self.assertTrue(geom_almost_equals(self.g4, res.skew(xs=-skew, 
            origin=Point(0,0))))
        res = self.g4.skew(ys=skew, origin=Point(0,0))
        self.assertTrue(geom_almost_equals(self.g4, res.skew(ys=-skew, 
            origin=Point(0,0))))

