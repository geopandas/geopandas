from __future__ import absolute_import

import os
import shutil
import tempfile
import numpy as np
from numpy.testing import assert_array_equal
from shapely.geometry import (Polygon, Point, LineString,
                              MultiPoint, MultiLineString, MultiPolygon)
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries
from geopandas.tests.util import unittest, geom_equals


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

    def test_single_geom_constructor(self):
        p = Point(1,2)
        line = LineString([(2, 3), (4, 5), (5, 6)])
        poly = Polygon([(0, 0), (1, 0), (1, 1)],
                          [[(.1, .1), (.9, .1), (.9, .9)]])
        mp = MultiPoint([(1, 2), (3, 4), (5, 6)])
        mline = MultiLineString([[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10)]])

        poly2 = Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1)],
                        [[(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)]])
        mpoly = MultiPolygon([poly, poly2])

        geoms = [p, line, poly, mp, mline, mpoly]
        index = ['a', 'b', 'c', 'd']

        for g in geoms:
            gs = GeoSeries(g)
            self.assert_(len(gs) == 1)
            self.assert_(gs.iloc[0] is g)

            gs = GeoSeries(g, index=index)
            self.assert_(len(gs) == len(index))
            for x in gs:
                self.assert_(x is g)

    def test_copy(self):
        gc = self.g3.copy()
        self.assertTrue(type(gc) is GeoSeries)
        self.assertEqual(self.g3.name, gc.name)
        self.assertEqual(self.g3.crs, gc.crs)

    def test_in(self):
        self.assertTrue(self.t1 in self.g1)
        self.assertTrue(self.sq in self.g1)
        self.assertTrue(self.t1 in self.a1)
        self.assertTrue(self.t2 in self.g3)
        self.assertTrue(self.sq not in self.g3)
        self.assertTrue(5 not in self.g3)

    def test_geom_equals(self):
        self.assertTrue(np.alltrue(self.g1.geom_equals(self.g1)))
        assert_array_equal(self.g1.geom_equals(self.sq), [False, True])

    def test_geom_equals_align(self):
        a = self.a1.geom_equals(self.a2)
        self.assertFalse(a['A'])
        self.assertTrue(a['B'])
        self.assertFalse(a['C'])

    def test_align(self):
        a1, a2 = self.a1.align(self.a2)
        self.assertTrue(a2['A'].is_empty)
        self.assertTrue(a1['B'].equals(a2['B']))
        self.assertTrue(a1['C'].is_empty)

    def test_geom_almost_equals(self):
        # TODO: test decimal parameter
        self.assertTrue(np.alltrue(self.g1.geom_almost_equals(self.g1)))
        assert_array_equal(self.g1.geom_almost_equals(self.sq), [False, True])

    def test_geom_equals_exact(self):
        # TODO: test tolerance parameter
        self.assertTrue(np.alltrue(self.g1.geom_equals_exact(self.g1, 0.001)))
        assert_array_equal(self.g1.geom_equals_exact(self.sq, 0.001), [False, True])

    def test_to_file(self):
        """ Test to_file and from_file """
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        self.g3.to_file(tempfilename)
        # Read layer back in?
        s = GeoSeries.from_file(tempfilename)
        self.assertTrue(all(self.g3.geom_equals(s)))
        # TODO: compare crs

    def test_representative_point(self):
        self.assertTrue(np.alltrue(self.g1.contains(self.g1.representative_point())))
        self.assertTrue(np.alltrue(self.g2.contains(self.g2.representative_point())))
        self.assertTrue(np.alltrue(self.g3.contains(self.g3.representative_point())))
        self.assertTrue(np.alltrue(self.g4.contains(self.g4.representative_point())))

    def test_transform(self):
        utm18n = self.landmarks.to_crs(epsg=26918)
        lonlat = utm18n.to_crs(epsg=4326)
        self.assertTrue(np.alltrue(self.landmarks.geom_almost_equals(lonlat)))
        with self.assertRaises(ValueError):
            self.g1.to_crs(epsg=4326)
        with self.assertRaises(TypeError):
            self.landmarks.to_crs(crs=None, epsg=None)

    def test_fillna(self):
        na = self.na_none.fillna(Point())
        self.assertTrue(isinstance(na[2], BaseGeometry))
        self.assertTrue(na[2].is_empty)
        self.assertTrue(geom_equals(self.na_none[:2], na[:2]))
        # XXX: method works inconsistently for different pandas versions
        #self.na_none.fillna(method='backfill')

    def test_coord_slice(self):
        """ Test CoordinateSlicer """
        # need some better test cases
        self.assertTrue(geom_equals(self.g3, self.g3.cx[:, :]))
        self.assertTrue(geom_equals(self.g3[[True, False]], self.g3.cx[0.9:, :0.1]))
        self.assertTrue(geom_equals(self.g3[[False, True]], self.g3.cx[0:0.1, 0.9:1.0]))

    def test_geoseries_geointerface(self):
        self.assertEqual(self.g1.__geo_interface__['type'], 'FeatureCollection')
        self.assertEqual(len(self.g1.__geo_interface__['features']),
                         self.g1.shape[0])

if __name__ == '__main__':
    unittest.main()
