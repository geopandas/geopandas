import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_series_equal, assert_frame_equal
from pandas import Series, DataFrame
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.collection import GeometryCollection

from geopandas import GeoSeries, GeoDataFrame
from geopandas.base import GeoPandasBase
from tests.util import geom_equals, geom_almost_equals, assert_geoseries_equal

class TestGeomMethods(unittest.TestCase):

    def setUp(self):
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

        # Placeholder for testing, will just drop in different geometries
        # when needed
        self.gdf1 = GeoDataFrame({'geometry' : self.g1,
                                  'col0' : [1.0, 2.0],
                                  'col1' : ['geo', 'pandas']})
        self.gdf2 = GeoDataFrame({'geometry' : self.g1,
                                  'col3' : [4, 5],
                                  'col4' : ['rand', 'string']})


    def _test_unary_real(self, op, expected, a):
        fcmp = assert_series_equal
        self._test_unary(op, expected, a, fcmp)

    def _test_unary_topological(self, op, expected, a):
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:
            fcmp = lambda a, b: self.assert_(geom_equals(a, b))
        self._test_unary(op, expected, a, fcmp)

    def _test_binary_topological(self, op, expected, a, b, *args, **kwargs):
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:
            fcmp = lambda a, b: self.assert_(geom_equals(a, b))
        self._test_binary(op, expected, a, b, fcmp, *args, **kwargs)

    def _test_binary_real(self, op, expected, a, b, *args, **kwargs):
        fcmp = assert_series_equal
        self._test_binary(op, expected, a, b, fcmp, *args, **kwargs)

    def _test_binary(self, op, expected, a, b, fcmp, *args, **kwargs):
        # GeoSeries, (GeoSeries or geometry)
        result = getattr(a, op)(b, *args, **kwargs)
        fcmp(result, expected)

        # GeoDataFrame, (GeoSeries or geometry)
        gdf = self.gdf1.set_geometry(a)
        result = getattr(gdf, op)(b, *args, **kwargs)
        fcmp(result, expected)

        if isinstance(b, GeoPandasBase):
            # GeoSeries, GeoDataFrame
            gdf = self.gdf1.set_geometry(b)
            result = getattr(a, op)(gdf, *args, **kwargs)
            fcmp(result, expected)

            # GeoDataFrame, GeoDataFrame
            gdfa = self.gdf1.set_geometry(a)
            gdfb = self.gdf2.set_geometry(b)
            result = getattr(gdfa, op)(gdfb, *args, **kwargs)
            fcmp(result, expected)

    def _test_unary(self, op, expected, a, fcmp):
        # GeoSeries, (GeoSeries or geometry)
        result = getattr(a, op)
        fcmp(result, expected)

        # GeoDataFrame, (GeoSeries or geometry)
        gdf = self.gdf1.set_geometry(a)
        result = getattr(gdf, op)
        fcmp(result, expected)

    def test_intersection(self):
        self._test_binary_topological('intersection', self.t1, 
                                      self.g1, self.g2)
        self._test_binary_topological('__and__', self.t1, self.g1, self.g2)

    def test_union_series(self):
        self._test_binary_topological('union', self.sq, self.g1, self.g2)
        self._test_binary_topological('__or__', self.sq, self.g1, self.g2)

    def test_union_polygon(self):
        self._test_binary_topological('union', self.sq, self.g1, self.t2)
        self._test_binary_topological('__or__', self.sq, self.g1, self.t2)

    def test_symmetric_difference_series(self):
        self._test_binary_topological('symmetric_difference', self.sq,
                                      self.g3, self.g4)
        self._test_binary_topological('__xor__', self.sq, self.g3, self.g4)

    def test_symmetric_difference_poly(self):
        expected = GeoSeries([GeometryCollection(), self.sq], crs=self.g3.crs)
        self._test_binary_topological('symmetric_difference', expected,
                                      self.g3, self.t1)

    def test_difference_series(self):
        expected = GeoSeries([GeometryCollection(), self.t2])
        self._test_binary_topological('difference', expected,
                                      self.g1, self.g2)
        self._test_binary_topological('__sub__', expected, self.g1, self.g2)

    def test_difference_poly(self):
        expected = GeoSeries([self.t1, self.t1])
        self._test_binary_topological('difference', expected,
                                      self.g1, self.t2)
        self._test_binary_topological('__sub__', expected,
                                      self.g1, self.t2)

    def test_boundary(self):
        l1 = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
        l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        expected = GeoSeries([l1, l2], index=self.g1.index, crs=self.g1.crs)

        self._test_unary_topological('boundary', expected, self.g1)

    def test_area(self):
        expected = Series(np.array([0.5, 1.0]), index=self.g1.index)
        self._test_unary_real('area', expected, self.g1)

    def test_bounds(self):
        # Set columns to get the order right
        expected = DataFrame({'minx': [0.0, 0.0], 'miny': [0.0, 0.0],
                              'maxx': [1.0, 1.0], 'maxy': [1.0, 1.0]},
                              index=self.g1.index,
                              columns=['minx', 'miny', 'maxx', 'maxy'])

        
        result = self.g1.bounds
        assert_frame_equal(expected, result)

        gdf = self.gdf1.set_geometry(self.g1)
        result = gdf.bounds
        assert_frame_equal(expected, result)

    def test_contains(self):
        expected = np.array([True] * len(self.g1))
        assert_array_equal(expected, self.g1.contains(self.t1))

        expected = np.array([False] * len(self.g1))
        assert_array_equal(expected, self.g1.contains(Point(5,5)))

    def test_length(self):
        expected = Series(np.array([2 + np.sqrt(2), 4]), index=self.g1.index)
        self._test_unary_real('length', expected, self.g1)


    @unittest.skip('TODO')
    def test_crosses(self):
        # TODO
        pass

    @unittest.skip('TODO')
    def test_disjoint(self):
        # TODO
        pass

    @unittest.skip('TODO')
    def test_intersects(self):
        # TODO
        pass

    @unittest.skip('TODO')
    def test_overlaps(self):
        # TODO
        pass

    @unittest.skip('TODO')
    def test_touches(self):
        # TODO
        pass

    @unittest.skip('TODO')
    def test_within(self):
        # TODO
        pass

    def test_is_valid(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real('is_valid', expected, self.g1)

    def test_is_empty(self):
        expected = Series(np.array([False] * len(self.g1)), self.g1.index)
        self._test_unary_real('is_empty', expected, self.g1)

    def test_is_ring(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real('is_ring', expected, self.g1)

    def test_is_simple(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real('is_simple', expected, self.g1)

    @unittest.skip('TODO')
    def test_exterior(self):
        # TODO
        pass

    @unittest.skip('TODO')
    def test_interiors(self):
        # TODO
        pass

    def test_interpolate(self):
        expected = GeoSeries([Point(0.5, 1.0), Point(0.75, 1.0)])
        self._test_binary_topological('interpolate', expected, self.g5,
                                      0.75, normalized=True)

        expected = GeoSeries([Point(0.5, 1.0), Point(1.0, 0.5)])
        self._test_binary_topological('interpolate', expected, self.g5,
                                      1.5)

    def test_project(self):
        expected = Series([2.0, 1.5], index=self.g5.index)
        p = Point(1.0, 0.5)
        self._test_binary_real('project', expected, self.g5, p)

        expected = Series([1.0, 0.5], index=self.g5.index)
        self._test_binary_real('project', expected, self.g5, p, 
                               normalized=True)

    def test_translate_tuple(self):
        trans = self.sol.x - self.esb.x, self.sol.y - self.esb.y
        self.assert_(self.landmarks.translate(*trans)[0].equals(self.sol))

        res = self.gdf1.set_geometry(self.landmarks).translate(*trans)[0]
        self.assert_(res.equals(self.sol))

    def test_rotate(self):
        angle = 98
        expected = self.g4

        o = Point(0,0)
        res = self.g4.rotate(angle, origin=o).rotate(-angle, origin=o)
        self.assert_(geom_almost_equals(self.g4, res))

        res = self.gdf1.set_geometry(self.g4).rotate(angle, origin=Point(0,0))
        self.assert_(geom_almost_equals(expected,
                                        res.rotate(-angle, origin=o)))

    def test_scale(self):
        expected = self.g4

        scale = 2., 1.
        inv = tuple(1./i for i in scale)

        o = Point(0,0)
        res = self.g4.scale(*scale, origin=o).scale(*inv, origin=o)
        self.assertTrue(geom_almost_equals(expected, res))

        res = self.gdf1.set_geometry(self.g4).scale(*scale, origin=o)
        res = res.scale(*inv, origin=o)
        self.assert_(geom_almost_equals(expected, res))

    def test_skew(self):
        expected = self.g4

        skew = 45.
        o = Point(0,0)

        # Test xs
        res = self.g4.skew(xs=skew, origin=o).skew(xs=-skew, origin=o)
        self.assert_(geom_almost_equals(expected, res))

        res = self.gdf1.set_geometry(self.g4).skew(xs=skew, origin=o)
        res = res.skew(xs=-skew, origin=o)
        self.assert_(geom_almost_equals(expected, res))

        # Test ys
        res = self.g4.skew(ys=skew, origin=o).skew(ys=-skew, origin=o)
        self.assert_(geom_almost_equals(expected, res))

        res = self.gdf1.set_geometry(self.g4).skew(ys=skew, origin=o)
        res = res.skew(ys=-skew, origin=o)
        self.assert_(geom_almost_equals(expected, res))

    def test_envelope(self):
        e = self.g3.envelope
        self.assertTrue(np.alltrue(e.equals(self.sq)))
        self.assertIsInstance(e, GeoSeries)
        self.assertEqual(self.g3.crs, e.crs)

    def test_total_bounds(self):
        bbox = self.sol.x, self.sol.y, self.esb.x, self.esb.y
        self.assert_(self.landmarks.total_bounds, bbox)

        df = GeoDataFrame({'geometry': self.landmarks,
                           'col1': range(len(self.landmarks))})
        self.assert_(df.total_bounds, bbox)
