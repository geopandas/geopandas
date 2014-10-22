from __future__ import absolute_import

import string

import numpy as np
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_series_equal, assert_frame_equal
from pandas import Series, DataFrame, MultiIndex
from shapely.geometry import (
    Point, LinearRing, LineString, Polygon, MultiPoint
)
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union

from geopandas import GeoSeries, GeoDataFrame
from geopandas.base import GeoPandasBase
from .util import (
    unittest, geom_equals, geom_almost_equals, assert_geoseries_equal
)

class TestGeomMethods(unittest.TestCase):

    def setUp(self):
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.inner_sq = Polygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75),
                            (0.25, 0.75)])
        self.nested_squares = Polygon(self.sq.boundary,
                                      [self.inner_sq.boundary])
        self.p0 = Point(5, 5)
        self.g0 = GeoSeries([self.t1, self.t2, self.sq, self.inner_sq,
                             self.nested_squares, self.p0])
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

        # Crossed lines
        self.l3 = LineString([(0, 0), (1, 1)])
        self.l4 = LineString([(0, 1), (1, 0)])
        self.crossed_lines = GeoSeries([self.l3, self.l4])

        # Placeholder for testing, will just drop in different geometries
        # when needed
        self.gdf1 = GeoDataFrame({'geometry' : self.g1,
                                  'col0' : [1.0, 2.0],
                                  'col1' : ['geo', 'pandas']})
        self.gdf2 = GeoDataFrame({'geometry' : self.g1,
                                  'col3' : [4, 5],
                                  'col4' : ['rand', 'string']})


    def _test_unary_real(self, op, expected, a):
        """ Tests for 'area', 'length', 'is_valid', etc. """
        fcmp = assert_series_equal
        self._test_unary(op, expected, a, fcmp)

    def _test_unary_topological(self, op, expected, a):
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:
            fcmp = lambda a, b: self.assert_(a.equals(b))
        self._test_unary(op, expected, a, fcmp)

    def _test_binary_topological(self, op, expected, a, b, *args, **kwargs):
        """ Tests for 'intersection', 'union', 'symmetric_difference', etc. """
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:
            fcmp = lambda a, b: self.assert_(geom_equals(a, b))

        if isinstance(b, GeoPandasBase):
            right_df = True
        else:
            right_df = False

        self._binary_op_test(op, expected, a, b, fcmp, True, right_df, 
                        *args, **kwargs)

    def _test_binary_real(self, op, expected, a, b, *args, **kwargs):
        fcmp = assert_series_equal
        self._binary_op_test(op, expected, a, b, fcmp, True, False, *args, **kwargs)

    def _test_binary_operator(self, op, expected, a, b):
        """
        The operators only have GeoSeries on the left, but can have
        GeoSeries or GeoDataFrame on the right.

        """
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:
            fcmp = lambda a, b: self.assert_(geom_equals(a, b))

        if isinstance(b, GeoPandasBase):
            right_df = True
        else:
            right_df = False

        self._binary_op_test(op, expected, a, b, fcmp, False, right_df)

    def _binary_op_test(self, op, expected, left, right, fcmp, left_df,
                        right_df, 
                        *args, **kwargs):
        """
        This is a helper to call a function on GeoSeries and GeoDataFrame
        arguments. For example, 'intersection' is a member of both GeoSeries
        and GeoDataFrame and can take either GeoSeries or GeoDataFrame inputs.
        This function has the ability to test all four combinations of input
        types.

        Parameters
        ----------
        
        expected : str
            The operation to be tested. e.g., 'intersection'
        left: GeoSeries
        right: GeoSeries
        fcmp: function 
            Called with the result of the operation and expected. It should
            assert if the result is incorrect
        left_df: bool
            If the left input should also be called with a GeoDataFrame
        right_df: bool
            Indicates whether the right input should be called with a
            GeoDataFrame

        """
        def _make_gdf(s):
            n = len(s)
            col1 = string.ascii_lowercase[:n]
            col2 = range(n)
            
            return GeoDataFrame({'geometry': s.values, 
                                 'col1' : col1, 
                                 'col2' : col2},
                                 index=s.index, crs=s.crs)

        # Test GeoSeries.op(GeoSeries)
        result = getattr(left, op)(right, *args, **kwargs)
        fcmp(result, expected)
        
        if left_df:
            # Test GeoDataFrame.op(GeoSeries)
            gdf_left = _make_gdf(left)
            result = getattr(gdf_left, op)(right, *args, **kwargs)
            fcmp(result, expected)

        if right_df:
            # Test GeoSeries.op(GeoDataFrame)
            gdf_right = _make_gdf(right)
            result = getattr(left, op)(gdf_right, *args, **kwargs)
            fcmp(result, expected)

            if left_df:
                # Test GeoDataFrame.op(GeoDataFrame)
                result = getattr(gdf_left, op)(gdf_right, *args, **kwargs)
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

    def test_union_series(self):
        self._test_binary_topological('union', self.sq, self.g1, self.g2)

    def test_union_polygon(self):
        self._test_binary_topological('union', self.sq, self.g1, self.t2)

    def test_symmetric_difference_series(self):
        self._test_binary_topological('symmetric_difference', self.sq,
                                      self.g3, self.g4)

    def test_symmetric_difference_poly(self):
        expected = GeoSeries([GeometryCollection(), self.sq], crs=self.g3.crs)
        self._test_binary_topological('symmetric_difference', expected,
                                      self.g3, self.t1)

    def test_difference_series(self):
        expected = GeoSeries([GeometryCollection(), self.t2])
        self._test_binary_topological('difference', expected,
                                      self.g1, self.g2)

    def test_difference_poly(self):
        expected = GeoSeries([self.t1, self.t1])
        self._test_binary_topological('difference', expected,
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

    def test_unary_union(self):
        p1 = self.t1
        p2 = Polygon([(2, 0), (3, 0), (3, 1)])
        expected = unary_union([p1, p2])
        g = GeoSeries([p1, p2])

        self._test_unary_topological('unary_union', expected, g)

    def test_contains(self):
        expected = [True, False, True, False, False, False]
        assert_array_equal(expected, self.g0.contains(self.t1))

    def test_length(self):
        expected = Series(np.array([2 + np.sqrt(2), 4]), index=self.g1.index)
        self._test_unary_real('length', expected, self.g1)

    def test_crosses(self):
        expected = [False, False, False, False, False, False]
        assert_array_equal(expected, self.g0.crosses(self.t1))

        expected = [False, True]
        assert_array_equal(expected, self.crossed_lines.crosses(self.l3))

    def test_disjoint(self):
        expected = [False, False, False, False, False, True]
        assert_array_equal(expected, self.g0.disjoint(self.t1))

    def test_intersects(self):
        expected = [True, True, True, True, True, False]
        assert_array_equal(expected, self.g0.intersects(self.t1))

    def test_overlaps(self):
        expected = [True, True, False, False, False, False]
        assert_array_equal(expected, self.g0.overlaps(self.inner_sq))

        expected = [False, False]
        assert_array_equal(expected, self.g4.overlaps(self.t1))

    def test_touches(self):
        expected = [False, True, False, False, False, False]
        assert_array_equal(expected, self.g0.touches(self.t1))

    def test_within(self):
        expected = [True, False, False, False, False, False]
        assert_array_equal(expected, self.g0.within(self.t1))

        expected = [True, True, True, True, True, False]
        assert_array_equal(expected, self.g0.within(self.sq))

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

    def test_exterior(self):
        exp_exterior = GeoSeries([LinearRing(p.boundary) for p in self.g3])
        for expected, computed in zip(exp_exterior, self.g3.exterior):
            assert computed.equals(expected)

    def test_interiors(self):
        square_series = GeoSeries(self.nested_squares)
        exp_interiors = GeoSeries([LinearRing(self.inner_sq.boundary)])
        for expected, computed in zip(exp_interiors, square_series.interiors):
            assert computed[0].equals(expected)


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
        self.assertTrue(np.alltrue(e.geom_equals(self.sq)))
        self.assertIsInstance(e, GeoSeries)
        self.assertEqual(self.g3.crs, e.crs)

    def test_total_bounds(self):
        bbox = self.sol.x, self.sol.y, self.esb.x, self.esb.y
        self.assert_(self.landmarks.total_bounds, bbox)

        df = GeoDataFrame({'geometry': self.landmarks,
                           'col1': range(len(self.landmarks))})
        self.assert_(df.total_bounds, bbox)

    def test_explode(self):
        s = GeoSeries([MultiPoint([(0,0), (1,1)]),
                      MultiPoint([(2,2), (3,3), (4,4)])])

        index = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        expected = GeoSeries([Point(0,0), Point(1,1), Point(2,2), Point(3,3),
                              Point(4,4)], index=MultiIndex.from_tuples(index))

        assert_geoseries_equal(expected, s.explode())

        df = self.gdf1[:2].set_geometry(s)
        assert_geoseries_equal(expected, df.explode())

    #
    # Test '&', '|', '^', and '-'
    # The left can only be a GeoSeries. The right hand side can be a
    # GeoSeries, GeoDataFrame or Shapely geometry
    #
    def test_intersection_operator(self):
        self._test_binary_operator('__and__', self.t1, self.g1, self.g2)

    def test_union_operator(self):
        self._test_binary_operator('__or__', self.sq, self.g1, self.g2)

    def test_union_operator_polygon(self):
        self._test_binary_operator('__or__', self.sq, self.g1, self.t2)

    def test_symmetric_difference_operator(self):
        self._test_binary_operator('__xor__', self.sq, self.g3, self.g4)

    def test_difference_series(self):
        expected = GeoSeries([GeometryCollection(), self.t2])
        self._test_binary_operator('__sub__', expected, self.g1, self.g2)

    def test_difference_poly(self):
        expected = GeoSeries([self.t1, self.t1])
        self._test_binary_operator('__sub__', expected, self.g1, self.t2)
