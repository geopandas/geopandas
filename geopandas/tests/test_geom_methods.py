import string
import warnings

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat

import shapely
from shapely import wkt
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union

from geopandas import GeoDataFrame, GeoSeries
from geopandas._compat import GEOS_GE_312, HAS_PYPROJ, SHAPELY_GE_21
from geopandas.base import GeoPandasBase

import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas.tests.util import assert_geoseries_equal, geom_almost_equals, geom_equals
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal


def assert_array_dtype_equal(a, b, *args, **kwargs):
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    assert a.dtype == b.dtype
    assert_array_equal(a, b, *args, **kwargs)


class TestGeomMethods:
    def setup_method(self):
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.t3 = Polygon([(2, 0), (3, 0), (3, 1)])
        self.tz = Polygon([(1, 1, 1), (2, 2, 2), (3, 3, 3)])
        self.tz1 = Polygon([(2, 2, 2), (1, 1, 1), (3, 3, 3)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.sqz = Polygon([(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)])
        self.t4 = Polygon([(0, 0), (3, 0), (3, 3), (0, 2)])
        self.t5 = Polygon([(2, 0), (3, 0), (3, 3), (2, 3)])
        self.t6 = Polygon([(2, 0), (2, 0), (3, 0), (3, 0)])
        self.inner_sq = Polygon(
            [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
        )
        self.nested_squares = Polygon(self.sq.boundary, [self.inner_sq.boundary])
        self.p0 = Point(5, 5)
        self.p3d = Point(5, 5, 5)
        self.g0 = GeoSeries(
            [
                self.t1,
                self.t2,
                self.sq,
                self.inner_sq,
                self.nested_squares,
                self.p0,
                None,
            ]
        )
        self.g1 = GeoSeries([self.t1, self.sq])
        self.g2 = GeoSeries([self.sq, self.t1])
        self.g3 = GeoSeries([self.t1, self.t2])
        self.gz = GeoSeries([self.tz, self.sqz, self.tz1])
        self.g3.crs = "epsg:4326"
        self.g4 = GeoSeries([self.t2, self.t1])
        self.g4.crs = "epsg:4326"
        self.g_3d = GeoSeries([self.p0, self.p3d])
        self.na = GeoSeries([self.t1, self.t2, Polygon()])
        self.na_none = GeoSeries([self.t1, None])
        self.a1 = self.g1.copy()
        self.a1.index = ["A", "B"]
        self.a2 = self.g2.copy()
        self.a2.index = ["B", "C"]
        self.esb = Point(-73.9847, 40.7484, 30.3244)
        self.sol = Point(-74.0446, 40.6893, 31.2344)
        self.landmarks = GeoSeries([self.esb, self.sol], crs="epsg:4326")
        self.pt2d = Point(-73.9847, 40.7484)
        self.landmarks_mixed = GeoSeries([self.esb, self.sol, self.pt2d], crs=4326)
        self.pt_empty = wkt.loads("POINT EMPTY")
        self.landmarks_mixed_empty = GeoSeries(
            [self.esb, self.sol, self.pt2d, self.pt_empty], crs=4326
        )
        self.l1 = LineString([(0, 0), (0, 1), (1, 1)])
        self.l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g5 = GeoSeries([self.l1, self.l2])
        self.g6 = GeoSeries([self.p0, self.t3])
        self.g7 = GeoSeries([self.sq, self.t4])
        self.g8 = GeoSeries([self.t1, self.t5])
        self.empty = GeoSeries([])
        self.all_none = GeoSeries([None, None])
        self.all_geometry_collection_empty = GeoSeries(
            [GeometryCollection([]), GeometryCollection([])]
        )
        self.empty_poly = Polygon()
        self.g9 = GeoSeries(self.g0, index=range(1, 8))
        self.g10 = GeoSeries([self.t1, self.t4])

        # Crossed lines
        self.l3 = LineString([(0, 0), (1, 1)])
        self.l4 = LineString([(0, 1), (1, 0)])
        self.crossed_lines = GeoSeries([self.l3, self.l4])

        # Placeholder for testing, will just drop in different geometries
        # when needed
        self.gdf1 = GeoDataFrame(
            {"geometry": self.g1, "col0": [1.0, 2.0], "col1": ["geo", "pandas"]}
        )
        self.gdf2 = GeoDataFrame(
            {"geometry": self.g1, "col3": [4, 5], "col4": ["rand", "string"]}
        )
        self.gdf3 = GeoDataFrame(
            {"geometry": self.g3, "col3": [4, 5], "col4": ["rand", "string"]}
        )
        self.gdfz = GeoDataFrame(
            {"geometry": self.gz, "col3": [4, 5, 6], "col4": ["rand", "string", "geo"]}
        )

        self.g11 = GeoSeries(
            [
                self.p0,
                self.p3d,
                self.pt_empty,
                self.t1,
                self.tz,
                self.empty_poly,
                self.l1,
            ]
        )
        # expected coordinates from g11
        self.expected_2d = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.expected_3d = np.array(
            [
                [5.0, 5.0, np.nan],
                [5.0, 5.0, 5.0],
                [0.0, 0.0, np.nan],
                [1.0, 0.0, np.nan],
                [1.0, 1.0, np.nan],
                [0.0, 0.0, np.nan],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, np.nan],
                [0.0, 1.0, np.nan],
                [1.0, 1.0, np.nan],
            ]
        )
        self.squares = GeoSeries([self.sq for _ in range(3)])

        self.l5 = LineString([(100, 0), (0, 0), (0, 100)])
        self.l6 = LineString([(5, 5), (5, 100), (100, 5)])
        self.g12 = GeoSeries([self.l5])
        self.g13 = GeoSeries([self.l6])
        self.lines = GeoSeries(
            [
                LineString([(0, 0), (1, 1)]),
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 1), (1, 1)]),
                LineString([(1, 1), (1, 0)]),
                LineString([(1, 0), (0, 0)]),
                LineString([(5, 5), (6, 6)]),
                LineString([(0.5, -1), (0.5, 2)]),
                Point(0, 0),
            ],
            crs=4326,
            index=range(2, 10),
        )

        self.l5 = LineString([(100, 0), (0, 0), (0, 100)])
        self.l6 = LineString([(5, 5), (5, 100), (100, 5)])
        self.g12 = GeoSeries([self.l5])
        self.g13 = GeoSeries([self.l6])
        self.g14 = GeoSeries(
            [
                MultiLineString([[(0, 2), (0, 10)], [(0, 10), (5, 10)]]),
                MultiLineString([[(0, 2), (0, 10)], [(0, 11), (5, 10)]]),
                MultiLineString(),
                MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]),
                Point(0, 0),
            ],
            crs=4326,
            index=range(2, 7),
        )

    def _test_unary_real(self, op, expected, a):
        """Tests for 'area', 'length', 'is_valid', etc."""
        fcmp = assert_series_equal
        self._test_unary(op, expected, a, fcmp)

    def _test_unary_topological(self, op, expected, a, method=False):
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:

            def fcmp(a, b):
                assert a.equals(b)

        self._test_unary(op, expected, a, fcmp, method=method)

    def _test_binary_topological(self, op, expected, a, b, *args, **kwargs):
        """Tests for 'intersection', 'union', 'symmetric_difference', etc."""
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:

            def fcmp(a, b):
                assert geom_equals(a, b)

        if isinstance(b, GeoPandasBase):
            right_df = True
        else:
            right_df = False

        self._binary_op_test(op, expected, a, b, fcmp, True, right_df, *args, **kwargs)

    def _test_binary_real(self, op, expected, a, b, *args, **kwargs):
        fcmp = assert_series_equal
        self._binary_op_test(op, expected, a, b, fcmp, True, False, *args, **kwargs)

    def _binary_op_test(
        self, op, expected, left, right, fcmp, left_df, right_df, *args, **kwargs
    ):
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

            return GeoDataFrame(
                {"geometry": s.values, "col1": col1, "col2": col2},
                index=s.index,
                crs=s.crs,
            )

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

    def _test_unary(self, op, expected, a, fcmp, method=False):
        # GeoSeries, (GeoSeries or geometry)
        if method:
            result = getattr(a, op)()
        else:
            result = getattr(a, op)
        fcmp(result, expected)

        # GeoDataFrame, (GeoSeries or geometry)
        gdf = self.gdf1.set_geometry(a)
        if method:
            result = getattr(gdf, op)()
        else:
            result = getattr(gdf, op)
        fcmp(result, expected)

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_crs_warning(self):
        # operations on geometries should warn for different CRS
        no_crs_g3 = self.g3.copy().set_crs(None, allow_override=True)
        with pytest.warns(UserWarning):
            self._test_binary_topological("intersection", self.g3, self.g3, no_crs_g3)

    def test_alignment_warning(self):
        with pytest.warns(
            UserWarning,
            match="The indices of the left and right GeoSeries' are not equal",
        ):
            self.g0.intersection(self.g9, align=None)

        with warnings.catch_warnings(record=True) as record:
            self.g0.intersection(self.g9, align=True)
            self.g0.intersection(self.g9, align=False)

            assert len(record) == 0

    def test_intersection(self):
        self._test_binary_topological("intersection", self.t1, self.g1, self.g2)
        self._test_binary_topological(
            "intersection", self.all_none, self.g1, self.empty, align=True
        )

        assert len(self.g0.intersection(self.g9, align=True) == 8)
        assert len(self.g0.intersection(self.g9, align=False) == 7)

    def test_clip_by_rect(self):
        self._test_binary_topological(
            "clip_by_rect", self.g1, self.g10, *self.sq.bounds
        )
        # self.g1 and self.t3.bounds do not intersect
        self._test_binary_topological(
            "clip_by_rect", self.all_geometry_collection_empty, self.g1, *self.t3.bounds
        )

    def test_union_series(self):
        self._test_binary_topological("union", self.sq, self.g1, self.g2)

        assert len(self.g0.union(self.g9, align=True) == 8)
        assert len(self.g0.union(self.g9, align=False) == 7)

    def test_union_polygon(self):
        self._test_binary_topological("union", self.sq, self.g1, self.t2)

    def test_symmetric_difference_series(self):
        self._test_binary_topological("symmetric_difference", self.sq, self.g3, self.g4)

        assert len(self.g0.symmetric_difference(self.g9, align=True) == 8)
        assert len(self.g0.symmetric_difference(self.g9, align=False) == 7)

    def test_symmetric_difference_poly(self):
        expected = GeoSeries([GeometryCollection(), self.sq], crs=self.g3.crs)
        self._test_binary_topological(
            "symmetric_difference", expected, self.g3, self.t1
        )

    def test_difference_series(self):
        expected = GeoSeries([GeometryCollection(), self.t2])
        self._test_binary_topological("difference", expected, self.g1, self.g2)

        assert len(self.g0.difference(self.g9, align=True) == 8)
        assert len(self.g0.difference(self.g9, align=False) == 7)

    def test_difference_poly(self):
        expected = GeoSeries([self.t1, self.t1])
        self._test_binary_topological("difference", expected, self.g1, self.t2)

    def test_shortest_line(self):
        expected = GeoSeries([LineString([(1, 1), (5, 5)]), None])
        assert_array_dtype_equal(expected, self.na_none.shortest_line(self.p0))

        expected = GeoSeries(
            [
                LineString([(5, 5), (1, 1)]),
                LineString([(2, 0), (2, 0)]),
            ]
        )
        assert_array_dtype_equal(expected, self.g6.shortest_line(self.g7))

        expected = GeoSeries(
            [LineString([(0.5, 0.5), (0.5, 0.5)]), LineString([(0.5, 0.5), (0.5, 0.5)])]
        )
        crossed_lines_inv = self.crossed_lines[::-1]
        assert_array_dtype_equal(
            expected, self.crossed_lines.shortest_line(crossed_lines_inv, align=False)
        )

    def test_snap(self):
        expected = GeoSeries([Polygon([(0, 0.5), (1, 0), (1, 1), (0, 0.5)]), None])
        assert_array_dtype_equal(
            expected, self.na_none.snap(Point(0, 0.5), tolerance=1)
        )

        expected = GeoSeries(
            [
                Point((5, 5)),
                Polygon([(0, 2), (0, 0), (3, 0), (3, 3), (0, 2)]),
            ]
        )
        assert_array_dtype_equal(expected, self.g6.snap(self.g7, tolerance=3))

    def test_geo_op_empty_result(self):
        l1 = LineString([(0, 0), (1, 1)])
        l2 = LineString([(2, 2), (3, 3)])
        expected = GeoSeries([GeometryCollection()])
        # binary geo resulting in empty geometry
        result = GeoSeries([l1]).intersection(l2)
        assert_geoseries_equal(result, expected)
        # binary geo empty result with right GeoSeries
        result = GeoSeries([l1]).intersection(GeoSeries([l2]))
        assert_geoseries_equal(result, expected)
        # unary geo resulting in empty geometry
        result = GeoSeries([GeometryCollection()]).convex_hull
        assert_geoseries_equal(result, expected)

    def test_boundary(self):
        l1 = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
        l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        expected = GeoSeries([l1, l2], index=self.g1.index, crs=self.g1.crs)

        self._test_unary_topological("boundary", expected, self.g1)

    def test_area(self):
        expected = Series(np.array([0.5, 1.0]), index=self.g1.index)
        self._test_unary_real("area", expected, self.g1)

        expected = Series(np.array([0.5, np.nan]), index=self.na_none.index)
        self._test_unary_real("area", expected, self.na_none)

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_area_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.area

    def test_bounds(self):
        # Set columns to get the order right
        expected = DataFrame(
            {
                "minx": [0.0, 0.0],
                "miny": [0.0, 0.0],
                "maxx": [1.0, 1.0],
                "maxy": [1.0, 1.0],
            },
            index=self.g1.index,
            columns=["minx", "miny", "maxx", "maxy"],
        )

        result = self.g1.bounds
        assert_frame_equal(expected, result)

        gdf = self.gdf1.set_geometry(self.g1)
        result = gdf.bounds
        assert_frame_equal(expected, result)

    def test_bounds_empty(self):
        # test bounds of empty GeoSeries
        # https://github.com/geopandas/geopandas/issues/1195
        s = GeoSeries([])
        result = s.bounds
        expected = DataFrame(
            columns=["minx", "miny", "maxx", "maxy"], index=s.index, dtype="float64"
        )
        assert_frame_equal(result, expected)

    def test_union_all(self):
        p1 = self.t1
        p2 = Polygon([(2, 0), (3, 0), (3, 1)])
        expected = unary_union([p1, p2])
        g = GeoSeries([p1, p2])

        self._test_unary_topological("union_all", expected, g, method=True)

        g2 = GeoSeries([p1, None])
        self._test_unary_topological("union_all", p1, g2, method=True)

        g3 = GeoSeries([None, None])
        assert g3.union_all().equals(shapely.GeometryCollection())

        assert g.union_all(method="coverage").equals(expected)
        if GEOS_GE_312 and SHAPELY_GE_21:
            assert g.union_all(method="disjoint_subset").equals(expected)

    def test_unary_union_deprecated(self):
        p1 = self.t1
        p2 = Polygon([(2, 0), (3, 0), (3, 1)])
        g = GeoSeries([p1, p2])
        with pytest.warns(
            DeprecationWarning, match="The 'unary_union' attribute is deprecated"
        ):
            result = g.unary_union
        assert result == g.union_all()

    def test_intersection_all(self):
        expected = Polygon([(1, 1), (1, 1.5), (1.5, 1.5), (1.5, 1), (1, 1)])
        g = GeoSeries([box(0, 0, 2, 2), box(1, 1, 3, 3), box(0, 0, 1.5, 1.5)])

        assert g.intersection_all().equals(expected)

        g2 = GeoSeries([box(0, 0, 2, 2), None])
        assert g2.intersection_all().equals(g2[0])

        g3 = GeoSeries([None, None])
        assert g3.intersection_all().equals(shapely.GeometryCollection())

    def test_contains(self):
        expected = [True, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.contains(self.t1))

        expected = [False, True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.contains(self.g9, align=True))

        expected = [False, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.contains(self.g9, align=False))

    def test_contains_properly(self):
        expected = [False, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.contains_properly(Point(0.25, 0.25)))

        expected = [False, False, False, False, False, True, False, False]
        assert_array_dtype_equal(
            expected, self.g0.contains_properly(self.g9, align=True)
        )

        expected = [False, False, True, False, False, False, False]
        assert_array_dtype_equal(
            expected, self.g0.contains_properly(self.g9, align=False)
        )

    @pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="requires GEOS>=3.10")
    def test_dwithin(self):
        expected = [True, True, True, False, True, True, False]
        assert_array_dtype_equal(expected, self.g0.dwithin(self.p0, 6))

        expected = [False, True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.dwithin(self.g9, 1, align=True))
        expected = [True, True, True, True, False, False, False]
        assert_array_dtype_equal(expected, self.g0.dwithin(self.g9, 1, align=False))

    def test_length(self):
        expected = Series(np.array([2 + np.sqrt(2), 4]), index=self.g1.index)
        self._test_unary_real("length", expected, self.g1)

        expected = Series(np.array([2 + np.sqrt(2), np.nan]), index=self.na_none.index)
        self._test_unary_real("length", expected, self.na_none)

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_length_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.length

    def test_count_coordinates(self):
        expected = Series(np.array([4, 5]), index=self.g1.index)
        assert_series_equal(self.g1.count_coordinates(), expected, check_dtype=False)

        expected = Series(np.array([4, 0]), index=self.na_none.index)
        assert_series_equal(
            self.na_none.count_coordinates(), expected, check_dtype=False
        )

    def test_count_geometries(self):
        expected = Series(np.array([4, 2, 1, 1, 0]))
        s = GeoSeries(
            [
                MultiPoint([(0, 0), (1, 1), (1, -1), (0, 1)]),
                MultiLineString([((0, 0), (1, 1)), ((-1, 0), (1, 0))]),
                LineString([(0, 0), (1, 1), (1, -1)]),
                Point(0, 0),
                None,
            ]
        )
        assert_series_equal(s.count_geometries(), expected, check_dtype=False)

    def test_count_interior_rings(self):
        expected = Series(np.array([1, 2, 0, 0]))
        s = GeoSeries(
            [
                Polygon(
                    [(0, 0), (0, 5), (5, 5), (5, 0)],
                    [[(1, 1), (1, 4), (4, 4), (4, 1)]],
                ),
                Polygon(
                    [(0, 0), (0, 5), (5, 5), (5, 0)],
                    [
                        [(1, 1), (1, 2), (2, 2), (2, 1)],
                        [(3, 2), (3, 3), (4, 3), (4, 2)],
                    ],
                ),
                Point(0, 1),
                None,
            ]
        )
        assert_series_equal(s.count_interior_rings(), expected, check_dtype=False)

    def test_crosses(self):
        expected = [False, False, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.crosses(self.t1))

        expected = [False, True]
        assert_array_dtype_equal(expected, self.crossed_lines.crosses(self.l3))

        expected = [False] * 8
        assert_array_dtype_equal(expected, self.g0.crosses(self.g9, align=True))

        expected = [False] * 7
        assert_array_dtype_equal(expected, self.g0.crosses(self.g9, align=False))

    def test_disjoint(self):
        expected = [False, False, False, False, False, True, False]
        assert_array_dtype_equal(expected, self.g0.disjoint(self.t1))

        expected = [False] * 8
        assert_array_dtype_equal(expected, self.g0.disjoint(self.g9, align=True))

        expected = [False, False, False, False, True, False, False]
        assert_array_dtype_equal(expected, self.g0.disjoint(self.g9, align=False))

    def test_relate(self):
        expected = Series(
            [
                "212101212",
                "212101212",
                "212FF1FF2",
                "2FFF1FFF2",
                "FF2F112F2",
                "FF0FFF212",
                None,
            ],
            index=self.g0.index,
        )
        assert_series_equal(expected, self.g0.relate(self.inner_sq))

        expected = Series(["FF0FFF212", None], index=self.g6.index)
        assert_series_equal(expected, self.g6.relate(self.na_none))

        expected = Series(
            [
                None,
                "2FFF1FFF2",
                "2FFF1FFF2",
                "2FFF1FFF2",
                "2FFF1FFF2",
                "0FFFFFFF2",
                None,
                None,
            ],
            index=range(8),
        )

        assert_series_equal(expected, self.g0.relate(self.g9, align=True))

        expected = Series(
            [
                "FF2F11212",
                "2FF11F212",
                "212FF1FF2",
                "FF2F1F212",
                "FF2FF10F2",
                None,
                None,
            ],
            index=self.g0.index,
        )
        assert_series_equal(expected, self.g0.relate(self.g9, align=False))

    def test_relate_pattern(self):
        expected = Series([True] * 4 + [False] * 3, index=self.g0.index, dtype=bool)
        assert_array_dtype_equal(
            expected, self.g0.relate_pattern(self.inner_sq, "2********")
        )

        expected = Series([True, False], index=self.g6.index, dtype=bool)
        assert_array_dtype_equal(
            expected, self.g6.relate_pattern(self.na_none, "FF0******")
        )

        expected = Series(
            [False] + [True] * 5 + [False, False], index=range(8), dtype=bool
        )
        with pytest.warns(UserWarning, match="The indices of the left and right"):
            assert_array_dtype_equal(
                expected, self.g0.relate_pattern(self.g9, "T********", align=None)
            )
        expected = Series(
            [False] + [True] * 2 + [False] * 4, index=self.g0.index, dtype=bool
        )
        assert_array_dtype_equal(
            expected, self.g0.relate_pattern(self.g9, "T********", align=False)
        )

    def test_distance(self):
        expected = Series(
            np.array([np.sqrt((5 - 1) ** 2 + (5 - 1) ** 2), np.nan]), self.na_none.index
        )
        assert_array_dtype_equal(expected, self.na_none.distance(self.p0))

        expected = Series(np.array([np.sqrt(4**2 + 4**2), np.nan]), self.g6.index)
        assert_array_dtype_equal(expected, self.g6.distance(self.na_none))

        expected = Series(np.array([np.nan, 0, 0, 0, 0, 0, np.nan, np.nan]), range(8))
        assert_array_dtype_equal(expected, self.g0.distance(self.g9, align=True))

        val = self.g0.iloc[4].distance(self.g9.iloc[4])
        expected = Series(np.array([0, 0, 0, 0, val, np.nan, np.nan]), self.g0.index)
        assert_array_dtype_equal(expected, self.g0.distance(self.g9, align=False))

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_distance_crs_warning(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.distance(self.p0)

    def test_hausdorff_distance(self):
        # closest point is (0, 0) in self.p1
        expected = Series(np.array([np.sqrt(5**2 + 5**2), np.nan]), self.na_none.index)
        assert_array_dtype_equal(expected, self.na_none.hausdorff_distance(self.p0))

        expected = Series(np.array([np.sqrt(5**2 + 5**2), np.nan]), self.na_none.index)
        assert_array_dtype_equal(expected, self.na_none.hausdorff_distance(self.p0))

        expected = Series(np.array([np.nan, 0, 0, 0, 0, 0, np.nan, np.nan]), range(8))
        assert_array_dtype_equal(
            expected, self.g0.hausdorff_distance(self.g9, align=True)
        )

        val_1 = self.g0.iloc[0].hausdorff_distance(self.g9.iloc[0])
        val_2 = self.g0.iloc[2].hausdorff_distance(self.g9.iloc[2])
        val_3 = self.g0.iloc[4].hausdorff_distance(self.g9.iloc[4])
        expected = Series(
            np.array([val_1, val_1, val_2, val_2, val_3, np.nan, np.nan]), self.g0.index
        )
        assert_array_dtype_equal(
            expected, self.g0.hausdorff_distance(self.g9, align=False)
        )

        expected = Series(np.array([52.5]), self.g12.index)
        assert_array_dtype_equal(
            expected, self.g12.hausdorff_distance(self.g13, densify=0.25)
        )

    @pytest.mark.skipif(
        shapely.geos_version < (3, 10, 0), reason="buggy with GEOS<3.10"
    )
    def test_frechet_distance(self):
        # closest point is (0, 0) in self.p1
        expected = Series(np.array([np.sqrt(5**2 + 5**2), np.nan]), self.na_none.index)
        assert_array_dtype_equal(expected, self.na_none.frechet_distance(self.p0))

        expected = Series(np.array([np.nan, 0, 0, 0, 0, 0, np.nan, np.nan]), range(8))
        assert_array_dtype_equal(
            expected, self.g0.frechet_distance(self.g9, align=True)
        )

        # expected returns
        val_1 = 1.0
        val_2 = np.sqrt(2) / 4
        val_3 = np.sqrt(2) / 2
        val_4 = (np.sqrt(2) / 2) * 10
        expected = Series(
            np.array([val_1, val_1, val_2, val_3, val_4, np.nan, np.nan]), self.g0.index
        )
        assert_array_dtype_equal(
            expected, self.g0.frechet_distance(self.g9, align=False)
        )

        expected = Series(np.array([np.sqrt(100**2 + (100 - 5) ** 2)]), self.g12.index)
        assert_array_dtype_equal(
            expected, self.g12.frechet_distance(self.g13, densify=0.25)
        )

    def test_intersects(self):
        expected = [True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.intersects(self.t1))

        expected = [True, False]
        assert_array_dtype_equal(expected, self.na_none.intersects(self.t2))

        expected = np.array([], dtype=bool)
        assert_array_dtype_equal(expected, self.empty.intersects(self.t1))

        expected = np.array([], dtype=bool)
        assert_array_dtype_equal(expected, self.empty.intersects(self.empty_poly))

        expected = [False] * 7
        assert_array_dtype_equal(expected, self.g0.intersects(self.empty_poly))

        expected = [False, True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.intersects(self.g9, align=True))

        expected = [True, True, True, True, False, False, False]
        assert_array_dtype_equal(expected, self.g0.intersects(self.g9, align=False))

    def test_overlaps(self):
        expected = [True, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.overlaps(self.inner_sq))

        expected = [False, False]
        assert_array_dtype_equal(expected, self.g4.overlaps(self.t1))

        expected = [False] * 8
        assert_array_dtype_equal(expected, self.g0.overlaps(self.g9, align=True))

        expected = [False] * 7
        assert_array_dtype_equal(expected, self.g0.overlaps(self.g9, align=False))

    def test_touches(self):
        expected = [False, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.touches(self.t1))

        expected = [False] * 8
        assert_array_dtype_equal(expected, self.g0.touches(self.g9, align=True))

        expected = [True, False, False, True, False, False, False]
        assert_array_dtype_equal(expected, self.g0.touches(self.g9, align=False))

    def test_within(self):
        expected = [True, False, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.within(self.t1))

        expected = [True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.within(self.sq))

        expected = [False, True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.within(self.g9, align=True))

        expected = [False, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.within(self.g9, align=False))

    def test_covers_itself(self):
        # Each polygon in a Series covers itself
        res = self.g1.covers(self.g1)
        exp = Series([True, True])
        assert_series_equal(res, exp)

    def test_covers(self):
        res = self.g7.covers(self.g8)
        exp = Series([True, False])
        assert_series_equal(res, exp)

        expected = [False, True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.covers(self.g9, align=True))

        expected = [False, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.covers(self.g9, align=False))

    def test_covers_inverse(self):
        res = self.g8.covers(self.g7)
        exp = Series([False, False])
        assert_series_equal(res, exp)

    def test_covered_by(self):
        res = self.g1.covered_by(self.g1)
        exp = Series([True, True])
        assert_series_equal(res, exp)

        expected = [False, True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.covered_by(self.g9, align=True))

        expected = [False, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.covered_by(self.g9, align=False))

    def test_is_valid(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_valid", expected, self.g1)

    def test_is_valid_reason(self):
        expected = Series(np.array(["Valid Geometry"] * len(self.g1)), self.g1.index)
        assert_series_equal(self.g1.is_valid_reason(), expected)

        s = GeoSeries(
            [
                Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]),  # bowtie geometry
                Polygon([(0, 0), (1, 1), (1, 1), (0, 1)]),
                None,
            ]
        )
        expected = Series(["Self-intersection[0.5 0.5]", "Valid Geometry", None])
        assert_series_equal(s.is_valid_reason(), expected)

    @pytest.mark.skipif(
        not (GEOS_GE_312 and SHAPELY_GE_21), reason="GEOS 3.12 and shapely 2.1 needed."
    )
    def test_is_valid_coverage(self):
        s = GeoSeries(
            [
                Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
                Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
            ]
        )
        assert s.is_valid_coverage()

        s2 = GeoSeries(
            [
                Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
                Polygon([(0, 0), (0.5, 0.5), (1, 1), (0, 1), (0, 0)]),
            ]
        )
        assert not s2.is_valid_coverage()

    @pytest.mark.skipif(
        not (GEOS_GE_312 and SHAPELY_GE_21), reason="GEOS 3.12 and shapely 2.1 needed."
    )
    def test_invalid_coverage_edges(self):
        s = GeoSeries(
            [
                Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
                Polygon([(0, 0), (0.5, 0.5), (1, 1), (0, 1), (0, 0)]),
            ]
        )
        expected = GeoSeries(
            [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0.5, 0.5), (1, 1)])]
        )
        assert_geoseries_equal(s.invalid_coverage_edges(), expected)

    def test_is_empty(self):
        expected = Series(np.array([False] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_empty", expected, self.g1)

    def test_is_ring(self):
        expected = Series(np.array([False] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_ring", expected, self.g1)
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_ring", expected, self.g1.exterior)

    def test_is_simple(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_simple", expected, self.g1)

    def test_is_ccw(self):
        expected = Series(np.array([False] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_ccw", expected, self.g1)

    def test_is_closed(self):
        expected = Series(np.array([False, False]), self.g5.index)
        self._test_unary_real("is_closed", expected, self.g5)

    def test_has_z(self):
        expected = Series([False, True], self.g_3d.index)
        self._test_unary_real("has_z", expected, self.g_3d)

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires shapely 2.1")
    @pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="requires GEOS>=3.12")
    def test_has_m(self):
        s = GeoSeries.from_wkt(
            [
                "POINT M (2 3 5)",
                "POINT Z (1 2 3)",
            ],
        )
        expected = Series([True, False])
        self._test_unary_real("has_m", expected, s)

    def test_xyz_points(self):
        expected_x = [-73.9847, -74.0446]
        expected_y = [40.7484, 40.6893]
        expected_z = [30.3244, 31.2344]

        assert_array_dtype_equal(expected_x, self.landmarks.geometry.x)
        assert_array_dtype_equal(expected_y, self.landmarks.geometry.y)
        assert_array_dtype_equal(expected_z, self.landmarks.geometry.z)

        # mixed dimensions
        expected_z = [30.3244, 31.2344, np.nan]
        assert_array_dtype_equal(expected_z, self.landmarks_mixed.geometry.z)

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires shapely 2.1")
    def test_m_points(self):
        s = GeoSeries.from_wkt(
            [
                "POINT M (2 3 5)",
                "POINT M (1 2 3)",
                "POINT (0 0)",
            ]
        )

        expected = [5, 3, np.nan]
        assert_array_dtype_equal(expected, s.m)

    def test_xyz_points_empty(self):
        expected_x = [-73.9847, -74.0446, -73.9847, np.nan]
        expected_y = [40.7484, 40.6893, 40.7484, np.nan]
        expected_z = [30.3244, 31.2344, np.nan, np.nan]

        assert_array_dtype_equal(expected_x, self.landmarks_mixed_empty.geometry.x)
        assert_array_dtype_equal(expected_y, self.landmarks_mixed_empty.geometry.y)
        assert_array_dtype_equal(expected_z, self.landmarks_mixed_empty.geometry.z)

    def test_xyz_polygons(self):
        # accessing x attribute in polygon geoseries should raise an error
        with pytest.raises(ValueError):
            _ = self.gdf1.geometry.x
        # and same for accessing y attribute in polygon geoseries
        with pytest.raises(ValueError):
            _ = self.gdf1.geometry.y
        # and same for accessing z attribute in polygon geoseries
        with pytest.raises(ValueError):
            _ = self.gdfz.geometry.z

    def test_centroid(self):
        polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        point = Point(0, 0)
        polygons = GeoSeries([polygon for i in range(3)])
        points = GeoSeries([point for i in range(3)])
        assert_geoseries_equal(polygons.centroid, points)

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_centroid_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.centroid

    def test_normalize(self):
        polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        linestring = LineString([(0, 0), (1, 1), (1, 0)])
        point = Point(0, 0)
        series = GeoSeries([polygon, linestring, point])
        polygon2 = Polygon([(0, 0), (0, 1), (1, 1)])
        expected = GeoSeries([polygon2, linestring, point])
        assert_geoseries_equal(series.normalize(), expected)

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires Shapely>=2.1")
    def test_orient_polygons(self):
        polygon = Polygon(
            [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
            holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
        )
        linestring = LineString([(0, 0), (1, 1), (1, 0)])
        point = Point(0, 0)
        series = GeoSeries([polygon, linestring, point])

        polygon2 = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
        )
        expected = GeoSeries([polygon2, linestring, point])
        assert_geoseries_equal(series.orient_polygons(), expected)

        polygon_cw = Polygon(
            [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
            holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
        )
        expected = GeoSeries([polygon_cw, linestring, point])
        assert_geoseries_equal(series.orient_polygons(exterior_cw=True), expected)

    def test_make_valid(self):
        polygon1 = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
        polygon2 = Polygon([(0, 2), (0, 1), (2, 0), (0, 0), (0, 2)])
        linestring = LineString([(0, 0), (1, 1), (1, 0)])
        series = GeoSeries([polygon1, polygon2, linestring])
        out_polygon1 = MultiPolygon(
            [
                Polygon([(1, 1), (0, 0), (0, 2), (1, 1)]),
                Polygon([(2, 0), (1, 1), (2, 2), (2, 0)]),
            ]
        )
        out_polygon2 = GeometryCollection(
            [Polygon([(2, 0), (0, 0), (0, 1), (2, 0)]), LineString([(0, 2), (0, 1)])]
        )
        expected = GeoSeries([out_polygon1, out_polygon2, linestring])
        assert not series.is_valid.all()
        result = series.make_valid()
        assert_geoseries_equal(result, expected)
        assert result.is_valid.all()

    @pytest.mark.parametrize(
        "method, keep_collapsed, expected",
        [
            (
                "linework",
                True,
                MultiLineString([[(0, 0), (1, 1)], [(1, 1), (1, 2)]]),
            ),
            (
                "structure",
                True,
                LineString([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
            ),
            ("structure", False, Polygon()),
        ],
    )
    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires Shapely>=2.1")
    def test_make_valid_method(self, method, keep_collapsed, expected):
        polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
        series = GeoSeries([polygon])
        expected = GeoSeries([expected])
        assert not series.is_valid.all()
        result = series.make_valid(method=method, keep_collapsed=keep_collapsed)
        assert_geoseries_equal(result, expected, check_geom_type=True)
        assert result.is_valid.all()

    @pytest.mark.skipif(SHAPELY_GE_21, reason="test for Shapely<2.1")
    def test_make_valid_old_shapely(self):
        """Only the 'linework' method is supported for shapely < 2.1."""
        polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
        series = GeoSeries([polygon])
        with pytest.raises(
            ValueError, match="Only the 'linework' method is supported for"
        ):
            series.make_valid(method="structure")

    def test_reverse(self):
        expected = GeoSeries(
            [
                LineString([(0, 0), (0, 1), (1, 1)]),
                LineString([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ]
        )
        assert_geoseries_equal(expected, self.g5.reverse())

    @pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="requires GEOS>=3.10")
    def test_segmentize_linestrings(self):
        expected_g1 = GeoSeries(
            [
                Polygon(
                    (
                        (0, 0),
                        (0.5, 0),
                        (1, 0),
                        (1, 0.5),
                        (1, 1),
                        (0.6666666666666666, 0.6666666666666666),
                        (0.3333333333333333, 0.3333333333333333),
                        (0, 0),
                    )
                ),
                Polygon(
                    (
                        (0, 0),
                        (0.5, 0),
                        (1, 0),
                        (1, 0.5),
                        (1, 1),
                        (0.5, 1),
                        (0, 1),
                        (0, 0.5),
                        (0, 0),
                    )
                ),
            ]
        )
        expected_g5 = GeoSeries(
            [
                LineString([(0, 0), (0, 0.5), (0, 1), (0.5, 1), (1, 1)]),
                LineString(
                    [(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1)]
                ),
            ]
        )
        result_g1 = self.g1.segmentize(max_segment_length=0.5)
        result_g5 = self.g5.segmentize(max_segment_length=0.5)
        assert_geoseries_equal(expected_g1, result_g1)
        assert_geoseries_equal(expected_g5, result_g5)

    def test_segmentize_wrong_index(self):
        with pytest.raises(
            ValueError,
            match="Index of the Series passed as 'max_segment_length' does not match",
        ):
            self.g1.segmentize(max_segment_length=Series([0.5, 0.5], index=[99, 98]))

    def test_transform(self):
        # Test 2D
        test_2d = GeoSeries(
            [LineString([(2, 2), (4, 4)]), Polygon([(0, 0), (1, 1), (0, 1)])]
        )
        expected_2d = GeoSeries(
            [LineString([(4, 6), (8, 12)]), Polygon([(0, 0), (2, 3), (0, 3)])]
        )
        result_2d = test_2d.transform(lambda x: x * [2, 3])
        assert_geoseries_equal(expected_2d, result_2d)
        # Test 3D
        test_3d = GeoSeries(
            [
                Point(0, 0, 0),
                LineString([(2, 2, 2), (4, 4, 4)]),
                Polygon([(0, 0, 0), (1, 1, 1), (0, 1, 0.5)]),
            ]
        )
        expected_3d = GeoSeries(
            [
                Point(1, 1, 1),
                LineString([(3, 3, 3), (5, 5, 5)]),
                Polygon([(1, 1, 1), (2, 2, 2), (1, 2, 1.5)]),
            ]
        )
        result_3d = test_3d.transform(lambda x: x + 1, include_z=True)
        assert_geoseries_equal(expected_3d, result_3d)
        # Test 3D as 2D transformation
        expected_3d_to_2d = GeoSeries(
            [
                Point(1, 1),
                LineString([(3, 3), (5, 5)]),
                Polygon([(1, 1), (2, 2), (1, 2)]),
            ]
        )
        result_3d_to_2d = test_3d.transform(lambda x: x + 1, include_z=False)
        assert_geoseries_equal(expected_3d_to_2d, result_3d_to_2d)

    @pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="requires GEOS>=3.11")
    def test_concave_hull(self):
        assert_geoseries_equal(self.squares, self.squares.concave_hull())

    @pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="requires GEOS>=3.11")
    @pytest.mark.parametrize(
        "expected_series,ratio",
        [
            ([(0, 0), (0, 3), (1, 1), (3, 3), (3, 0), (0, 0)], 0.0),
            ([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)], 1.0),
        ],
    )
    def test_concave_hull_accepts_kwargs(self, expected_series, ratio):
        expected = GeoSeries(Polygon(expected_series))
        s = GeoSeries(MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)]))
        assert_geoseries_equal(expected, s.concave_hull(ratio=ratio))

    def test_concave_hull_wrong_index(self):
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'ratio' does not match"
        ):
            self.g1.concave_hull(ratio=Series([0.0, 1.0], index=[99, 98]))

        with pytest.raises(
            ValueError,
            match="Index of the Series passed as 'allow_holes' does not match",
        ):
            self.g1.concave_hull(
                ratio=0.1, allow_holes=Series([True, False], index=[99, 98])
            )

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires shapely 2.1")
    def test_constrained_delaunay_triangles(self):
        input = GeoSeries([Polygon([(0, 0), (1, 1), (0, 1)])])
        expected = GeoSeries(
            [GeometryCollection([Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])])]
        )
        assert_geoseries_equal(
            input.constrained_delaunay_triangles(), expected, check_geom_type=True
        )

    def test_convex_hull(self):
        # the convex hull of a square should be the same as the square
        assert_geoseries_equal(self.squares, self.squares.convex_hull)

    def test_delaunay_triangles(self):
        expected = GeoSeries(
            [
                Polygon([(0, 1), (0, 0), (1, 0), (0, 1)]),
                Polygon([(0, 1), (1, 0), (1, 1), (0, 1)]),
            ]
        )
        dlt = self.g5.delaunay_triangles()
        assert_geoseries_equal(expected, dlt)

    def test_delaunay_triangles_pass_kwargs(self):
        expected = GeoSeries(
            [
                LineString([(0, 1), (1, 1)]),
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)]),
                LineString([(0, 1), (1, 0)]),
            ],
        )
        dlt = self.g5.delaunay_triangles(only_edges=True)
        assert_geoseries_equal(expected, dlt)

    def test_voronoi_polygons(self):
        expected = GeoSeries.from_wkt(
            [
                "POLYGON ((2 2, 2 0.5, 0.5 0.5, 0.5 2, 2 2))",
                "POLYGON ((-1 2, 0.5 2, 0.5 0.5, -1 0.5, -1 2))",
                "POLYGON ((-1 -1, -1 0.5, 0.5 0.5, 0.5 -1, -1 -1))",
                "POLYGON ((2 -1, 0.5 -1, 0.5 0.5, 2 0.5, 2 -1))",
            ],
            crs=self.g1.crs,
        )
        vp = self.g1.voronoi_polygons()
        assert_geoseries_equal(expected, vp)

    def test_voronoi_polygons_only_edges(self):
        expected = GeoSeries.from_wkt(
            [
                "LINESTRING (0.5 0.5, 0.5 2)",
                "LINESTRING (2 0.5, 0.5 0.5)",
                "LINESTRING (0.5 0.5, -1 0.5)",
                "LINESTRING (0.5 0.5, 0.5 -1)",
            ],
            crs=self.g1.crs,
        )
        vp = self.g1.voronoi_polygons(only_edges=True)
        assert_geoseries_equal(expected, vp, check_less_precise=True)

    def test_voronoi_polygons_extend_to(self):
        expected = GeoSeries.from_wkt(
            [
                "POLYGON ((3 3, 3 0.5, 0.5 0.5, 0.5 3, 3 3))",
                "POLYGON ((-2 3, 0.5 3, 0.5 0.5, -2 0.5, -2 3))",
                "POLYGON ((-2 -1, -2 0.5, 0.5 0.5, 0.5 -1, -2 -1))",
                "POLYGON ((3 -1, 0.5 -1, 0.5 0.5, 3 0.5, 3 -1))",
            ],
            crs=self.g1.crs,
        )
        vp = self.g1.voronoi_polygons(extend_to=box(-2, 0, 3, 3))
        assert_geoseries_equal(expected, vp)

    def test_exterior(self):
        exp_exterior = GeoSeries([LinearRing(p.boundary) for p in self.g3])
        for expected, computed in zip(exp_exterior, self.g3.exterior):
            assert computed.equals(expected)

    def test_interiors(self):
        original = GeoSeries([self.t1, self.nested_squares])

        # This is a polygon with no interior.
        expected = []
        assert original.interiors[0] == expected
        # This is a polygon with an interior.
        expected = LinearRing(self.inner_sq.boundary)
        assert original.interiors[1][0].equals(expected)

        no_interiors = GeoSeries([self.t1, self.sq])
        assert no_interiors.interiors[0] == []
        assert no_interiors.interiors[1] == []

    def test_interpolate(self):
        expected = GeoSeries([Point(0.5, 1.0), Point(0.75, 1.0)])
        self._test_binary_topological(
            "interpolate", expected, self.g5, 0.75, normalized=True
        )

        expected = GeoSeries([Point(0.5, 1.0), Point(1.0, 0.5)])
        self._test_binary_topological("interpolate", expected, self.g5, 1.5)

    def test_interpolate_distance_array(self):
        expected = GeoSeries([Point(0.0, 0.75), Point(1.0, 0.5)])
        self._test_binary_topological(
            "interpolate", expected, self.g5, np.array([0.75, 1.5])
        )

        expected = GeoSeries([Point(0.5, 1.0), Point(0.0, 1.0)])
        self._test_binary_topological(
            "interpolate", expected, self.g5, np.array([0.75, 1.5]), normalized=True
        )

    def test_interpolate_distance_wrong_length(self):
        distances = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            self.g5.interpolate(distances)

    def test_interpolate_distance_wrong_index(self):
        distances = Series([1, 2], index=[99, 98])
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'distance' does not match"
        ):
            self.g5.interpolate(distances)

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_interpolate_crs_warning(self):
        g5_crs = self.g5.copy()
        g5_crs.crs = 4326
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            g5_crs.interpolate(1)

    def test_project(self):
        expected = Series([2.0, 1.5], index=self.g5.index)
        p = Point(1.0, 0.5)
        self._test_binary_real("project", expected, self.g5, p)

        expected = Series([1.0, 0.5], index=self.g5.index)
        self._test_binary_real("project", expected, self.g5, p, normalized=True)

        s = GeoSeries([Point(2, 2), Point(0.5, 0.5)], index=[1, 2])
        expected = Series([np.nan, 2.0, np.nan])
        assert_series_equal(self.g5.project(s, align=True), expected)

        expected = Series([2.0, 0.5], index=self.g5.index)
        assert_series_equal(self.g5.project(s, align=False), expected)

    def test_affine_transform(self):
        # 45 degree reflection matrix
        matrix = [0, 1, 1, 0, 0, 0]
        expected = self.g4

        res = self.g3.affine_transform(matrix)
        assert_geoseries_equal(expected, res)

    def test_translate_tuple(self):
        trans = self.sol.x - self.esb.x, self.sol.y - self.esb.y
        assert self.landmarks.translate(*trans)[0].equals(self.sol)

        res = self.gdf1.set_geometry(self.landmarks).translate(*trans)[0]
        assert res.equals(self.sol)

    def test_rotate(self):
        angle = 98
        expected = self.g4

        o = Point(0, 0)
        res = self.g4.rotate(angle, origin=o).rotate(-angle, origin=o)
        assert geom_almost_equals(self.g4, res)

        res = self.gdf1.set_geometry(self.g4).rotate(angle, origin=Point(0, 0))
        assert geom_almost_equals(expected, res.rotate(-angle, origin=o))

    def test_scale(self):
        expected = self.g4

        scale = 2.0, 1.0
        inv = tuple(1.0 / i for i in scale)

        o = Point(0, 0)
        res = self.g4.scale(*scale, origin=o).scale(*inv, origin=o)
        assert geom_almost_equals(expected, res)

        res = self.gdf1.set_geometry(self.g4).scale(*scale, origin=o)
        res = res.scale(*inv, origin=o)
        assert geom_almost_equals(expected, res)

    def test_skew(self):
        expected = self.g4

        skew = 45.0
        o = Point(0, 0)

        # Test xs
        res = self.g4.skew(xs=skew, origin=o).skew(xs=-skew, origin=o)
        assert geom_almost_equals(expected, res)

        res = self.gdf1.set_geometry(self.g4).skew(xs=skew, origin=o)
        res = res.skew(xs=-skew, origin=o)
        assert geom_almost_equals(expected, res)

        # Test ys
        res = self.g4.skew(ys=skew, origin=o).skew(ys=-skew, origin=o)
        assert geom_almost_equals(expected, res)

        res = self.gdf1.set_geometry(self.g4).skew(ys=skew, origin=o)
        res = res.skew(ys=-skew, origin=o)
        assert geom_almost_equals(expected, res)

    def test_buffer(self):
        original = GeoSeries([Point(0, 0)])
        expected = GeoSeries([Polygon(((5, 0), (0, -5), (-5, 0), (0, 5), (5, 0)))])
        calculated = original.buffer(5, resolution=1)
        assert geom_almost_equals(expected, calculated)

    def test_buffer_args(self):
        args = {"cap_style": 3, "join_style": 2, "mitre_limit": 2.5}
        calculated_series = self.g0.buffer(10, **args)
        for original, calculated in zip(self.g0, calculated_series):
            if original is None:
                assert calculated is None
            else:
                expected = original.buffer(10, **args)
                assert calculated.equals(expected)

    def test_buffer_distance_array(self):
        original = GeoSeries([self.p0, self.p0])
        expected = GeoSeries(
            [
                Polygon(((6, 5), (5, 4), (4, 5), (5, 6), (6, 5))),
                Polygon(((10, 5), (5, 0), (0, 5), (5, 10), (10, 5))),
            ]
        )
        calculated = original.buffer(np.array([1, 5]), resolution=1)
        assert_geoseries_equal(calculated, expected, check_less_precise=True)

    def test_buffer_distance_wrong_length(self):
        original = GeoSeries([self.p0, self.p0])
        distances = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            original.buffer(distances)

    def test_buffer_distance_series(self):
        original = GeoSeries([self.p0, self.p0])
        expected = GeoSeries(
            [
                Polygon(((6, 5), (5, 4), (4, 5), (5, 6), (6, 5))),
                Polygon(((10, 5), (5, 0), (0, 5), (5, 10), (10, 5))),
            ]
        )
        calculated = original.buffer(Series([1, 5]), resolution=1)
        assert_geoseries_equal(calculated, expected, check_less_precise=True)

    def test_buffer_distance_wrong_index(self):
        original = GeoSeries([self.p0, self.p0], index=[0, 1])
        distances = Series(data=[1, 2], index=[99, 98])
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'distance' does not match"
        ):
            original.buffer(distances)

    def test_buffer_empty_none(self):
        p = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        s = GeoSeries([p, GeometryCollection(), None])
        result = s.buffer(0)
        assert_geoseries_equal(result, s)

        result = s.buffer(np.array([0, 0, 0]))
        assert_geoseries_equal(result, s)

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
    def test_buffer_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.buffer(1)

        with warnings.catch_warnings(record=True) as record:
            # do not warn for 0
            self.g4.buffer(0)

        for r in record:
            assert "Geometry is in a geographic CRS." not in str(r.message)

    def test_simplify(self):
        s = GeoSeries([shapely.LineString([(0, 0), (1, 0.1), (2, 0)])])
        e = GeoSeries([shapely.LineString([(0, 0), (2, 0)])])
        assert_geoseries_equal(s.simplify(0.2), e)

    def test_simplify_wrong_index(self):
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'tolerance' does not match"
        ):
            self.g1.simplify(Series([0.1], index=[99]))

    @pytest.mark.skipif(
        not (GEOS_GE_312 and SHAPELY_GE_21), reason="GEOS 3.12 and shapely 2.1 needed."
    )
    def test_simplify_coverage(self):
        s = GeoSeries(
            [
                shapely.Polygon(
                    [(0, 0), (10, 1), (20, 0), (20, 10), (10, 5), (0, 10), (0, 0)]
                ),
                shapely.Polygon(
                    [(0, 10), (10, 5), (20, 10), (20, 20), (0, 20), (0, 10)]
                ),
            ]
        )
        e = GeoSeries(
            [
                shapely.Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]),
                shapely.Polygon([(0, 10), (20, 10), (20, 20), (0, 20)]),
            ]
        )
        assert_geoseries_equal(s.simplify_coverage(8), e.normalize())

        e_boundary = GeoSeries(
            [
                shapely.Polygon([(0, 0), (10, 1), (20, 0), (20, 10), (0, 10)]),
                shapely.Polygon([(0, 10), (20, 10), (20, 20), (0, 20)]),
            ]
        )
        assert_geoseries_equal(
            s.simplify_coverage(8, simplify_boundary=False), e_boundary.normalize()
        )

    def test_envelope(self):
        e = self.g3.envelope
        assert np.all(e.geom_equals(self.sq))
        assert isinstance(e, GeoSeries)
        assert self.g3.crs == e.crs

    def test_minimum_rotated_rectangle(self):
        s = GeoSeries([self.sq, self.t5], crs=3857)
        r = s.minimum_rotated_rectangle()
        exp = GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
                "POLYGON ((2 0, 2 3, 3 3, 3 0, 2 0))",
            ],
            crs=3857,
        )

        assert np.all(r.normalize().geom_equals_exact(exp, 0.001))
        assert isinstance(r, GeoSeries)
        assert s.crs == r.crs

    def test_extract_unique_points(self):
        eup = GeoSeries([self.t6]).extract_unique_points()
        expected = GeoSeries([MultiPoint([(2, 0), (3, 0)])])
        assert_series_equal(eup, expected)

    def test_minimum_bounding_circle(self):
        mbc = self.g1.minimum_bounding_circle()
        centers = GeoSeries([Point(0.5, 0.5)] * 2)
        assert np.all(mbc.centroid.geom_equals_exact(centers, 0.001))
        assert_series_equal(
            mbc.area,
            Series([1.560723, 1.560723]),
        )
        assert isinstance(mbc, GeoSeries)
        assert self.g1.crs == mbc.crs

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires shapely 2.1")
    def test_maximum_inscribed_circle(self):
        gs = GeoSeries(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (0.5, -1), (1, 0), (1, 1), (-0.5, 0.5)]),
            ]
        )
        mic = gs.maximum_inscribed_circle()
        assert (mic.geom_type == "LineString").all()
        assert (shapely.get_num_points(mic) == 2).all()
        expected_centers = GeoSeries([Point(0.5, 0.5), Point(0.466796875, 0.259765625)])
        expected_length = Series([0.5, 0.533203125])
        assert_geoseries_equal(
            shapely.get_point(mic, 0), expected_centers, check_less_precise=True
        )
        assert_series_equal(shapely.length(mic), expected_length)

        # with tolerance for second polygon -> stops earlier with smaller circle, thus
        # just assert the length of the resulting line is lower
        mic_tolerance = gs.maximum_inscribed_circle(tolerance=np.array([0, 10]))
        assert (shapely.length(mic_tolerance) <= 0.5).all()

    def test_total_bounds(self):
        bbox = self.sol.x, self.sol.y, self.esb.x, self.esb.y
        assert isinstance(self.landmarks.total_bounds, np.ndarray)
        assert tuple(self.landmarks.total_bounds) == bbox

        df = GeoDataFrame(
            {"geometry": self.landmarks, "col1": range(len(self.landmarks))}
        )
        assert tuple(df.total_bounds) == bbox

    def test_explode_geoseries(self):
        s = GeoSeries(
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])],
            crs=4326,
        )
        s.index.name = "test_index_name"
        expected_index_name = ["test_index_name", None]
        index = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        expected = GeoSeries(
            [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)],
            index=MultiIndex.from_tuples(index, names=expected_index_name),
            crs=4326,
        )
        assert_geoseries_equal(expected, s.explode(index_parts=True))

    @pytest.mark.parametrize("index_name", [None, "test"])
    def test_explode_geodataframe(self, index_name):
        s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
        df = GeoDataFrame({"col": [1, 2], "geometry": s})
        df.index.name = index_name

        test_df = df.explode(index_parts=True)

        expected_s = GeoSeries([Point(1, 2), Point(2, 3), Point(5, 5)])
        expected_df = GeoDataFrame({"col": [1, 1, 2], "geometry": expected_s})
        expected_index = MultiIndex(
            [[0, 1], [0, 1]],  # levels
            [[0, 0, 1], [0, 1, 0]],  # labels/codes
            names=[index_name, None],
        )
        expected_df = expected_df.set_index(expected_index)
        assert_frame_equal(test_df, expected_df)

    @pytest.mark.parametrize("index_name", [None, "test"])
    def test_explode_geodataframe_level_1(self, index_name):
        # GH1393
        s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
        df = GeoDataFrame({"level_1": [1, 2], "geometry": s})
        df.index.name = index_name

        test_df = df.explode(index_parts=True)

        expected_s = GeoSeries([Point(1, 2), Point(2, 3), Point(5, 5)])
        expected_df = GeoDataFrame({"level_1": [1, 1, 2], "geometry": expected_s})
        expected_index = MultiIndex(
            [[0, 1], [0, 1]],  # levels
            [[0, 0, 1], [0, 1, 0]],  # labels/codes
            names=[index_name, None],
        )
        expected_df = expected_df.set_index(expected_index)
        assert_frame_equal(test_df, expected_df)

    @pytest.mark.parametrize("index_name", [None, "test"])
    def test_explode_geodataframe_no_multiindex(self, index_name):
        # GH1393
        s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
        df = GeoDataFrame({"level_1": [1, 2], "geometry": s})
        df.index.name = index_name

        test_df = df.explode(index_parts=False)

        expected_s = GeoSeries([Point(1, 2), Point(2, 3), Point(5, 5)])
        expected_df = GeoDataFrame({"level_1": [1, 1, 2], "geometry": expected_s})

        expected_index = Index([0, 0, 1], name=index_name)
        expected_df = expected_df.set_index(expected_index)
        assert_frame_equal(test_df, expected_df)

    def test_explode_pandas_fallback(self):
        d = {
            "col1": [["name1", "name2"], ["name3", "name4"]],
            "geometry": [MultiPoint([(1, 2), (3, 4)]), MultiPoint([(2, 1), (0, 0)])],
        }
        gdf = GeoDataFrame(d, crs=4326)
        expected_df = GeoDataFrame(
            {
                "col1": ["name1", "name2", "name3", "name4"],
                "geometry": [
                    MultiPoint([(1, 2), (3, 4)]),
                    MultiPoint([(1, 2), (3, 4)]),
                    MultiPoint([(2, 1), (0, 0)]),
                    MultiPoint([(2, 1), (0, 0)]),
                ],
            },
            index=[0, 0, 1, 1],
            crs=4326,
        )

        # Test with column provided as arg
        exploded_df = gdf.explode("col1")
        assert_geodataframe_equal(exploded_df, expected_df)

        # Test with column provided as kwarg
        exploded_df = gdf.explode(column="col1")
        assert_geodataframe_equal(exploded_df, expected_df)

    def test_explode_pandas_fallback_ignore_index(self):
        d = {
            "col1": [["name1", "name2"], ["name3", "name4"]],
            "geometry": [MultiPoint([(1, 2), (3, 4)]), MultiPoint([(2, 1), (0, 0)])],
        }
        gdf = GeoDataFrame(d, crs=4326)
        expected_df = GeoDataFrame(
            {
                "col1": ["name1", "name2", "name3", "name4"],
                "geometry": [
                    MultiPoint([(1, 2), (3, 4)]),
                    MultiPoint([(1, 2), (3, 4)]),
                    MultiPoint([(2, 1), (0, 0)]),
                    MultiPoint([(2, 1), (0, 0)]),
                ],
            },
            crs=4326,
        )

        # Test with column provided as arg
        exploded_df = gdf.explode("col1", ignore_index=True)
        assert_geodataframe_equal(exploded_df, expected_df)

        # Test with column provided as kwarg
        exploded_df = gdf.explode(column="col1", ignore_index=True)
        assert_geodataframe_equal(exploded_df, expected_df)

    @pytest.mark.parametrize("outer_index", [1, (1, 2), "1"])
    def test_explode_pandas_multi_index(self, outer_index):
        index = MultiIndex.from_arrays(
            [[outer_index, outer_index, outer_index], [1, 2, 3]],
            names=("first", "second"),
        )
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)],
            index=index,
        )

        test_df = df.explode(index_parts=True)

        expected_s = GeoSeries(
            [
                Point(0, 0),
                Point(0, 0),
                Point(1, 1),
                Point(1, 0),
                Point(2, 2),
                Point(2, 0),
            ]
        )
        expected_df = GeoDataFrame({"vals": [1, 1, 2, 2, 3, 3], "geometry": expected_s})
        expected_index = MultiIndex.from_tuples(
            [
                (outer_index, *pair)
                for pair in [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
            ],
            names=["first", "second", None],
        )
        expected_df = expected_df.set_index(expected_index)
        assert_frame_equal(test_df, expected_df)

    @pytest.mark.parametrize("outer_index", [1, (1, 2), "1"])
    def test_explode_pandas_multi_index_false(self, outer_index):
        index = MultiIndex.from_arrays(
            [[outer_index, outer_index, outer_index], [1, 2, 3]],
            names=("first", "second"),
        )
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)],
            index=index,
        )

        test_df = df.explode(index_parts=False)

        expected_s = GeoSeries(
            [
                Point(0, 0),
                Point(0, 0),
                Point(1, 1),
                Point(1, 0),
                Point(2, 2),
                Point(2, 0),
            ]
        )
        expected_df = GeoDataFrame({"vals": [1, 1, 2, 2, 3, 3], "geometry": expected_s})
        expected_index = MultiIndex.from_tuples(
            [
                (outer_index, 1),
                (outer_index, 1),
                (outer_index, 2),
                (outer_index, 2),
                (outer_index, 3),
                (outer_index, 3),
            ],
            names=["first", "second"],
        )
        expected_df = expected_df.set_index(expected_index)
        assert_frame_equal(test_df, expected_df)

    @pytest.mark.parametrize("outer_index", [1, (1, 2), "1"])
    def test_explode_pandas_multi_index_ignore_index(self, outer_index):
        index = MultiIndex.from_arrays(
            [[outer_index, outer_index, outer_index], [1, 2, 3]],
            names=("first", "second"),
        )
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)],
            index=index,
        )

        test_df = df.explode(ignore_index=True)

        expected_s = GeoSeries(
            [
                Point(0, 0),
                Point(0, 0),
                Point(1, 1),
                Point(1, 0),
                Point(2, 2),
                Point(2, 0),
            ]
        )
        expected_df = GeoDataFrame({"vals": [1, 1, 2, 2, 3, 3], "geometry": expected_s})
        expected_index = Index(range(len(expected_df)))
        expected_df = expected_df.set_index(expected_index)
        assert_frame_equal(test_df, expected_df)

        # index_parts is ignored if ignore_index=True
        test_df = df.explode(ignore_index=True, index_parts=True)
        assert_frame_equal(test_df, expected_df)

    def test_explode_order(self):
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)],
            index=[2, 9, 7],
        )
        test_df = df.explode(index_parts=True)

        expected_index = MultiIndex.from_arrays(
            [[2, 2, 9, 9, 7, 7], [0, 1, 0, 1, 0, 1]],
        )
        expected_geometry = GeoSeries(
            [
                Point(0, 0),
                Point(0, 0),
                Point(1, 1),
                Point(1, 0),
                Point(2, 2),
                Point(2, 0),
            ],
            index=expected_index,
        )
        expected_df = GeoDataFrame(
            {"vals": [1, 1, 2, 2, 3, 3]},
            geometry=expected_geometry,
            index=expected_index,
        )
        assert_geodataframe_equal(test_df, expected_df)

    def test_explode_order_no_multi(self):
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[Point(0, x) for x in range(3)],
            index=[2, 9, 7],
        )
        test_df = df.explode(index_parts=True)

        expected_index = MultiIndex.from_arrays(
            [[2, 9, 7], [0, 0, 0]],
        )
        expected_df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[Point(0, x) for x in range(3)],
            index=expected_index,
        )
        assert_geodataframe_equal(test_df, expected_df)

    def test_explode_order_mixed(self):
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(2)] + [Point(0, 10)],
            index=[2, 9, 7],
        )
        test_df = df.explode(index_parts=True)

        expected_index = MultiIndex.from_arrays(
            [[2, 2, 9, 9, 7], [0, 1, 0, 1, 0]],
        )
        expected_geometry = GeoSeries(
            [
                Point(0, 0),
                Point(0, 0),
                Point(1, 1),
                Point(1, 0),
                Point(0, 10),
            ],
            index=expected_index,
        )
        expected_df = GeoDataFrame(
            {"vals": [1, 1, 2, 2, 3]},
            geometry=expected_geometry,
            index=expected_index,
        )
        assert_geodataframe_equal(test_df, expected_df)

    def test_explode_duplicated_index(self):
        df = GeoDataFrame(
            {"vals": [1, 2, 3]},
            geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)],
            index=[1, 1, 2],
        )
        test_df = df.explode(index_parts=True)
        expected_index = MultiIndex.from_arrays(
            [[1, 1, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )
        expected_geometry = GeoSeries(
            [
                Point(0, 0),
                Point(0, 0),
                Point(1, 1),
                Point(1, 0),
                Point(2, 2),
                Point(2, 0),
            ],
            index=expected_index,
        )
        expected_df = GeoDataFrame(
            {"vals": [1, 1, 2, 2, 3, 3]},
            geometry=expected_geometry,
            index=expected_index,
        )
        assert_geodataframe_equal(test_df, expected_df)

    @pytest.mark.parametrize("geom_col", ["geom", "geometry"])
    def test_explode_geometry_name(self, geom_col):
        s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
        df = GeoDataFrame({"col": [1, 2], geom_col: s}, geometry=geom_col)
        test_df = df.explode(index_parts=True)

        assert test_df.geometry.name == geom_col
        assert test_df.geometry.name == test_df._geometry_column_name

    def test_explode_geometry_name_two_geoms(self):
        s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
        df = GeoDataFrame({"col": [1, 2], "geom": s, "geometry": s}, geometry="geom")
        test_df = df.explode(index_parts=True)

        assert test_df.geometry.name == "geom"
        assert test_df.geometry.name == test_df._geometry_column_name
        assert "geometry" in test_df.columns

    def test_get_coordinates(self):
        expected = DataFrame(
            data=self.expected_2d,
            columns=["x", "y"],
            index=[0, 1, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6],
        )
        assert_frame_equal(self.g11.get_coordinates(), expected)

    def test_get_coordinates_z(self):
        expected = DataFrame(
            data=self.expected_3d,
            columns=["x", "y", "z"],
            index=[0, 1, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6],
        )
        assert_frame_equal(self.g11.get_coordinates(include_z=True), expected)

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires shapely 2.1")
    def test_get_coordinates_m(self):
        s = GeoSeries.from_wkt(
            [
                "POINT M (2 3 5)",
                "POINT ZM (1 2 3 4)",
            ],
        )

        # only m
        expected = DataFrame(
            data=np.array([[2.0, 3.0, 5.0], [1.0, 2.0, 4.0]]),
            columns=["x", "y", "m"],
        )
        assert_frame_equal(s.get_coordinates(include_m=True), expected)

        # only z
        expected = DataFrame(
            data=np.array([[2.0, 3.0, np.nan], [1.0, 2.0, 3.0]]),
            columns=["x", "y", "z"],
        )
        assert_frame_equal(s.get_coordinates(include_z=True), expected)

        # both
        expected = DataFrame(
            data=np.array([[2.0, 3.0, np.nan, 5.0], [1.0, 2.0, 3.0, 4.0]]),
            columns=["x", "y", "z", "m"],
        )
        assert_frame_equal(s.get_coordinates(include_z=True, include_m=True), expected)

    def test_get_coordinates_ignore(self):
        expected = DataFrame(
            data=self.expected_2d,
            columns=["x", "y"],
        )
        assert_frame_equal(self.g11.get_coordinates(ignore_index=True), expected)

    def test_get_coordinates_parts(self):
        expected = DataFrame(
            data=self.expected_2d,
            columns=["x", "y"],
            index=MultiIndex.from_tuples(
                [
                    (0, 0),
                    (1, 0),
                    (3, 0),
                    (3, 1),
                    (3, 2),
                    (3, 3),
                    (4, 0),
                    (4, 1),
                    (4, 2),
                    (4, 3),
                    (6, 0),
                    (6, 1),
                    (6, 2),
                ]
            ),
        )
        assert_frame_equal(self.g11.get_coordinates(index_parts=True), expected)

    def test_minimum_bounding_radius(self):
        mbr_geoms = self.g1.minimum_bounding_radius()

        assert_series_equal(
            mbr_geoms,
            Series([0.707106, 0.707106]),
        )

        mbr_lines = self.g5.minimum_bounding_radius()

        assert_series_equal(
            mbr_lines,
            Series([0.707106, 0.707106]),
        )

    def test_minimum_clearance(self):
        mc_geoms = self.g1.minimum_clearance()

        assert_series_equal(
            mc_geoms,
            Series([0.707107, 1.000000]),
        )

        mc_lines = self.g5.minimum_clearance()

        assert_series_equal(
            mc_lines,
            Series([1.0, 1.0]),
        )

    @pytest.mark.skipif(not SHAPELY_GE_21, reason="requires shapely 2.1")
    def test_minimum_clearance_line(self):
        mcl_geoms = self.g1.minimum_clearance_line()

        expected = GeoSeries(
            [LineString([(1, 0), (0.5, 0.5)]), LineString([(0, 0), (1, 0)])]
        )
        assert_geoseries_equal(mcl_geoms, expected)

    @pytest.mark.parametrize("size", [10, 20, 50])
    def test_sample_points(self, size):
        for gs in (
            self.g1,
            self.na,
            self.a1,
            self.na_none,
        ):
            output = gs.sample_points(size)
            assert_index_equal(gs.index, output.index)
            assert (
                len(output.explode(ignore_index=True))
                == len(gs[~(gs.is_empty | gs.isna())]) * size
            )
        with pytest.warns(FutureWarning, match="The 'seed' keyword is deprecated"):
            _ = gs.sample_points(size, seed=1)

    def test_sample_points_array(self):
        output = concat([self.g1, self.g1]).sample_points([10, 15, 20, 25])
        expected = Series(
            [10, 15, 20, 25], index=[0, 1, 0, 1], name="sampled_points", dtype="int32"
        )
        assert_series_equal(shapely.get_num_geometries(output), expected)

    @pytest.mark.parametrize("size", [10, 20, 50])
    def test_sample_points_pointpats(self, size):
        pytest.importorskip("pointpats")
        for gs in (
            self.g1,
            self.na,
            self.a1,
        ):
            output = gs.sample_points(size, method="cluster_poisson")
            assert_index_equal(gs.index, output.index)
            assert (
                len(output.explode(ignore_index=True)) == len(gs[~gs.is_empty]) * size
            )

        with pytest.raises(AttributeError, match="pointpats.random module has no"):
            gs.sample_points(10, method="nonexistent")

    def test_offset_curve(self):
        oc = GeoSeries([self.l1]).offset_curve(1, join_style="mitre")
        expected = GeoSeries([LineString([[-1, 0], [-1, 2], [1, 2]])])
        assert_geoseries_equal(expected, oc)
        assert isinstance(oc, GeoSeries)

    def test_offset_curve_wrong_index(self):
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'distance' does not match"
        ):
            GeoSeries([self.l1]).offset_curve(Series([1], index=[99]))

    def test_polygonize(self):
        expected = GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 0.5 0.5, 0.5 0, 0 0))",
                "POLYGON ((0.5 0.5, 0 0, 0 1, 0.5 1, 0.5 0.5))",
                "POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0, 0.5 0.5))",
                "POLYGON ((1 1, 0.5 0.5, 0.5 1, 1 1))",
            ],
            name="polygons",
            crs=4326,
        )

        result = self.lines.polygonize()
        assert_geoseries_equal(expected, result)
        assert_index_equal(self.lines.index, Index(range(2, 10)))

    def test_polygonize_no_node(self):
        expected = GeoSeries.from_wkt(
            ["POLYGON ((0 0, 1 1, 1 0, 0 0))", "POLYGON ((1 1, 0 0, 0 1, 1 1))"],
            name="polygons",
            crs=4326,
        )
        result = self.lines.polygonize(node=False)
        assert_geoseries_equal(expected, result)
        assert_index_equal(self.lines.index, Index(range(2, 10)))

    def test_polygonize_full(self):
        expected_poly = GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 0.5 0.5, 0.5 0, 0 0))",
                "POLYGON ((0.5 0.5, 0 0, 0 1, 0.5 1, 0.5 0.5))",
                "POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0, 0.5 0.5))",
                "POLYGON ((1 1, 0.5 0.5, 0.5 1, 1 1))",
            ],
            name="polygons",
            crs=4326,
        )
        expected_cuts = GeoSeries([], name="cut edges", crs=4326)
        expected_dangles = GeoSeries.from_wkt(
            [
                "LINESTRING (5 5, 6 6)",
                "LINESTRING (0.5 1, 0.5 2)",
                "LINESTRING (0.5 -1, 0.5 0)",
            ],
            name="dangles",
            crs=4326,
        )
        expected_invalid = GeoSeries([], name="invalid ring lines", crs=4326)
        result = self.lines.polygonize(full=True)
        assert_geoseries_equal(expected_poly, result[0])
        assert_geoseries_equal(expected_cuts, result[1])
        assert_geoseries_equal(expected_dangles, result[2])
        assert_geoseries_equal(expected_invalid, result[3])
        assert_index_equal(self.lines.index, Index(range(2, 10)))

    @pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="requires GEOS>=3.11")
    @pytest.mark.parametrize(
        "geom,expected",
        [
            (
                GeoSeries(LinearRing([(0, 0), (1, 2), (1, 2), (1, 3), (0, 0)])),
                GeoSeries(LinearRing([(0, 0), (1, 2), (1, 3), (0, 0)])),
            ),
            (
                GeoSeries(Polygon([(0, 0), (0, 0), (1, 0), (1, 1), (1, 0), (0, 0)])),
                GeoSeries(Polygon([(0, 0), (1, 0), (1, 1), (1, 0), (0, 0)])),
            ),
        ],
    )
    def test_remove_repeated_points(self, geom, expected):
        assert_geoseries_equal(expected, geom.remove_repeated_points(tolerance=0.0))

    def test_remove_repeated_points_wrong_index(self):
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'tolerance' does not match"
        ):
            GeoSeries([self.l1]).remove_repeated_points(Series([1], index=[99]))

    def test_force_2d(self):
        expected = GeoSeries(
            [
                Point(-73.9847, 40.7484),
                Point(-74.0446, 40.6893),
                self.pt2d,
                self.pt_empty,
            ],
            crs=4326,
        )
        assert_geoseries_equal(expected, self.landmarks_mixed_empty.force_2d())

    def test_force_3d(self):
        expected = GeoSeries(
            [
                self.esb,
                self.sol,
                Point(-73.9847, 40.7484, 0),
                self.pt_empty,
            ],
            crs=4326,
        )
        assert_geoseries_equal(expected, self.landmarks_mixed_empty.force_3d())

        expected = GeoSeries(
            [
                self.esb,
                self.sol,
                Point(-73.9847, 40.7484, 2),
                self.pt_empty,
            ],
            crs=4326,
        )
        assert_geoseries_equal(expected, self.landmarks_mixed_empty.force_3d(2))

        expected = GeoSeries(
            [
                Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 1)]),
                Polygon([(0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 2), (0, 0, 2)]),
            ],
        )
        assert_geoseries_equal(expected, self.g1.force_3d([1, 2]))

    def test_shared_paths(self):
        line = LineString([(0, 0), (0.5, 0.5), (0, 1)])
        expected = GeoSeries.from_wkt(
            [
                "GEOMETRYCOLLECTION (MULTILINESTRING ((0 0, 0.5 0.5)),"
                " MULTILINESTRING EMPTY)",
                "GEOMETRYCOLLECTION (MULTILINESTRING EMPTY,"
                " MULTILINESTRING ((0 1, 0.5 0.5)))",
            ]
        )
        assert_geoseries_equal(expected, self.crossed_lines.shared_paths(line))

        s2 = GeoSeries(
            [
                LineString([(0, 0), (0.5, 0.5), (1, 0), (1, 1), (0.9, 0.9)]),
                LineString([(1, 1), (0, 1), (1, 0)]),
            ],
            index=[1, 2],
        )
        expected = GeoSeries.from_wkt(
            [
                None,
                "GEOMETRYCOLLECTION (MULTILINESTRING ((0.5 0.5, 1 0)),"
                " MULTILINESTRING EMPTY)",
                None,
            ]
        )

        with pytest.warns(
            UserWarning,
            match="The indices of the left and right GeoSeries' are not equal",
        ):
            assert_geoseries_equal(
                self.crossed_lines.shared_paths(s2, align=None), expected
            )

        expected = GeoSeries.from_wkt(
            [
                "GEOMETRYCOLLECTION (MULTILINESTRING ((0 0, 0.5 0.5)),"
                " MULTILINESTRING ((0.9 0.9, 1 1)))",
                "GEOMETRYCOLLECTION (MULTILINESTRING ((0 1, 1 0)),"
                " MULTILINESTRING EMPTY)",
            ]
        )
        assert_geoseries_equal(
            self.crossed_lines.shared_paths(s2, align=False), expected
        )

    def test_force_3d_wrong_index(self):
        with pytest.raises(
            ValueError, match="Index of the Series passed as 'z' does not match"
        ):
            self.g1.force_3d(Series([1], index=[99]))

    def test_line_merge(self):
        expected = GeoSeries(
            [
                LineString([(0, 2), (0, 10), (5, 10)]),
                MultiLineString([[(0, 2), (0, 10)], [(0, 11), (5, 10)]]),
                GeometryCollection(),
                LineString([(0, 0), (1, 0), (3, 0)]),
                GeometryCollection(),
            ],
            crs=4326,
            index=range(2, 7),
        )
        assert_geoseries_equal(expected, self.g14.line_merge())

    @pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="requires GEOS>=3.11")
    def test_line_merge_directed(self):
        expected = GeoSeries(
            [
                LineString([(0, 2), (0, 10), (5, 10)]),
                MultiLineString([[(0, 2), (0, 10)], [(0, 11), (5, 10)]]),
                GeometryCollection(),
                MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]),
                GeometryCollection(),
            ],
            crs=4326,
            index=range(2, 7),
        )
        assert_geoseries_equal(expected, self.g14.line_merge(directed=True))

    @pytest.mark.skipif(
        shapely.geos_version < (3, 11, 0), reason="different order in GEOS<3.11"
    )
    def test_build_area(self):
        # test with polygon in it
        s = GeoSeries.from_wkt(
            [
                "LINESTRING (18 4, 4 2, 2 9)",
                "LINESTRING (18 4, 16 16)",
                "LINESTRING (16 16, 8 19, 8 12, 2 9)",
                "LINESTRING (8 6, 12 13, 15 8)",
                "LINESTRING (8 6, 15 8)",
                "LINESTRING (0 0, 0 3, 3 3, 3 0, 0 0)",
                "POLYGON ((1 1, 2 2, 1 2, 1 1))",
                "LINESTRING (10 7, 13 8, 12 10, 10 7)",
            ],
            crs=4326,
        )

        expected = GeoSeries.from_wkt(
            [
                "POLYGON ((0 3, 3 3, 3 0, 0 0, 0 3), (2 2, 1 2, 1 1, 2 2))",
                "POLYGON ((13 8, 10 7, 12 10, 13 8))",
                "POLYGON ((2 9, 8 12, 8 19, 16 16, 18 4, 4 2, 2 9), "
                "(8 6, 15 8, 12 13, 8 6))",
            ],
            crs=4326,
            name="polygons",
        )
        assert_geoseries_equal(expected, s.build_area())

        # test difference caused by nodign
        s2 = GeoSeries.from_wkt(
            [
                "LINESTRING (8 6, 12 13, 15 8)",
                "LINESTRING (8 6, 15 8)",
                "LINESTRING (0 0, 0 15, 12 15, 12 0, 0 0)",
                "LINESTRING (10 7, 13 8, 12 10, 10 7)",
            ],
            crs=4326,
        )

        noded = GeoSeries.from_wkt(
            ["POLYGON ((12 0, 0 0, 0 15, 12 15, 12 13, 15 8, 12 7.142857, 12 0))"],
            crs=4326,
            name="polygons",
        )
        assert_geoseries_equal(noded, s2.build_area(node=True), check_less_precise=True)

        non_noded = GeoSeries.from_wkt(
            [
                "POLYGON ((0 15, 12 15, 12 13, 15 8, 12 7.142857, 12 0, 0 0, 0 15), "
                "(12 7.666667, 13 8, 12 10, 12 7.666667))"
            ],
            crs=4326,
            name="polygons",
        )
        assert_geoseries_equal(
            non_noded, s2.build_area(node=False), check_less_precise=True
        )

    @pytest.mark.skipif(
        shapely.geos_version < (3, 9, 5), reason="Empty geom bug in GEOS<3.9.5"
    )
    def test_set_precision(self):
        expected = GeoSeries(
            [
                Point(-74, 41, 30.3244),
                Point(-74, 41, 31.2344),
                Point(-74, 41),
                self.pt_empty,
            ],
            crs=4326,
        )
        assert_geoseries_equal(expected, self.landmarks_mixed_empty.set_precision(1))

        s = GeoSeries(
            [
                LineString([(0, 0), (0, 0.1), (0, 1), (1, 1)]),
                LineString([(0, 0), (0, 0.1), (0.1, 0.1)]),
            ],
        )
        expected = GeoSeries(
            [
                LineString([(0, 0), (0, 1), (1, 1)]),
                LineString(),
            ],
        )
        assert_geoseries_equal(expected, s.set_precision(1))

        expected = GeoSeries(
            [
                LineString([(0, 0), (0, 0), (0, 1), (1, 1)]),
                LineString([(0, 0), (0, 0), (0, 0)]),
            ]
        )
        assert_series_equal(
            expected.to_wkt(), s.set_precision(1, mode="pointwise").to_wkt()
        )

        expected = GeoSeries(
            [
                LineString([(0, 0), (0, 1), (1, 1)]),
                LineString([(0, 0), (0, 0)]),
            ]
        )
        assert_series_equal(
            expected.to_wkt(), s.set_precision(1, mode="keep_collapsed").to_wkt()
        )

    def test_get_precision(self):
        expected = Series([0.0, 0.0, 0.0, 0.0], index=self.landmarks_mixed_empty.index)
        assert_series_equal(expected, self.landmarks_mixed_empty.get_precision())
        with_precision = self.landmarks_mixed_empty.set_precision(1)
        expected = Series([1.0, 1.0, 1.0, 1.0], index=with_precision.index)
        assert_series_equal(expected, with_precision.get_precision())
        mixed = concat([self.landmarks_mixed_empty, with_precision])
        expected = Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=mixed.index)
        assert_series_equal(expected, mixed.get_precision())

    def test_get_geometry(self):
        expected = GeoSeries(
            [
                LineString([(0, 2), (0, 10)]),
                LineString([(0, 2), (0, 10)]),
                None,
                LineString([(0, 0), (1, 0)]),
                Point(0, 0),
            ],
            index=range(2, 7),
            crs=4326,
        )
        assert_series_equal(expected, self.g14.get_geometry(0))

        expected = GeoSeries(
            [
                LineString([(0, 10), (5, 10)]),
                LineString([(0, 11), (5, 10)]),
                None,
                LineString([(0, 0), (3, 0)]),
                None,
            ],
            index=range(2, 7),
            crs=4326,
        )
        assert_series_equal(expected, self.g14.get_geometry(1))

        expected = GeoSeries(
            [
                LineString([(0, 10), (5, 10)]),
                LineString([(0, 11), (5, 10)]),
                None,
                LineString([(0, 0), (3, 0)]),
                Point(0, 0),
            ],
            index=range(2, 7),
            crs=4326,
        )
        assert_series_equal(expected, self.g14.get_geometry(-1))

        expected = GeoSeries(
            [
                LineString([(0, 2), (0, 10)]),
                LineString([(0, 11), (5, 10)]),
                None,
                LineString([(0, 0), (3, 0)]),
                Point(0, 0),
            ],
            index=range(2, 7),
            crs=4326,
        )
        assert_series_equal(expected, self.g14.get_geometry([0, 1, 1, -1, 0]))
