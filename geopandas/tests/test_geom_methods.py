import string

import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame, Index, MultiIndex, Series

from shapely.geometry import LinearRing, LineString, MultiPoint, Point, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union

from geopandas import GeoDataFrame, GeoSeries
from geopandas.base import GeoPandasBase

from geopandas.testing import assert_geodataframe_equal
from geopandas.tests.util import assert_geoseries_equal, geom_almost_equals, geom_equals
from geopandas import _compat as compat
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest


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
        self.l1 = LineString([(0, 0), (0, 1), (1, 1)])
        self.l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g5 = GeoSeries([self.l1, self.l2])
        self.g6 = GeoSeries([self.p0, self.t3])
        self.g7 = GeoSeries([self.sq, self.t4])
        self.g8 = GeoSeries([self.t1, self.t5])
        self.empty = GeoSeries([])
        self.all_none = GeoSeries([None, None])
        self.empty_poly = Polygon()
        self.g9 = GeoSeries(self.g0, index=range(1, 8))

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

    def _test_unary_real(self, op, expected, a):
        """Tests for 'area', 'length', 'is_valid', etc."""
        fcmp = assert_series_equal
        self._test_unary(op, expected, a, fcmp)

    def _test_unary_topological(self, op, expected, a):
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:

            def fcmp(a, b):
                assert a.equals(b)

        self._test_unary(op, expected, a, fcmp)

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

    def _test_binary_operator(self, op, expected, a, b):
        """
        The operators only have GeoSeries on the left, but can have
        GeoSeries or GeoDataFrame on the right.
        If GeoDataFrame is on the left, geometry column is used.

        """
        if isinstance(expected, GeoPandasBase):
            fcmp = assert_geoseries_equal
        else:

            def fcmp(a, b):
                assert geom_equals(a, b)

        if isinstance(b, GeoPandasBase):
            right_df = True
        else:
            right_df = False

        self._binary_op_test(op, expected, a, b, fcmp, False, right_df)

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

    def _test_unary(self, op, expected, a, fcmp):
        # GeoSeries, (GeoSeries or geometry)
        result = getattr(a, op)
        fcmp(result, expected)

        # GeoDataFrame, (GeoSeries or geometry)
        gdf = self.gdf1.set_geometry(a)
        result = getattr(gdf, op)
        fcmp(result, expected)

    # TODO re-enable for all operations once we use pyproj > 2
    # def test_crs_warning(self):
    #     # operations on geometries should warn for different CRS
    #     no_crs_g3 = self.g3.copy()
    #     no_crs_g3.crs = None
    #     with pytest.warns(UserWarning):
    #         self._test_binary_topological('intersection', self.g3,
    #                                       self.g3, no_crs_g3)

    def test_intersection(self):
        self._test_binary_topological("intersection", self.t1, self.g1, self.g2)
        with pytest.warns(UserWarning, match="The indices .+ different"):
            self._test_binary_topological(
                "intersection", self.all_none, self.g1, self.empty
            )

        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert len(self.g0.intersection(self.g9, align=True) == 8)
        assert len(self.g0.intersection(self.g9, align=False) == 7)

    def test_union_series(self):
        self._test_binary_topological("union", self.sq, self.g1, self.g2)

        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert len(self.g0.union(self.g9, align=True) == 8)
        assert len(self.g0.union(self.g9, align=False) == 7)

    def test_union_polygon(self):
        self._test_binary_topological("union", self.sq, self.g1, self.t2)

    def test_symmetric_difference_series(self):
        self._test_binary_topological("symmetric_difference", self.sq, self.g3, self.g4)

        with pytest.warns(UserWarning, match="The indices .+ different"):
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

        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert len(self.g0.difference(self.g9, align=True) == 8)
        assert len(self.g0.difference(self.g9, align=False) == 7)

    def test_difference_poly(self):
        expected = GeoSeries([self.t1, self.t1])
        self._test_binary_topological("difference", expected, self.g1, self.t2)

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

    def test_unary_union(self):
        p1 = self.t1
        p2 = Polygon([(2, 0), (3, 0), (3, 1)])
        expected = unary_union([p1, p2])
        g = GeoSeries([p1, p2])

        self._test_unary_topological("unary_union", expected, g)

        g2 = GeoSeries([p1, None])
        self._test_unary_topological("unary_union", p1, g2)

        g3 = GeoSeries([None, None])
        assert g3.unary_union is None

    def test_cascaded_union_deprecated(self):
        p1 = self.t1
        p2 = Polygon([(2, 0), (3, 0), (3, 1)])
        g = GeoSeries([p1, p2])
        with pytest.warns(
            FutureWarning, match="The 'cascaded_union' attribute is deprecated"
        ):
            result = g.cascaded_union
        assert result == g.unary_union

    def test_contains(self):
        expected = [True, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.contains(self.t1))

        expected = [False, True, True, True, True, True, False, False]
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.contains(self.g9, align=True))

        expected = [False, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.contains(self.g9, align=False))

    def test_length(self):
        expected = Series(np.array([2 + np.sqrt(2), 4]), index=self.g1.index)
        self._test_unary_real("length", expected, self.g1)

        expected = Series(np.array([2 + np.sqrt(2), np.nan]), index=self.na_none.index)
        self._test_unary_real("length", expected, self.na_none)

    def test_length_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.length

    def test_crosses(self):
        expected = [False, False, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.crosses(self.t1))

        expected = [False, True]
        assert_array_dtype_equal(expected, self.crossed_lines.crosses(self.l3))

        expected = [False] * 8
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.crosses(self.g9, align=True))

        expected = [False] * 7
        assert_array_dtype_equal(expected, self.g0.crosses(self.g9, align=False))

    def test_disjoint(self):
        expected = [False, False, False, False, False, True, False]
        assert_array_dtype_equal(expected, self.g0.disjoint(self.t1))

        expected = [False] * 8
        with pytest.warns(UserWarning, match="The indices .+ different"):
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
        assert_array_dtype_equal(expected, self.g0.relate(self.inner_sq))

        expected = Series(["FF0FFF212", None], index=self.g6.index)
        assert_array_dtype_equal(expected, self.g6.relate(self.na_none))

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

        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.relate(self.g9, align=True))

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
        assert_array_dtype_equal(expected, self.g0.relate(self.g9, align=False))

    def test_distance(self):
        expected = Series(
            np.array([np.sqrt((5 - 1) ** 2 + (5 - 1) ** 2), np.nan]), self.na_none.index
        )
        assert_array_dtype_equal(expected, self.na_none.distance(self.p0))

        expected = Series(np.array([np.sqrt(4 ** 2 + 4 ** 2), np.nan]), self.g6.index)
        assert_array_dtype_equal(expected, self.g6.distance(self.na_none))

        expected = Series(np.array([np.nan, 0, 0, 0, 0, 0, np.nan, np.nan]), range(8))
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.distance(self.g9, align=True))

        val = self.g0.iloc[4].distance(self.g9.iloc[4])
        expected = Series(np.array([0, 0, 0, 0, val, np.nan, np.nan]), self.g0.index)
        assert_array_dtype_equal(expected, self.g0.distance(self.g9, align=False))

    def test_distance_crs_warning(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.distance(self.p0)

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
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.intersects(self.g9, align=True))

        expected = [True, True, True, True, False, False, False]
        assert_array_dtype_equal(expected, self.g0.intersects(self.g9, align=False))

    def test_overlaps(self):
        expected = [True, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.overlaps(self.inner_sq))

        expected = [False, False]
        assert_array_dtype_equal(expected, self.g4.overlaps(self.t1))

        expected = [False] * 8
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.overlaps(self.g9, align=True))

        expected = [False] * 7
        assert_array_dtype_equal(expected, self.g0.overlaps(self.g9, align=False))

    def test_touches(self):
        expected = [False, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.touches(self.t1))

        expected = [False] * 8
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.touches(self.g9, align=True))

        expected = [True, False, False, True, False, False, False]
        assert_array_dtype_equal(expected, self.g0.touches(self.g9, align=False))

    def test_within(self):
        expected = [True, False, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.within(self.t1))

        expected = [True, True, True, True, True, False, False]
        assert_array_dtype_equal(expected, self.g0.within(self.sq))

        expected = [False, True, True, True, True, True, False, False]
        with pytest.warns(UserWarning, match="The indices .+ different"):
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
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.covers(self.g9, align=True))

        expected = [False, False, True, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.covers(self.g9, align=False))

    def test_covers_inverse(self):
        res = self.g8.covers(self.g7)
        exp = Series([False, False])
        assert_series_equal(res, exp)

    @pytest.mark.skipif(
        not compat.USE_PYGEOS,
        reason="covered_by is only implemented for pygeos, not shapely",
    )
    def test_covered_by(self):
        res = self.g1.covered_by(self.g1)
        exp = Series([True, True])
        assert_series_equal(res, exp)

        expected = [False, True, True, True, True, True, False, False]
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_array_dtype_equal(expected, self.g0.covered_by(self.g9, align=True))

        expected = [False, True, False, False, False, False, False]
        assert_array_dtype_equal(expected, self.g0.covered_by(self.g9, align=False))

    def test_is_valid(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_valid", expected, self.g1)

    def test_is_empty(self):
        expected = Series(np.array([False] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_empty", expected, self.g1)

    # for is_ring we raise a warning about the value for Polygon changing
    @pytest.mark.filterwarnings("ignore:is_ring:FutureWarning")
    def test_is_ring(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_ring", expected, self.g1)

    def test_is_simple(self):
        expected = Series(np.array([True] * len(self.g1)), self.g1.index)
        self._test_unary_real("is_simple", expected, self.g1)

    def test_has_z(self):
        expected = Series([False, True], self.g_3d.index)
        self._test_unary_real("has_z", expected, self.g_3d)

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

    def test_centroid_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.centroid

    def test_convex_hull(self):
        # the convex hull of a square should be the same as the square
        squares = GeoSeries([self.sq for i in range(3)])
        assert_geoseries_equal(squares, squares.convex_hull)

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
        with pytest.raises(ValueError):
            self.g5.interpolate(distances)

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
        with pytest.warns(UserWarning, match="The indices .+ different"):
            assert_series_equal(self.g5.project(s), expected)

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
        args = dict(cap_style=3, join_style=2, mitre_limit=2.5)
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

    def test_buffer_distance_wrong_index(self):
        original = GeoSeries([self.p0, self.p0], index=[0, 1])
        distances = Series(data=[1, 2], index=[99, 98])
        with pytest.raises(ValueError):
            original.buffer(distances)

    def test_buffer_empty_none(self):
        p = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        s = GeoSeries([p, GeometryCollection(), None])
        result = s.buffer(0)
        assert_geoseries_equal(result, s)

        result = s.buffer(np.array([0, 0, 0]))
        assert_geoseries_equal(result, s)

    def test_buffer_crs_warn(self):
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS"):
            self.g4.buffer(1)

        with pytest.warns(None) as record:
            # do not warn for 0
            self.g4.buffer(0)

        for r in record:
            assert "Geometry is in a geographic CRS." not in str(r.message)

    def test_envelope(self):
        e = self.g3.envelope
        assert np.all(e.geom_equals(self.sq))
        assert isinstance(e, GeoSeries)
        assert self.g3.crs == e.crs

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
        with pytest.warns(FutureWarning, match="Currently, index_parts defaults"):
            assert_geoseries_equal(expected, s.explode())

    @pytest.mark.parametrize("index_name", [None, "test"])
    def test_explode_geodataframe(self, index_name):
        s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
        df = GeoDataFrame({"col": [1, 2], "geometry": s})
        df.index.name = index_name

        with pytest.warns(FutureWarning, match="Currently, index_parts defaults"):
            test_df = df.explode()

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

    @pytest.mark.skipif(
        not compat.PANDAS_GE_11,
        reason="ignore_index keyword introduced in pandas 1.1.0",
    )
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

    #
    # Test '&', '|', '^', and '-'
    #
    def test_intersection_operator(self):
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__and__", self.t1, self.g1, self.g2)
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__and__", self.t1, self.gdf1, self.g2)

    def test_union_operator(self):
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__or__", self.sq, self.g1, self.g2)
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__or__", self.sq, self.gdf1, self.g2)

    def test_union_operator_polygon(self):
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__or__", self.sq, self.g1, self.t2)
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__or__", self.sq, self.gdf1, self.t2)

    def test_symmetric_difference_operator(self):
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__xor__", self.sq, self.g3, self.g4)
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__xor__", self.sq, self.gdf3, self.g4)

    def test_difference_series2(self):
        expected = GeoSeries([GeometryCollection(), self.t2])
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__sub__", expected, self.g1, self.g2)
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__sub__", expected, self.gdf1, self.g2)

    def test_difference_poly2(self):
        expected = GeoSeries([self.t1, self.t1])
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__sub__", expected, self.g1, self.t2)
        with pytest.warns(DeprecationWarning):
            self._test_binary_operator("__sub__", expected, self.gdf1, self.t2)
