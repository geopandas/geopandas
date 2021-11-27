import json
import os
import random
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.util.testing import assert_index_equal

from pyproj import CRS
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from geopandas import GeoSeries, GeoDataFrame, read_file, datasets, clip
from geopandas._compat import PYPROJ_LT_3, ignore_shapely2_warnings
from geopandas.array import GeometryArray, GeometryDtype
from geopandas.testing import assert_geoseries_equal

from geopandas.tests.util import geom_equals
from pandas.testing import assert_series_equal
import pytest


class TestSeries:
    def setup_method(self):
        self.tempdir = tempfile.mkdtemp()
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g1 = GeoSeries([self.t1, self.sq])
        self.g2 = GeoSeries([self.sq, self.t1])
        self.g3 = GeoSeries([self.t1, self.t2])
        self.g3.crs = "epsg:4326"
        self.g4 = GeoSeries([self.t2, self.t1])
        self.na = GeoSeries([self.t1, self.t2, Polygon()])
        self.na_none = GeoSeries([self.t1, self.t2, None])
        self.a1 = self.g1.copy()
        self.a1.index = ["A", "B"]
        self.a2 = self.g2.copy()
        self.a2.index = ["B", "C"]
        self.esb = Point(-73.9847, 40.7484)
        self.sol = Point(-74.0446, 40.6893)
        self.landmarks = GeoSeries([self.esb, self.sol], crs="epsg:4326")
        self.l1 = LineString([(0, 0), (0, 1), (1, 1)])
        self.l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g5 = GeoSeries([self.l1, self.l2])

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_copy(self):
        gc = self.g3.copy()
        assert type(gc) is GeoSeries
        assert self.g3.name == gc.name
        assert self.g3.crs == gc.crs

    def test_in(self):
        assert self.t1 in self.g1
        assert self.sq in self.g1
        assert self.t1 in self.a1
        assert self.t2 in self.g3
        assert self.sq not in self.g3
        assert 5 not in self.g3

    def test_align(self):
        a1, a2 = self.a1.align(self.a2)
        assert isinstance(a1, GeoSeries)
        assert isinstance(a2, GeoSeries)
        assert a2["A"] is None
        assert a1["B"].equals(a2["B"])
        assert a1["C"] is None

    def test_align_crs(self):
        a1 = self.a1
        a1.crs = "epsg:4326"
        a2 = self.a2
        a2.crs = "epsg:31370"

        res1, res2 = a1.align(a2)
        assert res1.crs == "epsg:4326"
        assert res2.crs == "epsg:31370"

        a2.crs = None
        res1, res2 = a1.align(a2)
        assert res1.crs == "epsg:4326"
        assert res2.crs is None

    def test_align_mixed(self):
        a1 = self.a1
        s2 = pd.Series([1, 2], index=["B", "C"])
        res1, res2 = a1.align(s2)

        exp2 = pd.Series([np.nan, 1, 2], index=["A", "B", "C"])
        assert_series_equal(res2, exp2)

    def test_warning_if_not_aligned(self):
        # GH-816
        # Test that warning is issued when operating on non-aligned series

        # _series_op
        with pytest.warns(UserWarning, match="The indices .+ different"):
            self.a1.contains(self.a2)

        # _geo_op
        with pytest.warns(UserWarning, match="The indices .+ different"):
            self.a1.union(self.a2)

    def test_no_warning_if_aligned(self):
        # GH-816
        # Test that warning is not issued when operating on aligned series
        a1, a2 = self.a1.align(self.a2)

        with pytest.warns(None) as warnings:
            a1.contains(a2)  # _series_op, explicitly aligned
            self.g1.intersects(self.g2)  # _series_op, implicitly aligned
            a2.union(a1)  # _geo_op, explicitly aligned
            self.g2.intersection(self.g1)  # _geo_op, implicitly aligned

        user_warnings = [w for w in warnings if w.category is UserWarning]
        assert not user_warnings, user_warnings[0].message

    def test_geom_equals(self):
        assert np.all(self.g1.geom_equals(self.g1))
        assert_array_equal(self.g1.geom_equals(self.sq), [False, True])

    def test_geom_equals_align(self):
        with pytest.warns(UserWarning, match="The indices .+ different"):
            a = self.a1.geom_equals(self.a2, align=True)
        exp = pd.Series([False, True, False], index=["A", "B", "C"])
        assert_series_equal(a, exp)

        a = self.a1.geom_equals(self.a2, align=False)
        exp = pd.Series([False, False], index=["A", "B"])
        assert_series_equal(a, exp)

    def test_geom_almost_equals(self):
        # TODO: test decimal parameter
        assert np.all(self.g1.geom_almost_equals(self.g1))
        assert_array_equal(self.g1.geom_almost_equals(self.sq), [False, True])

        assert_array_equal(
            self.a1.geom_almost_equals(self.a2, align=True), [False, True, False]
        )
        assert_array_equal(
            self.a1.geom_almost_equals(self.a2, align=False), [False, False]
        )

    def test_geom_equals_exact(self):
        # TODO: test tolerance parameter
        assert np.all(self.g1.geom_equals_exact(self.g1, 0.001))
        assert_array_equal(self.g1.geom_equals_exact(self.sq, 0.001), [False, True])

        assert_array_equal(
            self.a1.geom_equals_exact(self.a2, 0.001, align=True), [False, True, False]
        )
        assert_array_equal(
            self.a1.geom_equals_exact(self.a2, 0.001, align=False), [False, False]
        )

    def test_equal_comp_op(self):
        s = GeoSeries([Point(x, x) for x in range(3)])
        res = s == Point(1, 1)
        exp = pd.Series([False, True, False])
        assert_series_equal(res, exp)

    def test_to_file(self):
        """Test to_file and from_file"""
        tempfilename = os.path.join(self.tempdir, "test.shp")
        self.g3.to_file(tempfilename)
        # Read layer back in?
        s = GeoSeries.from_file(tempfilename)
        assert all(self.g3.geom_equals(s))
        # TODO: compare crs

    def test_to_json(self):
        """
        Test whether GeoSeries.to_json works and returns an actual json file.
        """
        json_str = self.g3.to_json()
        json.loads(json_str)
        # TODO : verify the output is a valid GeoJSON.

    def test_representative_point(self):
        assert np.all(self.g1.contains(self.g1.representative_point()))
        assert np.all(self.g2.contains(self.g2.representative_point()))
        assert np.all(self.g3.contains(self.g3.representative_point()))
        assert np.all(self.g4.contains(self.g4.representative_point()))

    def test_transform(self):
        utm18n = self.landmarks.to_crs(epsg=26918)
        lonlat = utm18n.to_crs(epsg=4326)
        assert np.all(self.landmarks.geom_almost_equals(lonlat))
        with pytest.raises(ValueError):
            self.g1.to_crs(epsg=4326)
        with pytest.raises(ValueError):
            self.landmarks.to_crs(crs=None, epsg=None)

    def test_estimate_utm_crs__geographic(self):
        if PYPROJ_LT_3:
            with pytest.raises(RuntimeError, match=r"pyproj 3\+ required"):
                self.landmarks.estimate_utm_crs()
        else:
            assert self.landmarks.estimate_utm_crs() == CRS("EPSG:32618")
            assert self.landmarks.estimate_utm_crs("NAD83") == CRS("EPSG:26918")

    @pytest.mark.skipif(PYPROJ_LT_3, reason="requires pyproj 3 or higher")
    def test_estimate_utm_crs__projected(self):
        assert self.landmarks.to_crs("EPSG:3857").estimate_utm_crs() == CRS(
            "EPSG:32618"
        )

    @pytest.mark.skipif(PYPROJ_LT_3, reason="requires pyproj 3 or higher")
    def test_estimate_utm_crs__out_of_bounds(self):
        with pytest.raises(RuntimeError, match="Unable to determine UTM CRS"):
            GeoSeries(
                [Polygon([(0, 90), (1, 90), (2, 90)])], crs="EPSG:4326"
            ).estimate_utm_crs()

    @pytest.mark.skipif(PYPROJ_LT_3, reason="requires pyproj 3 or higher")
    def test_estimate_utm_crs__missing_crs(self):
        with pytest.raises(RuntimeError, match="crs must be set"):
            GeoSeries([Polygon([(0, 90), (1, 90), (2, 90)])]).estimate_utm_crs()

    def test_fillna(self):
        # default is to fill with empty geometry
        na = self.na_none.fillna()
        assert isinstance(na[2], BaseGeometry)
        assert na[2].is_empty
        assert geom_equals(self.na_none[:2], na[:2])
        # XXX: method works inconsistently for different pandas versions
        # self.na_none.fillna(method='backfill')

    def test_coord_slice(self):
        """Test CoordinateSlicer"""
        # need some better test cases
        assert geom_equals(self.g3, self.g3.cx[:, :])
        assert geom_equals(self.g3[[True, False]], self.g3.cx[0.9:, :0.1])
        assert geom_equals(self.g3[[False, True]], self.g3.cx[0:0.1, 0.9:1.0])

    def test_coord_slice_with_zero(self):
        # Test that CoordinateSlice correctly handles zero slice (#GH477).

        gs = GeoSeries([Point(x, x) for x in range(-3, 4)])
        assert geom_equals(gs.cx[:0, :0], gs.loc[:3])
        assert geom_equals(gs.cx[:, :0], gs.loc[:3])
        assert geom_equals(gs.cx[:0, :], gs.loc[:3])
        assert geom_equals(gs.cx[0:, 0:], gs.loc[3:])
        assert geom_equals(gs.cx[0:, :], gs.loc[3:])
        assert geom_equals(gs.cx[:, 0:], gs.loc[3:])

    def test_geoseries_geointerface(self):
        assert self.g1.__geo_interface__["type"] == "FeatureCollection"
        assert len(self.g1.__geo_interface__["features"]) == self.g1.shape[0]

    def test_proj4strings(self):
        # As string
        reprojected = self.g3.to_crs("+proj=utm +zone=30")
        reprojected_back = reprojected.to_crs(epsg=4326)
        assert np.all(self.g3.geom_almost_equals(reprojected_back))

        # As dict
        reprojected = self.g3.to_crs({"proj": "utm", "zone": "30"})
        reprojected_back = reprojected.to_crs(epsg=4326)
        assert np.all(self.g3.geom_almost_equals(reprojected_back))

        # Set to equivalent string, convert, compare to original
        copy = self.g3.copy()
        copy.crs = "epsg:4326"
        reprojected = copy.to_crs({"proj": "utm", "zone": "30"})
        reprojected_back = reprojected.to_crs(epsg=4326)
        assert np.all(self.g3.geom_almost_equals(reprojected_back))

        # Conversions by different format
        reprojected_string = self.g3.to_crs("+proj=utm +zone=30")
        reprojected_dict = self.g3.to_crs({"proj": "utm", "zone": "30"})
        assert np.all(reprojected_string.geom_almost_equals(reprojected_dict))

    def test_from_wkb(self):
        assert_geoseries_equal(self.g1, GeoSeries.from_wkb([self.t1.wkb, self.sq.wkb]))

    def test_from_wkb_series(self):
        s = pd.Series([self.t1.wkb, self.sq.wkb], index=[1, 2])
        expected = self.g1.copy()
        expected.index = pd.Index([1, 2])
        assert_geoseries_equal(expected, GeoSeries.from_wkb(s))

    def test_from_wkb_series_with_index(self):
        index = [0]
        s = pd.Series([self.t1.wkb, self.sq.wkb], index=[0, 2])
        expected = self.g1.reindex(index)
        assert_geoseries_equal(expected, GeoSeries.from_wkb(s, index=index))

    def test_from_wkt(self):
        assert_geoseries_equal(self.g1, GeoSeries.from_wkt([self.t1.wkt, self.sq.wkt]))

    def test_from_wkt_series(self):
        s = pd.Series([self.t1.wkt, self.sq.wkt], index=[1, 2])
        expected = self.g1.copy()
        expected.index = pd.Index([1, 2])
        assert_geoseries_equal(expected, GeoSeries.from_wkt(s))

    def test_from_wkt_series_with_index(self):
        index = [0]
        s = pd.Series([self.t1.wkt, self.sq.wkt], index=[0, 2])
        expected = self.g1.reindex(index)
        assert_geoseries_equal(expected, GeoSeries.from_wkt(s, index=index))

    def test_to_wkb(self):
        assert_series_equal(pd.Series([self.t1.wkb, self.sq.wkb]), self.g1.to_wkb())
        assert_series_equal(
            pd.Series([self.t1.wkb_hex, self.sq.wkb_hex]), self.g1.to_wkb(hex=True)
        )

    def test_to_wkt(self):
        assert_series_equal(pd.Series([self.t1.wkt, self.sq.wkt]), self.g1.to_wkt())

    @pytest.mark.skip_no_sindex
    def test_clip(self):
        left = read_file(datasets.get_path("naturalearth_cities"))
        world = read_file(datasets.get_path("naturalearth_lowres"))
        south_america = world[world["continent"] == "South America"]

        expected = clip(left.geometry, south_america)
        result = left.geometry.clip(south_america)
        assert_geoseries_equal(result, expected)

    def test_from_xy_points(self):
        x = self.landmarks.x.values
        y = self.landmarks.y.values
        index = self.landmarks.index.tolist()
        crs = self.landmarks.crs
        assert_geoseries_equal(
            self.landmarks, GeoSeries.from_xy(x, y, index=index, crs=crs)
        )
        assert_geoseries_equal(
            self.landmarks,
            GeoSeries.from_xy(self.landmarks.x, self.landmarks.y, crs=crs),
        )

    def test_from_xy_points_w_z(self):
        index_values = [5, 6, 7]
        x = pd.Series([0, -1, 2], index=index_values)
        y = pd.Series([8, 3, 1], index=index_values)
        z = pd.Series([5, -6, 7], index=index_values)
        expected = GeoSeries(
            [Point(0, 8, 5), Point(-1, 3, -6), Point(2, 1, 7)], index=index_values
        )
        assert_geoseries_equal(expected, GeoSeries.from_xy(x, y, z))

    def test_from_xy_points_unequal_index(self):
        x = self.landmarks.x
        y = self.landmarks.y
        y.index = -np.arange(len(y))
        crs = self.landmarks.crs
        assert_geoseries_equal(
            self.landmarks, GeoSeries.from_xy(x, y, index=x.index, crs=crs)
        )
        unindexed_landmarks = self.landmarks.copy()
        unindexed_landmarks.reset_index(inplace=True, drop=True)
        assert_geoseries_equal(
            unindexed_landmarks,
            GeoSeries.from_xy(x, y, crs=crs),
        )

    def test_from_xy_points_indexless(self):
        x = np.array([0.0, 3.0])
        y = np.array([2.0, 5.0])
        z = np.array([-1.0, 4.0])
        expected = GeoSeries([Point(0, 2, -1), Point(3, 5, 4)])
        assert_geoseries_equal(expected, GeoSeries.from_xy(x, y, z))


def test_missing_values_empty_warning():
    s = GeoSeries([Point(1, 1), None, np.nan, BaseGeometry(), Polygon()])
    with pytest.warns(UserWarning):
        s.isna()

    with pytest.warns(UserWarning):
        s.notna()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_missing_values():
    s = GeoSeries([Point(1, 1), None, np.nan, BaseGeometry(), Polygon()])

    # construction -> missing values get normalized to None
    assert s[1] is None
    assert s[2] is None
    assert s[3].is_empty
    assert s[4].is_empty

    # isna / is_empty
    assert s.isna().tolist() == [False, True, True, False, False]
    assert s.is_empty.tolist() == [False, False, False, True, True]
    assert s.notna().tolist() == [True, False, False, True, True]

    # fillna defaults to fill with empty geometry -> no missing values anymore
    assert not s.fillna().isna().any()

    # dropna drops the missing values
    assert not s.dropna().isna().any()
    assert len(s.dropna()) == 3


def test_isna_empty_geoseries():
    # ensure that isna() result for emtpy GeoSeries has the correct bool dtype
    s = GeoSeries([])
    result = s.isna()
    assert_series_equal(result, pd.Series([], dtype="bool"))


def test_geoseries_crs():
    gs = GeoSeries()
    gs.crs = "IGNF:ETRS89UTM28"
    assert gs.crs.to_authority() == ("IGNF", "ETRS89UTM28")


# -----------------------------------------------------------------------------
# # Constructor tests
# -----------------------------------------------------------------------------


def check_geoseries(s):
    assert isinstance(s, GeoSeries)
    assert isinstance(s.geometry, GeoSeries)
    assert isinstance(s.dtype, GeometryDtype)
    assert isinstance(s.values, GeometryArray)


class TestConstructor:
    def test_constructor(self):
        s = GeoSeries([Point(x, x) for x in range(3)])
        check_geoseries(s)

    def test_single_geom_constructor(self):
        p = Point(1, 2)
        line = LineString([(2, 3), (4, 5), (5, 6)])
        poly = Polygon(
            [(0, 0), (1, 0), (1, 1), (0, 1)], [[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]]
        )
        mp = MultiPoint([(1, 2), (3, 4), (5, 6)])
        mline = MultiLineString([[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10)]])

        poly2 = Polygon(
            [(0, 0), (0, -1), (-1, -1), (-1, 0)],
            [[(-0.1, -0.1), (-0.1, -0.5), (-0.5, -0.5), (-0.5, -0.1)]],
        )
        mpoly = MultiPolygon([poly, poly2])

        geoms = [p, line, poly, mp, mline, mpoly]
        index = ["a", "b", "c", "d"]

        for g in geoms:
            gs = GeoSeries(g)
            assert len(gs) == 1
            # accessing elements no longer give identical objects
            assert gs.iloc[0].equals(g)

            gs = GeoSeries(g, index=index)
            assert len(gs) == len(index)
            for x in gs:
                assert x.equals(g)

    def test_no_geometries_fallback(self):
        with pytest.warns(FutureWarning):
            s = GeoSeries([True, False, True])
        assert not isinstance(s, GeoSeries)
        assert type(s) == pd.Series

        with pytest.warns(FutureWarning):
            s = GeoSeries(["a", "b", "c"])
        assert not isinstance(s, GeoSeries)
        assert type(s) == pd.Series

        with pytest.warns(FutureWarning):
            s = GeoSeries([[1, 2], [3, 4]])
        assert not isinstance(s, GeoSeries)
        assert type(s) == pd.Series

    def test_empty(self):
        s = GeoSeries([])
        check_geoseries(s)

        s = GeoSeries()
        check_geoseries(s)

    def test_data_is_none(self):
        s = GeoSeries(index=range(3))
        check_geoseries(s)

    def test_empty_array(self):
        # with empty data that have an explicit dtype, we use the fallback or
        # not depending on the dtype
        arr = np.array([], dtype="bool")

        # dtypes that can never hold geometry-like data
        for arr in [
            np.array([], dtype="bool"),
            np.array([], dtype="int64"),
            np.array([], dtype="float32"),
            # this gets converted to object dtype by pandas
            # np.array([], dtype="str"),
        ]:
            with pytest.warns(FutureWarning):
                s = GeoSeries(arr)
            assert not isinstance(s, GeoSeries)
            assert type(s) == pd.Series

        # dtypes that can potentially hold geometry-like data (object) or
        # can come from empty data (float64)
        for arr in [
            np.array([], dtype="object"),
            np.array([], dtype="float64"),
            np.array([], dtype="str"),
        ]:
            with pytest.warns(None) as record:
                s = GeoSeries(arr)
            assert not record
            assert isinstance(s, GeoSeries)

    def test_from_series(self):
        shapes = [
            Polygon([(random.random(), random.random()) for _ in range(3)])
            for _ in range(10)
        ]
        with ignore_shapely2_warnings():
            # the warning here is not suppressed by GeoPandas, as this is a pure
            # pandas construction call
            s = pd.Series(shapes, index=list("abcdefghij"), name="foo")
        g = GeoSeries(s)
        check_geoseries(g)

        assert [a.equals(b) for a, b in zip(s, g)]
        assert s.name == g.name
        assert s.index is g.index

    # GH 1216
    def test_expanddim(self):
        s = GeoSeries(
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])]
        )
        s = s.explode(index_parts=True)
        df = s.reset_index()
        assert type(df) == GeoDataFrame

    def test_explode_without_multiindex(self):
        s = GeoSeries(
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])]
        )
        s = s.explode(index_parts=False)
        expected_index = pd.Index([0, 0, 1, 1, 1])
        assert_index_equal(s.index, expected_index)

    def test_explode_ignore_index(self):
        s = GeoSeries(
            [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])]
        )
        s = s.explode(ignore_index=True)
        expected_index = pd.Index(range(len(s)))
        print(expected_index)
        assert_index_equal(s.index, expected_index)

        # index_parts is ignored if ignore_index=True
        s = s.explode(index_parts=True, ignore_index=True)
        assert_index_equal(s.index, expected_index)
