from distutils.version import LooseVersion
import os

import random

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, LineString
import pyproj

from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray

from geopandas.testing import assert_geodataframe_equal
import pytest


# pyproj 2.3.1 fixed a segfault for the case working in an environment with
# 'init' dicts (https://github.com/pyproj4/pyproj/issues/415)
PYPROJ_LT_231 = LooseVersion(pyproj.__version__) < LooseVersion("2.3.1")


def _create_df(x, y=None, crs=None):
    y = y or x
    x = np.asarray(x)
    y = np.asarray(y)

    return GeoDataFrame(
        {"geometry": points_from_xy(x, y), "value1": x + y, "value2": x * y}, crs=crs
    )


def df_epsg26918():
    # EPSG:26918
    # Center coordinates
    # -1683723.64 6689139.23
    return _create_df(
        x=range(-1683723, -1683723 + 10, 1),
        y=range(6689139, 6689139 + 10, 1),
        crs="epsg:26918",
    )


def test_to_crs_transform():
    df = df_epsg26918()
    lonlat = df.to_crs(epsg=4326)
    utm = lonlat.to_crs(epsg=26918)
    assert_geodataframe_equal(df, utm, check_less_precise=True)


def test_to_crs_transform__missing_data():
    # https://github.com/geopandas/geopandas/issues/1573
    df = df_epsg26918()
    df.loc[3, "geometry"] = None
    lonlat = df.to_crs(epsg=4326)
    utm = lonlat.to_crs(epsg=26918)
    assert_geodataframe_equal(df, utm, check_less_precise=True)


def test_to_crs_inplace():
    df = df_epsg26918()
    lonlat = df.to_crs(epsg=4326)
    df.to_crs(epsg=4326, inplace=True)
    assert_geodataframe_equal(df, lonlat, check_less_precise=True)


def test_to_crs_geo_column_name():
    # Test to_crs() with different geometry column name (GH#339)
    df = df_epsg26918()
    df = df.rename(columns={"geometry": "geom"})
    df.set_geometry("geom", inplace=True)
    lonlat = df.to_crs(epsg=4326)
    utm = lonlat.to_crs(epsg=26918)
    assert lonlat.geometry.name == "geom"
    assert utm.geometry.name == "geom"
    assert_geodataframe_equal(df, utm, check_less_precise=True)


# -----------------------------------------------------------------------------
# Test different supported formats for CRS specification


@pytest.fixture(
    params=[
        4326,
        "epsg:4326",
        pytest.param(
            {"init": "epsg:4326"},
            marks=pytest.mark.skipif(PYPROJ_LT_231, reason="segfault"),
        ),
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84", "no_defs": True},
    ],
    ids=["epsg_number", "epsg_string", "epsg_dict", "proj4_string", "proj4_dict"],
)
def epsg4326(request):
    if isinstance(request.param, int):
        return dict(epsg=request.param)
    return dict(crs=request.param)


@pytest.fixture(
    params=[
        26918,
        "epsg:26918",
        pytest.param(
            {"init": "epsg:26918", "no_defs": True},
            marks=pytest.mark.skipif(PYPROJ_LT_231, reason="segfault"),
        ),
        "+proj=utm +zone=18 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ",
        {"proj": "utm", "zone": 18, "datum": "NAD83", "units": "m", "no_defs": True},
    ],
    ids=["epsg_number", "epsg_string", "epsg_dict", "proj4_string", "proj4_dict"],
)
def epsg26918(request):
    if isinstance(request.param, int):
        return dict(epsg=request.param)
    return dict(crs=request.param)


@pytest.mark.filterwarnings("ignore:'\\+init:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:'\\+init:FutureWarning")
def test_transform2(epsg4326, epsg26918):
    # with PROJ >= 7, the transformation using EPSG code vs proj4 string is
    # slightly different due to use of grid files or not -> turn off network
    # to not use grid files at all for this test
    os.environ["PROJ_NETWORK"] = "OFF"
    df = df_epsg26918()
    lonlat = df.to_crs(**epsg4326)
    utm = lonlat.to_crs(**epsg26918)
    # can't check for CRS equality, as the formats differ although representing
    # the same CRS
    assert_geodataframe_equal(df, utm, check_less_precise=True, check_crs=False)


def test_crs_axis_order__always_xy():
    df = GeoDataFrame(geometry=[Point(-1683723, 6689139)], crs="epsg:26918")
    lonlat = df.to_crs("epsg:4326")
    test_lonlat = GeoDataFrame(
        geometry=[Point(-110.1399901, 55.1350011)], crs="epsg:4326"
    )
    assert_geodataframe_equal(lonlat, test_lonlat, check_less_precise=True)


def test_skip_exact_same():
    df = df_epsg26918()
    utm = df.to_crs(df.crs)
    assert_geodataframe_equal(df, utm, check_less_precise=True)


# Test CRS on GeometryArray level
class TestGeometryArrayCRS:
    def setup_method(self):
        self.osgb = pyproj.CRS(27700)
        self.wgs = pyproj.CRS(4326)

        self.geoms = [Point(0, 0), Point(1, 1)]
        self.polys = [
            Polygon([(random.random(), random.random()) for i in range(3)])
            for _ in range(10)
        ]
        self.arr = from_shapely(self.polys, crs=27700)

    def test_array(self):
        arr = from_shapely(self.geoms)
        arr.crs = 27700
        assert arr.crs == self.osgb

        arr = from_shapely(self.geoms, crs=27700)
        assert arr.crs == self.osgb

        arr = GeometryArray(arr)
        assert arr.crs == self.osgb

        arr = GeometryArray(arr, crs=4326)
        assert arr.crs == self.wgs

    def test_series(self):
        s = GeoSeries(crs=27700)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb

        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb

        # manually change CRS
        s.crs = 4326
        assert s.crs == self.wgs
        assert s.values.crs == self.wgs

        s = GeoSeries(self.geoms, crs=27700)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb

        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(arr)
        assert s.crs == self.osgb
        assert s.values.crs == self.osgb

        with pytest.warns(FutureWarning):
            s = GeoSeries(arr, crs=4326)
        assert s.crs == self.osgb

    @pytest.mark.filterwarnings("ignore:Assigning CRS")
    def test_dataframe(self):
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame(geometry=arr)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        df = GeoDataFrame(geometry=s)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        # different passed CRS than array CRS is ignored
        with pytest.warns(FutureWarning):
            df = GeoDataFrame(geometry=s, crs=4326)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        with pytest.warns(FutureWarning):
            GeoDataFrame(geometry=s, crs=4326)
        with pytest.warns(FutureWarning):
            GeoDataFrame({"data": [1, 2], "geometry": s}, crs=4326)
        with pytest.warns(FutureWarning):
            GeoDataFrame(df, crs=4326).crs

        # manually change CRS
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        df = GeoDataFrame(geometry=s)
        df.crs = 4326
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs

        df = GeoDataFrame(self.geoms, columns=["geom"], crs=27700)
        assert df.crs == self.osgb
        df = df.set_geometry("geom")
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.geom.crs == self.osgb
        assert df.geom.values.crs == self.osgb

        df = GeoDataFrame(geometry=self.geoms, crs=27700)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        df = GeoDataFrame(crs=27700)
        df = df.set_geometry(self.geoms)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        # new geometry with set CRS has priority over GDF CRS
        df = GeoDataFrame(crs=27700)
        df = df.set_geometry(self.geoms, crs=4326)
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs

        df = GeoDataFrame()
        df = df.set_geometry(s)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame()
        df = df.set_geometry(arr)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        arr = from_shapely(self.geoms)
        df = GeoDataFrame({"col1": [1, 2], "geometry": arr}, crs=4326)
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs

        arr = from_shapely(self.geoms, crs=4326)
        df = GeoDataFrame({"col1": [1, 2], "geometry": arr})
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs

        # geometry column without geometry
        df = GeoDataFrame({"geometry": [0, 1]})
        df.crs = 27700
        assert df.crs == self.osgb

    @pytest.mark.parametrize(
        "scalar", [None, Point(0, 0), LineString([(0, 0), (1, 1)])]
    )
    def test_scalar(self, scalar):
        with pytest.warns(FutureWarning):
            df = GeoDataFrame()
            df.crs = 4326
        df["geometry"] = scalar
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs

    def test_read_file(self):
        nybb_filename = datasets.get_path("nybb")
        df = read_file(nybb_filename)
        assert df.crs == pyproj.CRS(2263)
        assert df.geometry.crs == pyproj.CRS(2263)
        assert df.geometry.values.crs == pyproj.CRS(2263)

    def test_multiple_geoms(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=["col1"])
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.col1.crs == self.wgs
        assert df.col1.values.crs == self.wgs

    def test_multiple_geoms_set_geom(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=["col1"])
        df = df.set_geometry("col1")
        assert df.crs == self.wgs
        assert df.geometry.crs == self.wgs
        assert df.geometry.values.crs == self.wgs
        assert df["geometry"].crs == self.osgb
        assert df["geometry"].values.crs == self.osgb

    def test_assign_cols(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=["col1"])
        df["geom2"] = s
        df["geom3"] = s.values
        df["geom4"] = from_shapely(self.geoms)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.geom2.crs == self.wgs
        assert df.geom2.values.crs == self.wgs
        assert df.geom3.crs == self.wgs
        assert df.geom3.values.crs == self.wgs
        assert df.geom4.crs is None
        assert df.geom4.values.crs is None

    def test_copy(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=["col1"])

        arr_copy = arr.copy()
        assert arr_copy.crs == arr.crs

        s_copy = s.copy()
        assert s_copy.crs == s.crs
        assert s_copy.values.crs == s.values.crs

        df_copy = df.copy()
        assert df_copy.crs == df.crs
        assert df_copy.geometry.crs == df.geometry.crs
        assert df_copy.geometry.values.crs == df.geometry.values.crs
        assert df_copy.col1.crs == df.col1.crs
        assert df_copy.col1.values.crs == df.col1.values.crs

    def test_rename(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=["col1"])
        df = df.rename(columns={"geometry": "geom"}).set_geometry("geom")
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        df = df.rename_geometry("geom2")
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        df = df.rename(columns={"col1": "column1"})
        assert df.column1.crs == self.wgs
        assert df.column1.values.crs == self.wgs

    def test_geoseries_to_crs(self):
        s = GeoSeries(self.geoms, crs=27700)
        s = s.to_crs(4326)
        assert s.crs == self.wgs
        assert s.values.crs == self.wgs

        df = GeoDataFrame(geometry=s)
        assert df.crs == self.wgs
        df = df.to_crs(27700)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        # make sure that only active geometry is transformed
        arr = from_shapely(self.geoms, crs=4326)
        df["col1"] = arr
        df = df.to_crs(3857)
        assert df.col1.crs == self.wgs
        assert df.col1.values.crs == self.wgs

    def test_array_to_crs(self):
        arr = from_shapely(self.geoms, crs=27700)
        arr = arr.to_crs(4326)
        assert arr.crs == self.wgs

    def test_from_shapely(self):
        arr = from_shapely(self.geoms, crs=27700)
        assert arr.crs == self.osgb

    def test_from_wkb(self):
        L_wkb = [p.wkb for p in self.geoms]
        arr = from_wkb(L_wkb, crs=27700)
        assert arr.crs == self.osgb

    def test_from_wkt(self):
        L_wkt = [p.wkt for p in self.geoms]
        arr = from_wkt(L_wkt, crs=27700)
        assert arr.crs == self.osgb

    def test_points_from_xy(self):
        df = pd.DataFrame([{"x": x, "y": x, "z": x} for x in range(10)])
        arr = points_from_xy(df["x"], df["y"], crs=27700)
        assert arr.crs == self.osgb

    # setting CRS in GeoSeries should not set it in passed array without CRS
    def test_original(self):
        arr = from_shapely(self.geoms)
        s = GeoSeries(arr, crs=27700)
        assert arr.crs is None
        assert s.crs == self.osgb

    def test_ops(self):
        arr = self.arr
        bound = arr.boundary
        assert bound.crs == self.osgb

        cent = arr.centroid
        assert cent.crs == self.osgb

        hull = arr.convex_hull
        assert hull.crs == self.osgb

        envelope = arr.envelope
        assert envelope.crs == self.osgb

        exterior = arr.exterior
        assert exterior.crs == self.osgb

        representative_point = arr.representative_point()
        assert representative_point.crs == self.osgb

    def test_binary_ops(self):
        arr = self.arr
        quads = []
        while len(quads) < 10:
            geom = Polygon([(random.random(), random.random()) for i in range(4)])
            if geom.is_valid:
                quads.append(geom)

        arr2 = from_shapely(quads, crs=27700)

        difference = arr.difference(arr2)
        assert difference.crs == self.osgb

        intersection = arr.intersection(arr2)
        assert intersection.crs == self.osgb

        symmetric_difference = arr.symmetric_difference(arr2)
        assert symmetric_difference.crs == self.osgb

        union = arr.union(arr2)
        assert union.crs == self.osgb

    def test_other(self):
        arr = self.arr

        buffer = arr.buffer(5)
        assert buffer.crs == self.osgb

        interpolate = arr.exterior.interpolate(0.1)
        assert interpolate.crs == self.osgb

        simplify = arr.simplify(5)
        assert simplify.crs == self.osgb

    @pytest.mark.parametrize(
        "attr, arg",
        [
            ("affine_transform", ([0, 1, 1, 0, 0, 0],)),
            ("translate", ()),
            ("rotate", (10,)),
            ("scale", ()),
            ("skew", ()),
        ],
    )
    def test_affinity_methods(self, attr, arg):
        result = getattr(self.arr, attr)(*arg)

        assert result.crs == self.osgb

    def test_slice(self):
        s = GeoSeries(self.arr, crs=27700)
        assert s.iloc[1:].values.crs == self.osgb

        df = GeoDataFrame({"col1": self.arr}, geometry=s)
        assert df.iloc[1:].geometry.values.crs == self.osgb
        assert df.iloc[1:].col1.values.crs == self.osgb

    def test_concat(self):
        s = GeoSeries(self.arr, crs=27700)
        assert pd.concat([s, s]).values.crs == self.osgb

        df = GeoDataFrame({"col1": from_shapely(self.geoms, crs=4326)}, geometry=s)
        assert pd.concat([df, df]).geometry.values.crs == self.osgb
        assert pd.concat([df, df]).col1.values.crs == self.wgs

    def test_merge(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame({"col1": s}, geometry=arr)
        df2 = GeoDataFrame({"col2": s}, geometry=arr).rename_geometry("geom")
        merged = df.merge(df2, left_index=True, right_index=True)
        assert merged.col1.values.crs == self.wgs
        assert merged.geometry.values.crs == self.osgb
        assert merged.col2.values.crs == self.wgs
        assert merged.geom.values.crs == self.osgb
        assert merged.crs == self.osgb

    # CRS should be assigned to geometry
    def test_deprecation(self):
        with pytest.warns(FutureWarning):
            df = GeoDataFrame([], crs=27700)

        # https://github.com/geopandas/geopandas/issues/1548
        # ensure we still have converted the crs value to a CRS object
        assert isinstance(df.crs, pyproj.CRS)

        with pytest.warns(FutureWarning):
            df = GeoDataFrame([])
            df.crs = 27700

        assert isinstance(df.crs, pyproj.CRS)

    # make sure that geometry column from list has CRS (__setitem__)
    def test_setitem_geometry(self):
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame({"col1": [0, 1]}, geometry=arr)

        df["geometry"] = [g for g in df.geometry]
        assert df.geometry.values.crs == self.osgb

        df2 = GeoDataFrame({"col1": [0, 1]}, geometry=arr)
        df2["geometry"] = from_shapely(self.geoms, crs=4326)
        assert df2.geometry.values.crs == self.wgs

    def test_astype(self):
        arr = from_shapely(self.geoms, crs=27700)
        df = GeoDataFrame({"col1": [0, 1]}, geometry=arr)
        df2 = df.astype({"col1": str})
        assert df2.crs == self.osgb

    def test_apply(self):
        s = GeoSeries(self.arr)
        assert s.crs == 27700

        # apply preserves the CRS if the result is a GeoSeries
        result = s.apply(lambda x: x.centroid)
        assert result.crs == 27700

    def test_apply_geodataframe(self):
        df = GeoDataFrame({"col1": [0, 1]}, geometry=self.geoms, crs=27700)
        assert df.crs == 27700

        # apply preserves the CRS if the result is a GeoDataFrame
        result = df.apply(lambda col: col, axis=0)
        assert result.crs == 27700
        result = df.apply(lambda row: row, axis=1)
        assert result.crs == 27700


class TestSetCRS:
    @pytest.mark.parametrize(
        "constructor",
        [
            lambda geoms, crs: GeoSeries(geoms, crs=crs),
            lambda geoms, crs: GeoDataFrame(geometry=geoms, crs=crs),
        ],
        ids=["geoseries", "geodataframe"],
    )
    def test_set_crs(self, constructor):
        naive = constructor([Point(0, 0), Point(1, 1)], crs=None)
        assert naive.crs is None

        # by default returns a copy
        result = naive.set_crs(crs="EPSG:4326")
        assert result.crs == "EPSG:4326"
        assert naive.crs is None

        result = naive.set_crs(epsg=4326)
        assert result.crs == "EPSG:4326"
        assert naive.crs is None

        # with inplace=True
        result = naive.set_crs(crs="EPSG:4326", inplace=True)
        assert result is naive
        assert result.crs == naive.crs == "EPSG:4326"

        # raise for non-naive when crs would be overridden
        non_naive = constructor([Point(0, 0), Point(1, 1)], crs="EPSG:4326")
        assert non_naive.crs == "EPSG:4326"
        with pytest.raises(ValueError, match="already has a CRS"):
            non_naive.set_crs("EPSG:3857")

        # allow for equal crs
        result = non_naive.set_crs("EPSG:4326")
        assert result.crs == "EPSG:4326"

        # replace with allow_override=True
        result = non_naive.set_crs("EPSG:3857", allow_override=True)
        assert non_naive.crs == "EPSG:4326"
        assert result.crs == "EPSG:3857"

        result = non_naive.set_crs("EPSG:3857", allow_override=True, inplace=True)
        assert non_naive.crs == "EPSG:3857"
        assert result.crs == "EPSG:3857"

        # raise error when no crs is passed
        with pytest.raises(ValueError):
            naive.set_crs(crs=None, epsg=None)
