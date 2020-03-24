from distutils.version import LooseVersion

import numpy as np
import pandas as pd

from shapely.geometry import Point
import pyproj

from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt

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
def test_transform2(epsg4326, epsg26918):
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

    def test_array(self):
        arr = from_shapely(self.geoms)
        arr.crs = 27700
        assert arr.crs == self.osgb

        arr = from_shapely(self.geoms, crs=27700)
        assert arr.crs == self.osgb

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

    def test_dataframe(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(arr)
        df = GeoDataFrame(geometry=s)
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
        # TODO: raise warning?
        df = GeoDataFrame(geometry=s, crs=4326)
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb

        # manually change CRS
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

    def test_assing_cols(self):
        arr = from_shapely(self.geoms, crs=27700)
        s = GeoSeries(self.geoms, crs=4326)
        df = GeoDataFrame(s, geometry=arr, columns=["col1"])
        df["geom2"] = s
        df["geom3"] = s.values
        assert df.crs == self.osgb
        assert df.geometry.crs == self.osgb
        assert df.geometry.values.crs == self.osgb
        assert df.geom2.crs == self.wgs
        assert df.geom2.values.crs == self.wgs
        assert df.geom3.crs == self.wgs
        assert df.geom3.values.crs == self.wgs

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

    def test_to_crs(self):
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
