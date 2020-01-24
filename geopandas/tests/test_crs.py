from distutils.version import LooseVersion

import numpy as np

from shapely.geometry import Point
import pyproj

from geopandas import GeoDataFrame, points_from_xy

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
