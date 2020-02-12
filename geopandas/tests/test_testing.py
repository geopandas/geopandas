import numpy as np

from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, GeoSeries

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest

s1 = GeoSeries(
    [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
)
s2 = GeoSeries(
    [
        Polygon([(0, 2), (0, 0), (2, 0), (2, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
)

df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1})
df2 = GeoDataFrame({"col1": [1, 2], "geometry": s2})


def test_geoseries():
    assert_geoseries_equal(s1, s2)

    with pytest.raises(AssertionError):
        assert_geoseries_equal(s1, s2, check_less_precise=True)


def test_geodataframe():
    assert_geodataframe_equal(df1, df2)

    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2, check_less_precise=True)

    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2[["geometry", "col1"]])

    assert_geodataframe_equal(df1, df2[["geometry", "col1"]], check_like=True)

    df3 = df2.copy()
    df3.loc[0, "col1"] = 10
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df3)


def test_equal_nans():
    s = GeoSeries([Point(0, 0), np.nan])
    assert_geoseries_equal(s, s.copy())
    assert_geoseries_equal(s, s.copy(), check_less_precise=True)


def test_no_crs():
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs=None)
    df2 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs={})
    assert_geodataframe_equal(df1, df2)


def test_ignore_crs_mismatch():
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs="EPSG:4326")
    df2 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs="EPSG:31370")

    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2)

    # assert that with `check_crs=False` the assert passes, and also does not
    # generate any warning from comparing both geometries with different crs
    with pytest.warns(None) as record:
        assert_geodataframe_equal(df1, df2, check_crs=False)

    assert len(record) == 0
