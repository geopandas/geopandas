import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, GeoSeries
from geopandas._compat import HAS_PYPROJ
from geopandas.array import from_shapely

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

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


s3 = Series(
    [
        Polygon([(0, 2), (0, 0), (2, 0), (2, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
)

a = from_shapely(
    [
        Polygon([(0, 2), (0, 0), (2, 0), (2, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
)

s4 = Series(a)

df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1})
df2 = GeoDataFrame({"col1": [1, 2], "geometry": s2})

s4 = s1.copy()
s4.array.crs = 4326
s5 = s2.copy()
s5.array.crs = 27700

s6 = GeoSeries(
    [
        Polygon([(0, 3), (0, 0), (2, 0), (2, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
)

df4 = GeoDataFrame(
    {"col1": [1, 2], "geometry": s1.copy(), "geom2": s4.copy(), "geom3": s5.copy()},
    crs=3857,
)
df5 = GeoDataFrame(
    {"col1": [1, 2], "geometry": s1.copy(), "geom3": s5.copy(), "geom2": s4.copy()},
    crs=3857,
)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_geoseries():
    assert_geoseries_equal(s1, s2)
    assert_geoseries_equal(s1, s3, check_series_type=False, check_dtype=False)
    assert_geoseries_equal(s3, s2, check_series_type=False, check_dtype=False)
    assert_geoseries_equal(s1, s4, check_series_type=False)

    with pytest.raises(AssertionError) as error:
        assert_geoseries_equal(s1, s2, check_less_precise=True)
    assert "1 out of 2 geometries are not almost equal" in str(error.value)
    assert "not almost equal: [0]" in str(error.value)

    with pytest.raises(AssertionError) as error:
        assert_geoseries_equal(s2, s6, check_less_precise=False)
    assert "1 out of 2 geometries are not equal" in str(error.value)
    assert "not equal: [0]" in str(error.value)


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

    assert_geodataframe_equal(df5, df4, check_like=True)
    if HAS_PYPROJ:
        df5["geom2"] = df5.geom2.set_crs(3857, allow_override=True)
        with pytest.raises(AssertionError):
            assert_geodataframe_equal(df5, df4, check_like=True)


def test_equal_nans():
    s = GeoSeries([Point(0, 0), np.nan])
    assert_geoseries_equal(s, s.copy())
    assert_geoseries_equal(s, s.copy(), check_less_precise=True)


def test_no_crs():
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs=None)
    df2 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs={})
    assert_geodataframe_equal(df1, df2)


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
def test_ignore_crs_mismatch():
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1.copy()}, crs="EPSG:4326")
    df2 = GeoDataFrame({"col1": [1, 2], "geometry": s1}, crs="EPSG:31370")

    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2)

    # assert that with `check_crs=False` the assert passes, and also does not
    # generate any warning from comparing both geometries with different crs
    with warnings.catch_warnings(record=True) as record:
        assert_geodataframe_equal(df1, df2, check_crs=False)

    assert len(record) == 0


def test_almost_equal_but_not_equal():
    s_origin = GeoSeries([Point(0, 0)])
    s_almost_origin = GeoSeries([Point(0.0000001, 0)])
    assert_geoseries_equal(s_origin, s_almost_origin, check_less_precise=True)
    with pytest.raises(AssertionError):
        assert_geoseries_equal(s_origin, s_almost_origin)


def test_geodataframe_no_active_geometry_column():
    def create_dataframe():
        gdf = GeoDataFrame({"value": [1, 2], "geometry": [Point(1, 1), Point(2, 2)]})
        gdf["geom2"] = GeoSeries([Point(3, 3), Point(4, 4)])
        return gdf

    # no active geometry column (None)
    df1 = create_dataframe()
    df1._geometry_column_name = None
    df2 = create_dataframe()
    df2._geometry_column_name = None
    assert_geodataframe_equal(df1, df2)

    # active geometry column ("geometry") not present
    df1 = create_dataframe()[["value", "geom2"]]
    df2 = create_dataframe()[["value", "geom2"]]
    assert_geodataframe_equal(df1, df2)

    df1 = GeoDataFrame(create_dataframe()[["value"]])
    df2 = GeoDataFrame(create_dataframe()[["value"]])
    assert_geodataframe_equal(df1, df2)


def test_geodataframe_multiindex():
    def create_dataframe():
        gdf = DataFrame([[Point(0, 0), Point(1, 1)], [Point(2, 2), Point(3, 3)]])
        gdf = GeoDataFrame(gdf.astype("geometry"))
        gdf.columns = pd.MultiIndex.from_product([["geometry"], [0, 1]])
        return gdf

    df1 = create_dataframe()
    df2 = create_dataframe()
    assert_geodataframe_equal(df1, df2)

    df1 = create_dataframe()
    df1._geometry_column_name = None
    df2 = create_dataframe()
    df2._geometry_column_name = None
    assert_geodataframe_equal(df1, df2)
