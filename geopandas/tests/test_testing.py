import pytest

import numpy as np
from shapely.geometry import Polygon, Point

from geopandas import GeoSeries, GeoDataFrame
from geopandas.testing import (
    assert_geoseries_equal, assert_geodataframe_equal)


s1 = GeoSeries([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])])
s2 = GeoSeries([Polygon([(0, 2), (0, 0), (2, 0), (2, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])])

df1 = GeoDataFrame({'col1': [1, 2], 'geometry': s1})
df2 = GeoDataFrame({'col1': [1, 2], 'geometry': s2})


def test_geoseries():
    assert_geoseries_equal(s1, s2)

    with pytest.raises(AssertionError):
        assert_geoseries_equal(s1, s2, check_less_precise=True)


def test_geodataframe():
    assert_geodataframe_equal(df1, df2)

    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2, check_less_precise=True)

    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2[['geometry', 'col1']])

    assert_geodataframe_equal(df1, df2[['geometry', 'col1']], check_like=True)

    df3 = df2.copy()
    df3.loc[0, 'col1'] = 10
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df3)


def test_equal_nans():
    s = GeoSeries([Point(0, 0), np.nan])
    assert_geoseries_equal(s, s.copy())
    assert_geoseries_equal(s, s.copy(), check_less_precise=True)


def test_no_crs():
    df1 = GeoDataFrame({'col1': [1, 2], 'geometry': s1}, crs=None)
    df2 = GeoDataFrame({'col1': [1, 2], 'geometry': s1}, crs={})
    assert_geodataframe_equal(df1, df2)
