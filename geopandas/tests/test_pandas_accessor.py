"""Ensure geopandas methods can be accessed via pandas 'geo' accessor."""

import numpy as np
import pandas as pd
import pandas.testing
import pytest
from shapely.geometry import Point

from geopandas.array import GeometryDtype


@pytest.fixture
def s():
    return pd.Series(
        [Point(x, y) for x, y in zip(range(3), range(3))], dtype=GeometryDtype()
    )


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "geometry": pd.Series(
                [Point(x, x) for x in range(3)], dtype=GeometryDtype()
            ),
            "value1": np.arange(3, dtype="int64"),
            "value2": np.array([1, 2, 1], dtype="int64"),
        }
    )


def test_series_geo_x(s):
    x = s.geo.x
    pandas.testing.assert_series_equal(
        x,
        pd.Series([0.0, 1.0, 2.0]),
    )


def test_series_geo_y(s):
    y = s.geo.y
    pandas.testing.assert_series_equal(
        y,
        pd.Series([0.0, 1.0, 2.0]),
    )


def test_dataframe_geo_covers(df, s):
    s = s.copy()
    s[1] = Point(7, 7)
    got = df.geo.covers(s)
    pandas.testing.assert_series_equal(
        got,
        pd.Series([True, False, True]),
    )
