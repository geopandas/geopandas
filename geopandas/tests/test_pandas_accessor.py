"""Ensure geopandas methods can be accessed via pandas 'geo' accessor."""

import pandas as pd
from shapely.geometry import Point

# Use this import to register the "geo" accessor.
import geopandas.accessors  # noqa: F401 # pylint: disable=unused-import
from geopandas.array import GeometryDtype

import pandas.testing
import pytest


@pytest.fixture
def s():
    return pd.Series(
        [Point(x, y) for x, y in zip(range(3), range(3))], dtype=GeometryDtype()
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


def test_series_geo_distance(s):
    got = s.geo.distance(s)
    pandas.testing.assert_series_equal(
        got,
        pd.Series([0.0, 0.0, 0.0]),
    )
