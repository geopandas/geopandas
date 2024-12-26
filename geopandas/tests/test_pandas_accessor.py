"""Ensure geopandas methods can be accessed via pandas 'geo' accessor."""

import re

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


@pytest.fixture
def s2():
    return pd.Series(
        [Point(x, y + 1) for x, y in zip(range(3), range(3))], dtype=GeometryDtype()
    )


def test_series_geo_buffer(s):
    """Ensure returned geometry values have the expected dtype."""
    got = s.geo.buffer(0.2)
    assert isinstance(got.dtype, GeometryDtype)

    # Double-check that the .geo accessor works on the result.
    radius = got.geo.minimum_bounding_radius()
    for row in radius.index:
        assert radius[row] >= 0.1999  # Allow for some rounding error.


def test_series_geo_distance(s, s2):
    got = s.geo.distance(s2)
    pandas.testing.assert_series_equal(
        got,
        pd.Series([1.0, 1.0, 1.0]),
    )


def test_series_geo_x(s):
    x = s.geo.x
    pandas.testing.assert_series_equal(
        x,
        pd.Series([0.0, 1.0, 2.0]),
    )


def test_series_geo_x_attributeerror_for_not_geo_dtype():
    s = pd.Series([1, 2, 3])

    with pytest.raises(
        AttributeError,
        match=re.escape("Can only use .geo accessor with GeometryDtype values"),
    ):
        s.geo.x


def test_series_geo_y(s):
    y = s.geo.y
    pandas.testing.assert_series_equal(
        y,
        pd.Series([0.0, 1.0, 2.0]),
    )
