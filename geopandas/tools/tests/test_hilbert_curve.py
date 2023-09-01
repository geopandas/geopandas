import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads

import geopandas

import pytest
from pandas.testing import assert_series_equal


def test_hilbert_distance():
    # test the actual Hilbert Code algorithm against some hardcoded values
    geoms = geopandas.GeoSeries.from_wkt(
        [
            "POINT (0 0)",
            "POINT (1 1)",
            "POINT (1 0)",
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        ]
    )
    result = geoms.hilbert_distance(total_bounds=(0, 0, 1, 1), level=2)
    assert result.tolist() == [0, 10, 15, 2]

    result = geoms.hilbert_distance(total_bounds=(0, 0, 1, 1), level=3)
    assert result.tolist() == [0, 42, 63, 10]

    result = geoms.hilbert_distance(total_bounds=(0, 0, 1, 1), level=16)
    assert result.tolist() == [0, 2863311530, 4294967295, 715827882]


@pytest.fixture
def geoseries_points():
    p1 = Point(1, 2)
    p2 = Point(2, 3)
    p3 = Point(3, 4)
    p4 = Point(4, 1)
    return geopandas.GeoSeries([p1, p2, p3, p4])


def test_hilbert_distance_level(geoseries_points):
    with pytest.raises(ValueError):
        geoseries_points.hilbert_distance(level=20)


def test_specified_total_bounds(geoseries_points):
    result = geoseries_points.hilbert_distance(
        total_bounds=geoseries_points.total_bounds
    )
    expected = geoseries_points.hilbert_distance()
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "empty",
    [
        None,
        loads("POLYGON EMPTY"),
    ],
)
def test_empty(geoseries_points, empty):
    s = geoseries_points
    s.iloc[-1] = empty
    with pytest.raises(
        ValueError, match="cannot be computed on a GeoSeries with empty"
    ):
        s.hilbert_distance()


def test_zero_width():
    # special case of all points on the same line -> avoid warnings because
    # of division by 0 and introducing NaN
    s = geopandas.GeoSeries([Point(0, 0), Point(0, 2), Point(0, 1)])
    with np.errstate(all="raise"):
        result = s.hilbert_distance()
    assert np.array(result).argsort().tolist() == [0, 2, 1]
