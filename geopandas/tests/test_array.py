import random

import numpy as np

import shapely
import shapely.geometry
import shapely.wkb

from geopandas.array import (
    GeometryArray, points_from_xy, from_shapely, from_wkb, from_wkt, to_wkb,
    to_wkt)

import pytest
import six


triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                       for i in range(3)])
             for _ in range(10)]
T = from_shapely(triangles)

points = [shapely.geometry.Point(random.random(), random.random())
          for _ in range(20)]
P = from_shapely(points)


point = points[0]


def test_points():
    x = np.arange(10).astype(np.float)
    y = np.arange(10).astype(np.float) ** 2

    points = points_from_xy(x, y)

    for i in range(10):
        assert isinstance(points[i], shapely.geometry.Point)
        assert points[i].x == x[i]
        assert points[i].y == y[i]


def test_from_shapely():
    assert isinstance(T, GeometryArray)
    assert [v.equals(t) for v, t in zip(T, triangles)]


def test_from_wkb():
    # list
    L_wkb = [p.wkb for p in points]
    res = from_wkb(L_wkb)
    assert isinstance(res, GeometryArray)
    assert all(v.equals(t) for v, t in zip(res, points))

    # array
    res = from_wkb(np.array(L_wkb, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.equals(t) for v, t in zip(res, points))

    # missing values
    L_wkb.extend([b'', None])
    res = from_wkb(L_wkb)
    assert res[-1] is None
    assert res[-2] is None


def test_to_wkb():
    res = to_wkb(P)
    exp = np.array([p.wkb for p in points], dtype=object)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, exp)

    # missing values
    a = from_shapely([None, points[0]])
    res = to_wkb(a)
    assert res[0] is None


@pytest.mark.parametrize('string_type', ['str', 'bytes'])
def test_from_wkt(string_type):
    if string_type == 'str':
        f = six.text_type
    else:
        if six.PY3:
            def f(x): return bytes(x, 'utf8')
        else:
            def f(x): return x

    # list
    L_wkt = [f(p.wkt) for p in points]
    res = from_wkt(L_wkt)
    assert isinstance(res, GeometryArray)
    assert all(v.almost_equals(t) for v, t in zip(res, points))

    # array
    res = from_wkt(np.array(L_wkt, dtype=object))
    assert isinstance(res, GeometryArray)
    assert all(v.almost_equals(t) for v, t in zip(res, points))

    # missing values
    L_wkt.extend([f(''), None])
    res = from_wkt(L_wkt)
    assert res[-1] is None
    assert res[-2] is None


def test_to_wkt():
    res = to_wkt(P)
    exp = np.array([p.wkt for p in points], dtype=object)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, exp)

    # missing values
    a = from_shapely([None, points[0]])
    res = to_wkt(a)
    assert res[0] is None
