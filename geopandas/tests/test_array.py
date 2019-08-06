import random

import numpy as np
import pandas as pd

import shapely
import shapely.geometry
from shapely.geometry.base import (CAP_STYLE, JOIN_STYLE)
import shapely.wkb
import shapely.affinity

import geopandas
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
    assert isinstance(points, GeometryArray)

    for i in range(10):
        assert isinstance(points[i], shapely.geometry.Point)
        assert points[i].x == x[i]
        assert points[i].y == y[i]


def test_points_from_xy():
    # testing the top-level interface

    # using DataFrame column
    df = pd.DataFrame([{'x': x, 'y': x, 'z': x} for x in range(10)])
    gs = [shapely.geometry.Point(x, x) for x in range(10)]
    gsz = [shapely.geometry.Point(x, x, x) for x in range(10)]
    geometry1 = geopandas.points_from_xy(df['x'], df['y'])
    geometry2 = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    assert geometry1 == gs
    assert geometry2 == gsz

    # using Series or numpy arrays or lists
    for s in [pd.Series(range(10)), np.arange(10), list(range(10))]:
        geometry1 = geopandas.points_from_xy(s, s)
        geometry2 = geopandas.points_from_xy(s, s, s)
        assert geometry1 == gs
        assert geometry2 == gsz

    # using different lengths should throw error
    arr_10 = np.arange(10)
    arr_20 = np.arange(20)
    with pytest.raises(ValueError):
        geopandas.points_from_xy(x=arr_10, y=arr_20)
        geopandas.points_from_xy(x=arr_10, y=arr_10, z=arr_20)

    # Using incomplete arguments should throw error
    with pytest.raises(TypeError):
        geopandas.points_from_xy(x=s)
        geopandas.points_from_xy(y=s)
        geopandas.points_from_xy(z=s)


def test_from_shapely():
    assert isinstance(T, GeometryArray)
    assert [v.equals(t) for v, t in zip(T, triangles)]


def test_from_shapely_geo_interface():

    class Point:

        def __init__(self, x, y):
            self.x = x
            self.y = y

        @property
        def __geo_interface__(self):
            return {'type': 'Point', 'coordinates': (self.x, self.y)}

    result = from_shapely([Point(1.0, 2.0), Point(3.0, 4.0)])
    expected = from_shapely([
        shapely.geometry.Point(1.0, 2.0), shapely.geometry.Point(3.0, 4.0)])
    assert all(v.equals(t) for v, t in zip(result, expected))


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


@pytest.mark.parametrize('attr,args', [
    ('contains', ()),
    ('covers', ()),
    ('crosses', ()),
    ('disjoint', ()),
    ('equals', ()),
    ('intersects', ()),
    ('overlaps', ()),
    ('touches', ()),
    ('within', ()),
    ('equals_exact', (0.1,))
])
def test_predicates_vector_scalar(attr, args):
    point = points[0]
    tri = triangles[0]

    for other in [point, tri]:
        result = getattr(T, attr)(other, *args)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        expected = [getattr(tri, attr)(other, *args) for tri in triangles]

        assert result.tolist() == expected


@pytest.mark.parametrize('attr,args', [
    ('contains', ()),
    ('covers', ()),
    ('crosses', ()),
    ('disjoint', ()),
    ('equals', ()),
    ('intersects', ()),
    ('overlaps', ()),
    ('touches', ()),
    ('within', ()),
    ('equals_exact', (0.1,))
])
def test_predicates_vector_vector(attr, args):
    A = [shapely.geometry.Polygon([(random.random(), random.random())
                                   for i in range(3)])
         for _ in range(100)]
    B = [shapely.geometry.Polygon([(random.random(), random.random())
                                   for i in range(3)])
         for _ in range(100)]

    vec_A = from_shapely(A)
    vec_B = from_shapely(B)

    result = getattr(vec_A, attr)(vec_B, *args)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool

    expected = [getattr(a, attr)(b, *args) for a, b in zip(A, B)]

    assert result.tolist() == expected


@pytest.mark.parametrize('attr', [
    'boundary',
    'centroid',
    'convex_hull',
    'envelope',
    'exterior',
    # 'interiors',
])
def test_unary_geo(attr):
    result = getattr(T, attr)
    expected = [getattr(t, attr) for t in triangles]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize('attr', [
    'representative_point',
])
def test_unary_geo_callable(attr):

    result = getattr(T, attr)()
    expected = [getattr(t, attr)() for t in triangles]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize('attr', [
    'difference',
    'symmetric_difference',
    'union',
    'intersection',
])
def test_binary_geo_vector(attr):
    quads = []
    while len(quads) < 10:
        geom = shapely.geometry.Polygon([(random.random(), random.random())
                                         for i in range(4)])
        if geom.is_valid:
            quads.append(geom)

    Q = from_shapely(quads)

    result = getattr(T, attr)(Q)
    expected = [getattr(t, attr)(q) for t, q in zip(triangles, quads)]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize('attr', [
    'difference',
    'symmetric_difference',
    'union',
    'intersection',
])
def test_binary_geo_scalar(attr):
    quads = []
    while len(quads) < 1:
        geom = shapely.geometry.Polygon([(random.random(), random.random())
                                        for i in range(4)])
        if geom.is_valid:
            quads.append(geom)

    q = quads[0]

    T = from_shapely(triangles)

    result = getattr(T, attr)(q)
    expected = [getattr(t, attr)(q) for t in triangles]

    assert all([a.equals(b) for a, b in zip(result, expected)])


@pytest.mark.parametrize('attr', [
    'is_closed',
    'is_valid',
    'is_empty',
    'is_simple',
    'has_z',
    'is_ring',
])
def test_unary_predicates(attr):
    result = getattr(T, attr)
    if attr == 'is_ring':
        expected = [getattr(t.exterior, attr) for t in triangles]
    else:
        expected = [getattr(t, attr) for t in triangles]
    assert result.tolist() == expected


@pytest.mark.parametrize('attr', ['area', 'length'])
def test_unary_float(attr):
    result = getattr(T, attr)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float
    expected = [getattr(tri, attr) for tri in triangles]

    assert result.tolist() == expected


def test_geom_types():
    cat = T.geom_type
    assert list(cat) == ['Polygon'] * len(T)


def test_geom_types_null_mixed():
    geoms = [shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
             None,
             shapely.geometry.Point(0, 1)]

    G = from_shapely(geoms)
    cat = G.geom_type

    assert list(cat) == ['Polygon', None, 'Point']


@pytest.mark.parametrize('attr', ['distance', 'relate'])
def test_binary_vector_vector(attr):
    result = getattr(P[:len(T)], attr)(T)
    expected = [getattr(p, attr)(t) for t, p in zip(triangles, points)]

    assert list(result) == expected


@pytest.mark.parametrize('attr', ['distance', 'relate'])
def test_binary_vector_scalar(attr):
    p = points[0]
    result = getattr(T, attr)(p)
    expected = [getattr(t, attr)(p) for t in triangles]

    assert list(result) == expected


@pytest.mark.parametrize('normalized', [True, False])
def test_project(normalized):
    lines = [shapely.geometry.LineString([(random.random(), random.random())
                                          for _ in range(2)])
             for _ in range(20)]
    L = from_shapely(lines)

    result = L.project(P, normalized=normalized)
    expected = [l.project(p, normalized=normalized)
                for p, l in zip(points, lines)]
    assert list(result) == expected


@pytest.mark.parametrize('cap_style', [CAP_STYLE.round, CAP_STYLE.square])
@pytest.mark.parametrize('join_style', [JOIN_STYLE.round, JOIN_STYLE.bevel])
@pytest.mark.parametrize('resolution', [16, 25])
def test_buffer(resolution, cap_style, join_style):
    expected = [p.buffer(0.1, resolution=resolution, cap_style=cap_style,
                         join_style=join_style)
                for p in points]
    result = P.buffer(0.1, resolution=resolution, cap_style=cap_style,
                      join_style=join_style)

    assert all(a.equals(b) for a, b in zip(expected, result))


def test_simplify():
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                          for i in range(3)]).buffer(10)
                 for _ in range(10)]
    T = from_shapely(triangles)

    result = T.simplify(1)
    expected = [t.simplify(1) for t in triangles]
    assert all(a.equals(b) for a, b in zip(expected, result))


def test_unary_union():
    geoms = [shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]),
             shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1)])]
    G = from_shapely(geoms)
    u = G.unary_union()

    expected = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert u.equals(expected)


@pytest.mark.parametrize('attr, arg', [
    ('affine_transform', ([0, 1, 1, 0, 0, 0], )),
    ('translate', ()),
    ('rotate', (10,)),
    ('scale', ()),
    ('skew', ()),
])
def test_affinity_methods(attr, arg):
    result = getattr(T, attr)(*arg)
    expected = [getattr(shapely.affinity, attr)(t, *arg) for t in triangles]
    assert all(a.equals(b) for a, b in zip(result, expected))


# def test_coords():
#     L = T.exterior.coords
#     assert L == [tuple(t.exterior.coords) for t in triangles]

def test_coords_x_y():
    result = P.x
    expected = [p.x for p in points]
    assert list(result) == expected

    result = P.y
    expected = [p.y for p in points]
    assert list(result) == expected


def test_bounds():
    result = T.bounds
    expected = [t.bounds for t in triangles]
    np.testing.assert_allclose(result, expected)


def test_getitem():
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    P = from_shapely(points)

    P2 = P[P.area > 0.3]
    assert isinstance(P2, GeometryArray)

    P3 = P[[1, 3, 5]]
    assert len(P3) == 3
    assert isinstance(P3, GeometryArray)
    assert [p.x for p in P3] == [1, 3, 5]

    P4 = P[1::2]
    assert len(P4) == 5
    assert isinstance(P3, GeometryArray)
    assert [p.x for p in P4] == [1, 3, 5, 7, 9]

    P5 = P[1]
    assert isinstance(P5, shapely.geometry.Point)
    assert P5.equals(points[1])


def test_dir():
    assert 'contains' in dir(P)
    assert 'data' in dir(P)


def test_chaining():
    assert T.contains(T.centroid).all()


def test_pickle():
    import pickle
    T2 = pickle.loads(pickle.dumps(T))
    # assert (T.data != T2.data).all()
    assert T.equals(T2).all()


def test_raise_on_bad_sizes():
    with pytest.raises(ValueError) as info:
        T.contains(P)

    assert "lengths" in str(info.value).lower()
    assert '10' in str(info.value)
    assert '20' in str(info.value)
