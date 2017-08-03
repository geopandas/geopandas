
import time
import random
import shapely
from geopandas.vectorized import (GeometryArray, points_from_xy,
        from_shapely, serialize, deserialize, cysjoin)
from shapely.geometry.base import (CAP_STYLE, JOIN_STYLE)

import pytest
import numpy as np


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
    assert (points.data != 0).all()

    assert (x == points.x).all()
    assert (y == points.y).all()

    assert isinstance(points[0], shapely.geometry.Point)


def test_from_shapely():
    assert isinstance(T, GeometryArray)
    assert [v.equals(t) for v, t in zip(T, triangles)]
    # TODO: handle gc


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
def test_vector_scalar_predicates(attr, args):
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(100)]

    T = from_shapely(triangles)

    point = shapely.geometry.Point(random.random(), random.random())
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
def test_vector_vector_predicates(attr, args):
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
])
def test_unary_geo(attr):

    result = getattr(T, attr)()
    expected = [getattr(t, attr) for t in triangles]

    assert [a.equals(b) for a, b in zip(result, expected)]

@pytest.mark.parametrize('attr', [
    'representative_point',
])
def test_unary_geo_callable(attr):

    result = getattr(T, attr)()
    expected = [getattr(t, attr)() for t in triangles]

    assert [a.equals(b) for a, b in zip(result, expected)]

@pytest.mark.parametrize('attr', [
    'difference',
    'symmetric_difference',
    'union',
    'intersection',
])
def test_vector_binary_geo(attr):
    quads = []
    while len(quads) < 10:
        geom = shapely.geometry.Polygon([(random.random(), random.random())
                                        for i in range(4)])
        if geom.is_valid:
            quads.append(geom)

    Q = from_shapely(quads)

    result = getattr(T, attr)(Q)
    expected = [getattr(t, attr)(q) for t, q in zip(triangles, quads)]

    assert [a.equals(b) for a, b in zip(result, expected)]


@pytest.mark.parametrize('attr', [
    'difference',
    'symmetric_difference',
    'union',
    'intersection',
])
def test_binary_geo(attr):
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

    assert [a.equals(b) for a, b in zip(result, expected)]


@pytest.mark.parametrize('attr', [
    pytest.mark.xfail('is_closed'),
    'is_valid',
    'is_empty',
    'is_simple',
    'has_z',
    'is_ring',
])
def test_unary_predicates(attr):
    result = getattr(T, attr)()
    expected = [getattr(t, attr) for t in triangles]

    assert result.tolist() == expected


def test_getitem():
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    P = from_shapely(points)

    P2 = P[P.x % 2 == 0]
    assert len(P2) == 5
    assert isinstance(P2, GeometryArray)
    assert all(p.x % 2 == 0 for p in P2)

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


@pytest.mark.xfail(reason="We don't yet clean up memory well")
def test_clean_up_on_gc():
    psutil = pytest.importorskip('psutil')
    proc = psutil.Process()
    mem_1 = proc.memory_info().rss
    x = np.random.random(1000000)
    y = np.random.random(1000000)
    nb = x.nbytes
    mem_2 = proc.memory_info().rss - mem_1
    assert mem_2 >= x.nbytes + y.nbytes

    points = points_from_xy(x, y)
    mem_3 = proc.memory_info().rss - mem_1
    assert mem_3 >= (x.nbytes + y.nbytes) * 2

    del x, y
    import gc; gc.collect()
    mem_4 = proc.memory_info().rss - mem_1
    assert mem_4 < mem_3

    points2 = points[::2]
    mem_5 = proc.memory_info().rss - mem_1
    assert mem_5 < mem_4 + nb  # at most a small increase

    del points
    import gc; gc.collect()
    mem_6 = proc.memory_info().rss - mem_1
    assert mem_6 < mem_5 + nb  # still holding onto most of the data

    del points2
    import gc; gc.collect()
    time.sleep(0.1)
    mem_7 = proc.memory_info().rss - mem_1
    assert mem_7 <= mem_5
    assert proc.memory_info().rss - mem_1 < nb


def test_dir():
    assert 'contains' in dir(P)
    assert 'data' in dir(P)


def test_chaining():
    assert T.contains(T.centroid()).all()


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


@pytest.mark.parametrize('attr', ['area', 'length'])
def test_vector_float(attr):
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(100)]

    T = from_shapely(triangles)

    result = getattr(T, attr)()
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float

    expected = [getattr(tri, attr) for tri in triangles]

    assert result.tolist() == expected


def test_serialize_deserialize():
    ba, sizes = serialize(T.data)
    vec2 = GeometryArray(deserialize(ba, sizes))

    assert (T.data != vec2.data).all()
    assert T.equals(vec2).all()


def test_pickle():
    import pickle
    T2 = pickle.loads(pickle.dumps(T))

    assert (T.data != T2.data).all()
    assert T.equals(T2).all()


@pytest.mark.parametrize('predicate', [
    'contains',
    'covers',
    'crosses',
    'disjoint',
    # 'equals',
    'intersects',
    'overlaps',
    'touches',
    'within',
])
def test_sjoin(predicate):
    result = cysjoin(T.data, P.data, predicate)

    assert isinstance(result, np.ndarray)
    assert result.dtype == T.data.dtype
    n, m = result.shape
    assert m == 2
    assert n < (len(T) * len(P))

    for i, j in result:
        left = triangles[i]
        right = points[j]
        assert getattr(left, predicate)(right)


@pytest.mark.skip
def test_bench_sjoin():
    last = time.time()
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(1000)]

    points = points_from_xy(np.random.random(10000),
                            np.random.random(10000))
    print("creation", time.time() - last); last = time.time()

    T = from_shapely(triangles)
    P = from_shapely(points)

    print("vectorize", time.time() - last); last = time.time()

    result = cysjoin(T.data, P.data, 'intersects')

    print("join", time.time() - last); last = time.time()


def test_raise_on_bad_sizes():
    with pytest.raises(ValueError) as info:
        T.contains(P)

    assert "shape" in str(info.value).lower()
    assert '10' in str(info.value)
    assert '20' in str(info.value)
