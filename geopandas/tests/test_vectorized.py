
import time
import random
import shapely
from geopandas.vectorized import (VectorizedGeometry, points_from_xy,
        from_shapely)
from shapely.geometry.base import (CAP_STYLE, JOIN_STYLE)

import pytest
import numpy as np

def test_points():
    x = np.arange(10).astype(np.float)
    y = np.arange(10).astype(np.float) ** 2

    points = points_from_xy(x, y)
    assert (points.data != 0).all()

    assert (x == points.x).all()
    assert (y == points.y).all()

    assert isinstance(points[0], shapely.geometry.Point)


def test_from_shapely():
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(10)]

    vec = from_shapely(triangles)
    assert isinstance(vec, VectorizedGeometry)
    assert [v.equals(t) for v, t in zip(vec, triangles)]
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

    vec = from_shapely(triangles)

    point = shapely.geometry.Point(random.random(), random.random())
    tri = triangles[0]

    for other in [point, tri]:
        result = getattr(vec, attr)(other, *args)
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
#    'representative_point',
    'convex_hull',
    'envelope',
])
def test_unary_geo_operations(attr):
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(10)]

    vec = from_shapely(triangles)

    result = getattr(vec, attr)()
    expected = [getattr(t, attr) for t in triangles]

    assert [a.equals(b) for a, b in zip(result, expected)]


@pytest.mark.parametrize('attr', [
    pytest.mark.xfail('is_closed'),
    'is_valid',
    'is_empty',
    'is_simple',
    'has_z',
    'is_ring',
])
def test_unary_geo_operations(attr):
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(10)]

    vec = from_shapely(triangles)

    result = getattr(vec, attr)()
    expected = [getattr(t, attr) for t in triangles]

    assert result.tolist() == expected


def test_getitem():
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    vec = from_shapely(points)

    vec2 = vec[vec.x % 2 == 0]
    assert len(vec2) == 5
    assert all(p.x % 2 == 0 for p in vec2)

    vec3 = vec[[1, 3, 5]]
    assert len(vec3) == 3
    assert [p.x for p in vec3] == [1, 3, 5]

    vec4 = vec[1::2]
    assert len(vec4) == 5
    assert [p.x for p in vec4] == [1, 3, 5, 7, 9]


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
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    vec = from_shapely(points)

    assert 'contains' in dir(vec)
    assert 'data' in dir(vec)


def test_chaining():
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(10)]

    vec = from_shapely(triangles)

    assert vec.contains(vec.centroid()).all()


@pytest.mark.parametrize('resolution', [16, 25])
def test_buffer(resolution):
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    vec = from_shapely(points)

    expected = [p.buffer(0.1, resolution=resolution, cap_style=CAP_STYLE.round,
                         join_style=JOIN_STYLE.round)
                for p in points]
    result = vec.buffer(0.1, resolution=resolution, cap_style=CAP_STYLE.round,
                        join_style=JOIN_STYLE.round)

    assert all(a.equals(b) for a, b in zip(expected, result))


@pytest.mark.parametrize('attr', ['area', 'length'])
def test_vector_float(attr):
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(100)]

    vec = from_shapely(triangles)

    result = getattr(vec, attr)()
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float

    expected = [getattr(tri, attr) for tri in triangles]

    assert result.tolist() == expected
