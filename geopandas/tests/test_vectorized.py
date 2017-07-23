
import time
import random
import shapely
from geopandas.vectorized import (VectorizedGeometry, points_from_xy,
        from_shapely)

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


@pytest.mark.parametrize('attr', ['contains', 'covers'])
def test_prepared_operations(attr):
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(100)]

    vec = from_shapely(triangles)

    point = shapely.geometry.Point(random.random(), random.random())
    result = getattr(vec, attr)(point)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool

    expected = [getattr(tri, attr)(point) for tri in triangles]

    assert result.tolist() == expected


def test_unary_geo_operations():
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(10)]

    vec = from_shapely(triangles)

    centroids = vec.centroid()

    assert [c == t.centroid for c, t in zip(vec, triangles)]


def test_selection():
    points = [shapely.geometry.Point(i, i) for i in range(10)]
    vec = from_shapely(points)
    vec2 = vec[vec.x % 2 == 0]

    assert len(vec2) == 5
    assert all(p.x % 2 == 0 for p in vec2)

    vec3 = vec[[1, 3, 5]]
    assert len(vec3) == 3
    assert [p.x for p in vec3] == [1, 3, 5]


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
