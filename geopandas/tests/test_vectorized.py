
import random
import shapely
from geopandas.vectorized import (VectorizedGeometry, points_from_xy,
        from_shapely)
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


def test_prepared_operations():
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(100)]

    vec = from_shapely(triangles)

    point = shapely.geometry.Point(random.random(), random.random())
    result = vec.covers(point)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool

    expected = [tri.covers(point) for tri in triangles]
    assert any(expected)

    assert result.tolist() == expected


def test_unary_geo_operations():
    triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                           for i in range(3)])
                 for _ in range(10)]

    vec = from_shapely(triangles)

    centroids = vec.centroid()

    assert [c == t.centroid for c, t in zip(vec, triangles)]
