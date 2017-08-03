
import random
import shapely.geometry

import numpy as np

import geopandas as gpd
from geopandas.vectorized import GeometryArray, from_shapely


triangles = [shapely.geometry.Polygon([(random.random(), random.random())
                                       for i in range(3)])
             for _ in range(10)]

points = [shapely.geometry.Point(random.random(), random.random())
          for _ in range(20)]

point = points[0]


def test_shapely_coercion():
    s = gpd.GeoSeries(triangles)
    assert s.values.dtype == object
    assert isinstance(s.iloc[0], shapely.geometry.base.BaseGeometry)


def test_basic():
    vec = from_shapely(triangles)
    s = gpd.GeoSeries(vec)

    assert s.contains(point).tolist() == [t.contains(point) for t in triangles]


def test_iterate_to_shapely():
    s = gpd.GeoSeries(points)
    assert s[3].equals(points[3])

    assert [a.equals(b) for a, b in zip(s, points)]


def test_buffer():
    s = gpd.GeoSeries(points)
    s2 = s.buffer(distance=10, resolution=24)

    assert isinstance(s2, gpd.GeoSeries)

    assert all(a.equals(b.buffer(distance=10, resolution=24))
               for a, b in zip(s2, points))

