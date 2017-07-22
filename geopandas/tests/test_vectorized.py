
import shapely
from geopandas.vectorized import VectorizedGeometry, points_from_xy
import numpy as np

def test_points():
    x = np.arange(10).astype(np.float)
    y = np.arange(10).astype(np.float) ** 2

    points = points_from_xy(x, y)
    assert (points.data != 0).all()

    assert (x == points.x).all()
    assert (y == points.y).all()

    assert isinstance(points[0], shapely.geometry.Point)
