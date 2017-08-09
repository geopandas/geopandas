from __future__ import absolute_import

import random
import string

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from geopandas.tools.sjoin import sjoin

import pytest


triangles = [Polygon([(random.random(), random.random())
                     for i in range(3)])
             for _ in range(10)]


points = [Point(random.random(), random.random())
          for _ in range(20)]


@pytest.mark.parametrize('op', ['contains', 'intersects', 'covers', 'within'])
@pytest.mark.parametrize('lsuffix', ['left'])
@pytest.mark.parametrize('rsuffix', ['r'])
@pytest.mark.parametrize('how', ['inner', 'left', 'right'])
@pytest.mark.parametrize('missing', [True, False])
def test_sjoin(op, lsuffix, rsuffix, how, missing):
    triangles2 = list(triangles)
    if missing:
        for i in range(len(triangles)):
            if random.random() < 0.2:
                triangles2[i] = None

    points2 = list(points)
    if missing:
        for i in range(len(points)):
            if random.random() < 0.2:
                points2[i] = None

    left = gpd.GeoDataFrame({'geometry': triangles2,
                             'x': np.random.random(len(triangles)),
                             'y': np.random.random(len(triangles))},
                            index=np.arange(len(triangles)) * 2)

    right = gpd.GeoDataFrame({'geometry': points2,
                              'x': np.random.random(len(points)),
                              'z': np.random.random(len(points))},
                             index=list(string.ascii_lowercase[:len(points)]))

    result = sjoin(left, right, op=op, rsuffix=rsuffix, lsuffix=lsuffix,
                   how=how)

    left_out = []
    right_out = []
    left_touched = set()
    right_touched = set()
    for left_index, left_row in left.iterrows():
        for right_index, right_row in right.iterrows():
            if getattr(left_row['geometry'], op)(right_row['geometry']):
                left_out.append(left_index)
                right_out.append(right_index)

    columns = ['y', 'z', 'x_' + lsuffix, 'x_' + rsuffix, 'geometry']

    if how == 'inner':
        assert len(left_out) == len(result)
        assert set(result.columns) == set(columns + ['index_right'])

    if how == 'left':
        assert len(result) >= len(left_out)
        assert set(result.columns) == set(columns + ['index_right'])
        L = list(result.geometry)
        for t in triangles2:
            if t:
                assert any(t.equals(t2) for t2 in L)

    if how == 'right':
        assert len(result) >= len(right_out)
        assert set(result.columns) == set(columns + ['index_left'])
        L = list(result.geometry)
        for p in points2:
            if p:
                assert any(p.equals(p2) for p2 in L)


def test_errors():
    left = gpd.GeoDataFrame({'geometry': triangles,
                             'x': np.random.random(len(triangles)),
                             'y': np.random.random(len(triangles))},
                             index=np.arange(len(triangles)) * 2)

    right = gpd.GeoDataFrame({'geometry': points,
                              'x': np.random.random(len(points)),
                              'z': np.random.random(len(points))},
                             index=list(string.ascii_lowercase[:len(points)]))

    with pytest.raises(ValueError) as info:
        result = sjoin(left, right, how="both")

    assert "both" in str(info.value)
    assert "inner" in str(info.value)


@pytest.mark.parametrize('l', [0, 1, 5])
@pytest.mark.parametrize('r', [0, 1, 5])
def test_small(l, r):
    left = gpd.GeoDataFrame({'geometry': triangles,
                             'x': np.random.random(len(triangles)),
                             'y': np.random.random(len(triangles))},
                             index=np.arange(len(triangles)) * 2).iloc[:l]

    right = gpd.GeoDataFrame({'geometry': points,
                              'x': np.random.random(len(points)),
                              'z': np.random.random(len(points))},
                              index=list(string.ascii_lowercase[:len(points)])).iloc[:r]

    result = sjoin(left, right)
