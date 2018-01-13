from __future__ import absolute_import

import random
import string

import numpy as np

from shapely.geometry import Point, Polygon

import geopandas as gpd
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
            left_geom = left_row['geometry']
            right_geom = right_row['geometry']
            if left_geom and right_geom and getattr(left_row['geometry'], op)(right_row['geometry']):
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
                assert any(t2 and t.equals(t2) for t2 in L)

    if how == 'right':
        assert len(result) >= len(right_out)
        assert set(result.columns) == set(columns + ['index_left'])
        L = list(result.geometry)
        for p in points2:
            if p:
                assert any(p2 and p.equals(p2) for p2 in L)


def test_crs_mismatch():
    left = gpd.GeoDataFrame({'geometry': triangles,
                             'x': np.random.random(len(triangles)),
                             'y': np.random.random(len(triangles))},
                            crs={'init': 'epsg:4326'})

    right = gpd.GeoDataFrame({'geometry': points,
                              'x': np.random.random(len(points)),
                              'z': np.random.random(len(points))})

    with pytest.warns(UserWarning):
        sjoin(left, right)


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


@pytest.mark.parametrize('how', ['left', 'right', 'inner'])
def test_sjoin_named_index(how):

    left = gpd.GeoDataFrame({'geometry': triangles,
                             'x': np.random.random(len(triangles)),
                             'y': np.random.random(len(triangles))})

    right = gpd.GeoDataFrame({'geometry': points,
                              'x': np.random.random(len(points)),
                              'z': np.random.random(len(points))})

    # original index names should be unchanged
    right2 = right.copy()
    right2.index.name = 'pointid'
    _ = sjoin(right2, left, how=how)
    assert right2.index.name == 'pointid'
    assert right.index.name is None


@pytest.mark.skipif(not gpd.base.HAS_SINDEX, reason='Rtree absent, skipping')
class TestSpatialJoinNaturalEarth:

    def setup_method(self):
        world_path = gpd.datasets.get_path("naturalearth_lowres")
        cities_path = gpd.datasets.get_path("naturalearth_cities")
        self.world = gpd.read_file(world_path)
        self.cities = gpd.read_file(cities_path)

    def test_sjoin_inner(self):
        # GH637
        countries = self.world[["geometry", "name"]]
        countries = countries.rename(columns={"name": "country"})
        cities_with_country = sjoin(self.cities, countries, how="inner",
                                    op="intersects")
        assert cities_with_country.shape == (172, 4)
