from distutils.version import LooseVersion

import numpy as np
import pandas as pd

import shapely
from shapely.geometry.base import BaseGeometry

from geopandas.vectorized import GeometryArray, from_shapely
from geopandas._block import GeometryBlock

import pytest


def test_block():
    a = np.array([shapely.geometry.Point(i, i) for i in range(10)],
                 dtype=object)
    ga = from_shapely(a)
    geom_block = GeometryBlock(ga, placement=slice(1, 2, 1))

    assert isinstance(geom_block.values, GeometryArray)


@pytest.fixture
def df():
    int_block = pd.core.internals.IntBlock(np.arange(5).reshape(1, -1),
                                           placement=slice(0, 1, 1))
    a = np.array([shapely.geometry.Point(i, i) for i in range(10)],
                 dtype=object)
    ga = from_shapely(a)
    geom_block = GeometryBlock(ga, placement=slice(1, 2, 1))
    blk_mgr = pd.core.internals.BlockManager([geom_block, int_block],
                                             [['A', 'geometry'], range(5)])
    return pd.DataFrame(blk_mgr)


@pytest.mark.cython
@pytest.mark.xfail(str(pd.__version__) < LooseVersion('0.21'),
                   reason="GEOPANDAS-CYTHON")
def test_repr(df):
    assert 'POINT' in repr(df)
    assert 'POINT' in repr(df['geometry'])


@pytest.mark.cython
@pytest.mark.xfail(reason="GEOPANDAS-CYTHON")
def test_repr_truncated(df):
    with pd.option_context('display.max_rows', 4):
        repr(df)


def test_accessing_scalar(df):
    assert isinstance(df.loc[0, 'geometry'], BaseGeometry)
    assert isinstance(df.iloc[0, 1], BaseGeometry)
    assert isinstance(df['geometry'][0], BaseGeometry)
    assert isinstance(df['geometry'].loc[0], BaseGeometry)
    assert isinstance(df['geometry'].iloc[0], BaseGeometry)


def test_iter(df):
    s = df['geometry']
    assert isinstance(list(s)[0], BaseGeometry)
    assert isinstance([i for i in s][0], BaseGeometry)
