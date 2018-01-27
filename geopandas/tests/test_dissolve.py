from __future__ import absolute_import

import pytest

import numpy as np
import pandas as pd

import geopandas
from geopandas import GeoDataFrame, read_file

from pandas.util.testing import assert_frame_equal

@pytest.fixture
def nybb_polydf():    
    nybb_filename = geopandas.datasets.get_path('nybb')
    nybb_polydf = read_file(nybb_filename)
    nybb_polydf = nybb_polydf[['geometry', 'BoroName', 'BoroCode']]
    nybb_polydf = nybb_polydf.rename(columns={'geometry': 'myshapes'})
    nybb_polydf = nybb_polydf.set_geometry('myshapes')
    nybb_polydf['manhattan_bronx'] = 5
    nybb_polydf.loc[3:4, 'manhattan_bronx'] = 6 
    return  nybb_polydf


@pytest.fixture
def merged_shapes(nybb_polydf):
    # Merged geometry
    manhattan_bronx = nybb_polydf.loc[3:4, ]
    others = nybb_polydf.loc[0:2, ]

    collapsed = [others.geometry.unary_union,
                 manhattan_bronx.geometry.unary_union]
    merged_shapes = GeoDataFrame(
        {'myshapes': collapsed}, geometry='myshapes',
        index=pd.Index([5, 6], name='manhattan_bronx'))

    return merged_shapes


@pytest.fixture
def first(merged_shapes):
    first = merged_shapes.copy()
    first['BoroName'] = ['Staten Island', 'Manhattan']
    first['BoroCode'] = [5, 1]
    return first


@pytest.fixture
def expected_mean(merged_shapes):
    test_mean = merged_shapes.copy()
    test_mean['BoroCode'] = [4, 1.5]
    return test_mean


def test_geom_dissolve(nybb_polydf, first):
    test = nybb_polydf.dissolve('manhattan_bronx')
    assert test.geometry.name == 'myshapes'
    assert test.geom_almost_equals(first).all()


def test_dissolve_retains_existing_crs(nybb_polydf):
    assert nybb_polydf.crs is not None
    test = nybb_polydf.dissolve('manhattan_bronx')
    assert test.crs is not None


def test_dissolve_retains_nonexisting_crs(nybb_polydf):
    nybb_polydf.crs = None
    test = nybb_polydf.dissolve('manhattan_bronx')
    assert test.crs is None


def first_dissolve(nybb_polydf, first):
    test = nybb_polydf.dissolve('manhattan_bronx')
    assert_frame_equal(first, test, check_column_type=False)


def test_mean_dissolve(nybb_polydf, first, expected_mean):
    test = nybb_polydf.dissolve('manhattan_bronx', aggfunc='mean')
    assert_frame_equal(expected_mean, test, check_column_type=False)

    test = nybb_polydf.dissolve('manhattan_bronx', aggfunc=np.mean)
    assert_frame_equal(expected_mean, test, check_column_type=False)


def test_multicolumn_dissolve(nybb_polydf, first):
    multi = nybb_polydf.copy()
    multi['dup_col'] = multi.manhattan_bronx
    multi_test = multi.dissolve(['manhattan_bronx', 'dup_col'],
                                aggfunc='first')

    first_copy = first.copy()
    first_copy['dup_col'] = first_copy.index
    first_copy = first_copy.set_index([first_copy.index, 'dup_col'])

    assert_frame_equal(multi_test, first_copy, check_column_type=False)


def test_reset_index(nybb_polydf, first):
    test = nybb_polydf.dissolve('manhattan_bronx', as_index=False)
    comparison = first.reset_index()
    assert_frame_equal(comparison, test, check_column_type=False)
