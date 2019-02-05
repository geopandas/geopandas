from __future__ import absolute_import

from distutils.version import LooseVersion
import os

from six import PY3

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, GeoSeries
from geopandas.tests.util import assert_geoseries_equal

import pytest
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal, assert_series_equal


@pytest.fixture
def s():
    return GeoSeries([Point(x, y) for x, y in zip(range(3), range(3))])


@pytest.fixture
def df():
    return GeoDataFrame({'geometry': [Point(x, x) for x in range(3)],
                         'value1': np.arange(3, dtype='int64'),
                         'value2': np.array([1, 2, 1], dtype='int64')})


def test_repr(s, df):
    assert 'POINT' in repr(s)
    assert 'POINT' in repr(df)


def test_indexing(s, df):

    # accessing scalar from the geometry (colunm)
    exp = Point(1, 1)
    assert s[1] == exp
    assert s.loc[1] == exp
    assert s.iloc[1] == exp
    assert df.loc[1, 'geometry'] == exp
    assert df.iloc[1, 0] == exp

    # multiple values
    exp = GeoSeries([Point(2, 2), Point(0, 0)], index=[2, 0])
    assert_geoseries_equal(s.loc[[2, 0]], exp)
    assert_geoseries_equal(s.iloc[[2, 0]], exp)
    assert_geoseries_equal(s.reindex([2, 0]), exp)
    assert_geoseries_equal(df.loc[[2, 0], 'geometry'], exp)
    # TODO here iloc does not return a GeoSeries
    assert_series_equal(df.iloc[[2, 0], 0], exp, check_series_type=False,
                        check_names=False)

    # boolean indexing
    exp = GeoSeries([Point(0, 0), Point(2, 2)], index=[0, 2])
    mask = np.array([True, False, True])
    assert_geoseries_equal(s[mask], exp)
    assert_geoseries_equal(s.loc[mask], exp)
    assert_geoseries_equal(df[mask]['geometry'], exp)
    assert_geoseries_equal(df.loc[mask, 'geometry'], exp)


def test_assignment(s, df):
    exp = GeoSeries([Point(10, 10), Point(1, 1), Point(2, 2)])

    s2 = s.copy()
    s2[0] = Point(10, 10)
    assert_geoseries_equal(s2, exp)

    s2 = s.copy()
    s2.loc[0] = Point(10, 10)
    assert_geoseries_equal(s2, exp)

    s2 = s.copy()
    s2.iloc[0] = Point(10, 10)
    assert_geoseries_equal(s2, exp)

    df2 = df.copy()
    df2.loc[0, 'geometry'] = Point(10, 10)
    assert_geoseries_equal(df2['geometry'], exp)

    df2 = df.copy()
    df2.iloc[0, 0] = Point(10, 10)
    assert_geoseries_equal(df2['geometry'], exp)


def test_assign(df):
    res = df.assign(new=1)
    exp = df.copy()
    exp['new'] = 1
    assert isinstance(res, GeoDataFrame)
    assert_frame_equal(res, exp, )


def test_astype(s):

    with pytest.raises(TypeError):
        s.astype(int)

    assert s.astype(str)[0] == 'POINT (0 0)'


def test_to_csv(df):

    exp = ('geometry,value1,value2\nPOINT (0 0),0,1\nPOINT (1 1),1,2\n'
           'POINT (2 2),2,1\n').replace('\n', os.linesep)
    assert df.to_csv(index=False) == exp


@pytest.mark.skipif(str(pd.__version__) < LooseVersion('0.17'),
                    reason="s.max() does not raise on 0.16")
def test_numerical_operations(s, df):

    # df methods ignore the geometry column
    exp = pd.Series([3, 4], index=['value1', 'value2'])
    assert_series_equal(df.sum(), exp)

    # series methods raise error
    with pytest.raises(TypeError):
        s.sum()

    if PY3:
        # in python 2, objects are still orderable
        with pytest.raises(TypeError):
            s.max()

    with pytest.raises(TypeError):
        s.idxmax()

    # numerical ops raise an error
    with pytest.raises(TypeError):
        df + 1

    with pytest.raises(TypeError):
        s + 1

    # boolean comparisons work
    res = df == 100
    exp = pd.DataFrame(False, index=df.index, columns=df.columns)
    assert_frame_equal(res, exp)


def test_where(s):
    res = s.where(np.array([True, False, True]))
    exp = s.copy()
    exp[1] = np.nan
    assert_series_equal(res, exp)


def test_select_dtypes(df):
    res = df.select_dtypes(include=[np.number])
    exp = df[['value1', 'value2']]
    assert_frame_equal(res, exp)

# Missing values


@pytest.mark.xfail
def test_fillna():
    # this currently does not work (it seems to fill in the second coordinate
    # of the point
    s2 = GeoSeries([Point(0, 0), None, Point(2, 2)])
    res = s2.fillna(Point(1, 1))
    assert_geoseries_equal(res, s)


@pytest.mark.xfail
def test_dropna():
    # this currently does not work (doesn't drop)
    s2 = GeoSeries([Point(0, 0), None, Point(2, 2)])
    res = s2.dropna()
    exp = s2.loc[[0, 2]]
    assert_geoseries_equal(res, exp)


@pytest.mark.parametrize("NA", [None, np.nan, Point(), Polygon()])
def test_isna(NA):
    s2 = GeoSeries([Point(0, 0), NA, Point(2, 2)])
    exp = pd.Series([False, True, False])
    res = s2.isnull()
    assert_series_equal(res, exp)
    res = s2.isna()
    assert_series_equal(res, exp)
    res = s2.notnull()
    assert_series_equal(res, ~exp)
    res = s2.notna()
    assert_series_equal(res, ~exp)


# Groupby / algos


@pytest.mark.xfail
def test_unique():
    # this currently raises a TypeError
    s = GeoSeries([Point(0, 0), Point(0, 0), Point(2, 2)])
    exp = np.array([Point(0, 0), Point(2, 2)])
    assert_array_equal(s.unique(), exp)


@pytest.mark.xfail
def test_value_counts():
    # each object is considered unique
    s = GeoSeries([Point(0, 0), Point(1, 1), Point(0, 0)])
    res = s.value_counts()
    exp = pd.Series([2, 1], index=[Point(0, 0), Point(1, 1)])
    assert_series_equal(res, exp)


@pytest.mark.xfail
def test_drop_duplicates_series():
    # currently, geoseries with identical values are not recognized as
    # duplicates
    dups = GeoSeries([Point(0, 0), Point(0, 0)])
    dropped = dups.drop_duplicates()
    assert len(dropped) == 1


@pytest.mark.xfail
def test_drop_duplicates_frame():
    # currently, dropping duplicates in a geodataframe produces a TypeError
    # better behavior would be dropping the duplicated points
    gdf_len = 3
    dup_gdf = GeoDataFrame({'geometry': [Point(0, 0) for _ in range(gdf_len)],
                            'value1': range(gdf_len)})
    dropped_geometry = dup_gdf.drop_duplicates(subset="geometry")
    assert len(dropped_geometry) == 1
    dropped_all = dup_gdf.drop_duplicates()
    assert len(dropped_all) == gdf_len


def test_groupby(df):

    # counts work fine
    res = df.groupby('value2').count()
    exp = pd.DataFrame({'geometry': [2, 1], 'value1': [2, 1],
                        'value2': [1, 2]}).set_index('value2')
    assert_frame_equal(res, exp)

    # reductions ignore geometry column
    res = df.groupby('value2').sum()
    exp = pd.DataFrame({'value1': [2, 1],
                        'value2': [1, 2]}, dtype='int64').set_index('value2')
    assert_frame_equal(res, exp)

    # applying on the geometry column
    res = df.groupby('value2')['geometry'].apply(lambda x: x.cascaded_union)
    exp = pd.Series([shapely.geometry.MultiPoint([(0, 0), (2, 2)]),
                     Point(1, 1)],
                    index=pd.Index([1, 2], name='value2'), name='geometry')
    assert_series_equal(res, exp)


def test_groupby_groups(df):
    g = df.groupby('value2')
    res = g.get_group(1)
    assert isinstance(res, GeoDataFrame)
    exp = df.loc[[0, 2]]
    assert_frame_equal(res, exp)
