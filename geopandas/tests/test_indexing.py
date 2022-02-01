import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point
import numpy as np
import pandas as pd


@pytest.fixture
def df():
    return GeoDataFrame(
        {
            "geometry": [Point(x, x) for x in range(3)],
            "geometry2": [Point(x, x) for x in range(3)],
            "geometry3": [Point(x, x) for x in range(3)],
            "value": [1, 2, 1],
            "value_nan": np.nan,
        }
    )


test_case_column_sets = [
    ["geometry"],
    ["geometry2"],
    ["geometry", "geometry2"],
    # non active geo col case
    ["geometry", "value"],
    ["geometry", "value_nan"],
    ["geometry2", "value"],
    ["geometry2", "value_nan"],
]


@pytest.mark.parametrize(
    "column_set",
    test_case_column_sets,
    ids=[", ".join(i) for i in test_case_column_sets],
)
def test_constructor_sliced_row_slices(df, column_set):
    # https://github.com/geopandas/geopandas/issues/2282
    print("TEST START")
    df_subset = df[column_set]
    assert isinstance(df_subset, pd.DataFrame)
    print("PRE LOC SUBSET")
    res = df_subset.loc[0]
    # row slices shouldn't be GeoSeries, even if they have a geometry col
    assert type(res) == pd.Series
    if "geometry" in column_set:
        assert not isinstance(res.geometry, pd.Series)
        assert res.geometry == Point(0, 0)


def test_constructor_sliced_column_slices(df):
    # Note loc doesn't used _constructor_sliced so it's not tested here
    geo_idx = df.columns.get_loc("geometry")
    sub = df.head(1)
    # column slices should be GeoSeries if of geometry type
    assert type(sub.iloc[:, geo_idx]) == GeoSeries
    sub = df.head(2)
    assert type(sub.iloc[:, geo_idx]) == GeoSeries

    # check iloc columns slices are pd.Series instead
    assert type(df.iloc[0, :]) == pd.Series


def test_constructor_sliced_in_pandas_methods(df):
    # constructor sliced is used in many places, checking a sample of non
    # geometry cases are sensible
    assert type(df.count()) == pd.Series
    # drop the secondary geometry columns as not hashable
    assert type(df.drop(columns=["geometry2", "geometry3"]).duplicated()) == pd.Series
    assert type(df.quantile()) == pd.Series
    assert type(df.memory_usage()) == pd.Series
