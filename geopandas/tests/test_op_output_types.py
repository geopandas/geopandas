import pandas as pd
import pyproj
import pytest
import geopandas._compat as compat

from shapely.geometry import Point
import numpy as np

from geopandas import GeoDataFrame, GeoSeries


crs_osgb = pyproj.CRS(27700)
crs_wgs = pyproj.CRS(4326)


N = 10


@pytest.fixture(params=["geometry", "point"])
def df(request):
    geo_name = request.param

    df = GeoDataFrame(
        [
            {
                "value1": x + y,
                "value2": x * y,
                geo_name: Point(x, y),  # rename this col in tests
            }
            for x, y in zip(range(N), range(N))
        ],
        crs=crs_wgs,
        geometry=geo_name,
    )
    # want geometry2 to be a GeoSeries not Series, test behaviour of non geom col
    df["geometry2"] = df[geo_name].set_crs(crs_osgb, allow_override=True)
    return df


@pytest.fixture
def df2():
    """For constructor_sliced tests"""
    return GeoDataFrame(
        {
            "geometry": GeoSeries([Point(x, x) for x in range(3)]),
            "geometry2": GeoSeries([Point(x, x) for x in range(3)]),
            "geometry3": GeoSeries([Point(x, x) for x in range(3)]),
            "value": [1, 2, 1],
            "value_nan": np.nan,
        }
    )


def _check_metadata_gdf(gdf, geo_name="geometry", crs=crs_wgs):
    assert gdf._geometry_column_name == geo_name
    assert gdf.geometry.name == geo_name
    assert gdf.crs == crs


def _check_metadata_gs(gs, name="geometry", crs=crs_wgs):
    assert gs.name == name
    assert gs.crs == crs


def assert_object(
    result, expected_type, geo_name="geometry", crs=crs_wgs, check_none_name=False
):
    """
    Helper method to make tests easier to read. Checks result is of the expected
    type. If result is a GeoDataFrame or GeoSeries, checks geo_name
    and crs match. If geo_name is None, then we expect a GeoDataFrame
    where the geometry column is invalid/ isn't set. This is never desirable,
    but is a reality of this first stage of implementation.
    """
    assert type(result) is expected_type

    if expected_type == GeoDataFrame:
        if geo_name is not None:
            _check_metadata_gdf(result, geo_name=geo_name, crs=crs)
        else:
            if check_none_name:  # TODO this is awkward
                assert result._geometry_column_name is None

            if result._geometry_column_name is None:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    "but the active"
                )
            else:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, but "
                    r"the active geometry column \("
                    rf"'{result._geometry_column_name}'\) is not present"
                )
            with pytest.raises(AttributeError, match=msg):
                result.geometry.name  # be explicit that geometry is invalid here
    elif expected_type == GeoSeries:
        _check_metadata_gs(result, name=geo_name, crs=crs)


def test_getitem(df):
    geo_name = df.geometry.name
    assert_object(df[["value1", "value2"]], pd.DataFrame)
    assert_object(df[[geo_name, "geometry2"]], GeoDataFrame, geo_name)
    assert_object(df[[geo_name]], GeoDataFrame, geo_name)
    assert_object(df[["geometry2", "value1"]], GeoDataFrame, None)
    assert_object(df[["geometry2"]], GeoDataFrame, None)
    assert_object(df[["value1"]], pd.DataFrame)
    # Series
    assert_object(df[geo_name], GeoSeries, geo_name)
    assert_object(df["geometry2"], GeoSeries, "geometry2", crs=crs_osgb)
    assert_object(df["value1"], pd.Series)


def test_loc(df):
    geo_name = df.geometry.name
    assert_object(df.loc[:, ["value1", "value2"]], pd.DataFrame)
    assert_object(df.loc[:, [geo_name, "geometry2"]], GeoDataFrame, geo_name)
    assert_object(df.loc[:, [geo_name]], GeoDataFrame, geo_name)
    assert_object(df.loc[:, ["geometry2", "value1"]], GeoDataFrame, None)
    assert_object(df.loc[:, ["geometry2"]], GeoDataFrame, None)
    assert_object(df.loc[:, ["value1"]], pd.DataFrame)
    # Series
    assert_object(df.loc[:, geo_name], GeoSeries, geo_name)
    assert_object(df.loc[:, "geometry2"], GeoSeries, "geometry2", crs=crs_osgb)
    assert_object(df.loc[:, "value1"], pd.Series)


def test_iloc(df):
    geo_name = df.geometry.name
    assert_object(df.iloc[:, 0:2], pd.DataFrame)
    assert_object(df.iloc[:, 2:4], GeoDataFrame, geo_name)
    assert_object(df.iloc[:, [2]], GeoDataFrame, geo_name)
    assert_object(df.iloc[:, [3, 0]], GeoDataFrame, None)
    assert_object(df.iloc[:, [3]], GeoDataFrame, None)
    assert_object(df.iloc[:, [0]], pd.DataFrame)
    # Series
    assert_object(df.iloc[:, 2], GeoSeries, geo_name)
    assert_object(df.iloc[:, 3], GeoSeries, "geometry2", crs=crs_osgb)
    assert_object(df.iloc[:, 0], pd.Series)


def test_squeeze(df):
    geo_name = df.geometry.name
    assert_object(df[[geo_name]].squeeze(), GeoSeries, geo_name)
    assert_object(df[["geometry2"]].squeeze(), GeoSeries, "geometry2", crs=crs_osgb)


def test_to_frame(df):
    geo_name = df.geometry.name
    res1 = df[geo_name].to_frame()
    assert_object(res1, GeoDataFrame, geo_name, crs=df[geo_name].crs)

    res2 = df["geometry2"].to_frame()
    assert_object(res2, GeoDataFrame, "geometry2", crs=crs_osgb)

    res3 = df["value1"].to_frame()
    assert_object(res3, pd.DataFrame)


def test_reindex(df):
    geo_name = df.geometry.name
    assert_object(df.reindex(columns=["value1", "value2"]), pd.DataFrame)
    assert_object(df.reindex(columns=[geo_name, "geometry2"]), GeoDataFrame, geo_name)
    assert_object(df.reindex(columns=[geo_name]), GeoDataFrame, geo_name)
    assert_object(df.reindex(columns=["new_col", geo_name]), GeoDataFrame, geo_name)
    assert_object(df.reindex(columns=["geometry2", "value1"]), GeoDataFrame, None)
    assert_object(df.reindex(columns=["geometry2"]), GeoDataFrame, None)
    assert_object(df.reindex(columns=["value1"]), pd.DataFrame)

    # reindexing the rows always preserves the GeoDataFrame
    assert_object(df.reindex(index=[0, 1, 20]), GeoDataFrame, geo_name)

    # reindexing both rows and columns
    assert_object(
        df.reindex(index=[0, 1, 20], columns=[geo_name]), GeoDataFrame, geo_name
    )
    assert_object(df.reindex(index=[0, 1, 20], columns=["value1"]), pd.DataFrame)


def test_drop(df):
    geo_name = df.geometry.name
    assert_object(df.drop(columns=[geo_name, "geometry2"]), pd.DataFrame)
    assert_object(df.drop(columns=["value1", "value2"]), GeoDataFrame, geo_name)
    cols = ["value1", "value2", "geometry2"]
    assert_object(df.drop(columns=cols), GeoDataFrame, geo_name)
    assert_object(df.drop(columns=[geo_name, "value2"]), GeoDataFrame, None)
    assert_object(df.drop(columns=["value1", "value2", geo_name]), GeoDataFrame, None)
    assert_object(df.drop(columns=["geometry2", "value2", geo_name]), pd.DataFrame)


def test_apply(df):
    geo_name = df.geometry.name

    def identity(x):
        return x

    # axis = 0
    assert_object(df[["value1", "value2"]].apply(identity), pd.DataFrame)
    assert_object(df[[geo_name, "geometry2"]].apply(identity), GeoDataFrame, geo_name)
    assert_object(df[[geo_name]].apply(identity), GeoDataFrame, geo_name)
    assert_object(df[["geometry2", "value1"]].apply(identity), GeoDataFrame, None, None)
    assert_object(df[["geometry2"]].apply(identity), GeoDataFrame, None, None)
    assert_object(df[["value1"]].apply(identity), pd.DataFrame)

    # axis = 0, Series
    assert_object(df[geo_name].apply(identity), GeoSeries, geo_name)
    assert_object(df["geometry2"].apply(identity), GeoSeries, "geometry2", crs=crs_osgb)
    assert_object(df["value1"].apply(identity), pd.Series)

    # axis = 0, Series, no longer geometry
    assert_object(df[geo_name].apply(lambda x: str(x)), pd.Series)
    assert_object(df["geometry2"].apply(lambda x: str(x)), pd.Series)

    # axis = 1
    assert_object(df[["value1", "value2"]].apply(identity, axis=1), pd.DataFrame)
    assert_object(
        df[[geo_name, "geometry2"]].apply(identity, axis=1), GeoDataFrame, geo_name
    )
    assert_object(df[[geo_name]].apply(identity, axis=1), GeoDataFrame, geo_name)
    # TODO below should be a GeoDataFrame to be consistent with new getitem logic
    #   leave as follow up as quite complicated
    #   FrameColumnApply.series_generator returns object dtypes Series, so will have
    #   patch result of apply
    assert_object(df[["geometry2", "value1"]].apply(identity, axis=1), pd.DataFrame)

    assert_object(df[["value1"]].apply(identity, axis=1), pd.DataFrame)


@pytest.mark.xfail(not compat.PANDAS_GE_11, reason="apply is different in pandas 1.0.5")
def test_apply_axis1_secondary_geo_cols(df):
    # note #GH2436 would also fix this
    def identity(x):
        return x

    assert_object(df[["geometry2"]].apply(identity, axis=1), GeoDataFrame, None, None)


def test_expanddim_in_apply():
    # https://github.com/geopandas/geopandas/pull/2296#issuecomment-1021966443
    s = GeoSeries.from_xy([0, 1], [0, 1])
    result = s.apply(lambda x: pd.Series([x.x, x.y]))
    assert_object(result, pd.DataFrame)


@pytest.mark.xfail(
    not compat.PANDAS_GE_11,
    reason="pandas <1.1 don't preserve subclass through groupby ops",  # Pandas GH33884
)
def test_expandim_in_groupby_aggregate_multiple_funcs():
    # https://github.com/geopandas/geopandas/pull/2296#issuecomment-1021966443
    # There are two calls to _constructor_expanddim here
    # SeriesGroupBy._aggregate_multiple_funcs() and
    # SeriesGroupBy._wrap_series_output() len(output) > 1

    s = GeoSeries.from_xy([0, 1, 2], [0, 1, 3])

    def union(s):
        return s.unary_union

    def total_area(s):
        return s.area.sum()

    grouped = s.groupby([0, 1, 0])
    agg = grouped.agg([total_area, union])
    assert_object(agg, GeoDataFrame, None, None, check_none_name=True)
    result = grouped.agg([union, total_area])
    assert_object(result, GeoDataFrame, None, None, check_none_name=True)
    assert_object(grouped.agg([total_area, total_area]), pd.DataFrame)
    assert_object(grouped.agg([total_area]), pd.DataFrame)


@pytest.mark.xfail(
    not compat.PANDAS_GE_11,
    reason="pandas <1.1 uses concat([Series]) in unstack",  # Pandas GH33356
)
def test_expanddim_in_unstack():
    # https://github.com/geopandas/geopandas/pull/2296#issuecomment-1021966443
    s = GeoSeries.from_xy(
        [0, 1, 2],
        [0, 1, 3],
        index=pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "a")]),
    )
    unstack = s.unstack()
    assert_object(unstack, GeoDataFrame, None, None, False)

    if compat.PANDAS_GE_12:
        assert unstack._geometry_column_name is None
    else:  # pandas GH37369, unstack doesn't call finalize
        assert unstack._geometry_column_name == "geometry"

    # https://github.com/geopandas/geopandas/issues/2486
    s.name = "geometry"
    unstack = s.unstack()
    assert_object(unstack, GeoDataFrame, None, None)


# indexing /  constructor_sliced tests

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
def test_constructor_sliced_row_slices(df2, column_set):
    # https://github.com/geopandas/geopandas/issues/2282
    df_subset = df2[column_set]
    assert isinstance(df_subset, GeoDataFrame)
    res = df_subset.loc[0]
    # row slices shouldn't be GeoSeries, even if they have a geometry col
    assert type(res) == pd.Series
    if "geometry" in column_set:
        assert not isinstance(res.geometry, pd.Series)
        assert res.geometry == Point(0, 0)


def test_constructor_sliced_column_slices(df2):
    # Note loc doesn't use _constructor_sliced so it's not tested here
    geo_idx = df2.columns.get_loc("geometry")
    sub = df2.head(1)
    # column slices should be GeoSeries if of geometry type
    assert type(sub.iloc[:, geo_idx]) == GeoSeries
    assert type(sub.iloc[[0], geo_idx]) == GeoSeries
    sub = df2.head(2)
    assert type(sub.iloc[:, geo_idx]) == GeoSeries
    assert type(sub.iloc[[0, 1], geo_idx]) == GeoSeries

    # check iloc row slices are pd.Series instead
    assert type(df2.iloc[0, :]) == pd.Series


def test_constructor_sliced_in_pandas_methods(df2):
    # constructor sliced is used in many places, checking a sample of non
    # geometry cases are sensible
    assert type(df2.count()) == pd.Series
    # drop the secondary geometry columns as not hashable
    hashable_test_df = df2.drop(columns=["geometry2", "geometry3"])
    assert type(hashable_test_df.duplicated()) == pd.Series
    assert type(df2.quantile()) == pd.Series
    assert type(df2.memory_usage()) == pd.Series
