import os

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing

import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest


@pytest.fixture
def s():
    return GeoSeries([Point(x, y) for x, y in zip(range(3), range(3))])


@pytest.fixture
def df():
    return GeoDataFrame(
        {
            "geometry": [Point(x, x) for x in range(3)],
            "value1": np.arange(3, dtype="int64"),
            "value2": np.array([1, 2, 1], dtype="int64"),
        }
    )


def test_repr(s, df):
    assert "POINT" in repr(s)
    assert "POINT" in repr(df)
    assert "POINT" in df._repr_html_()


def test_repr_boxed_display_precision():
    # geographic coordinates
    p1 = Point(10.123456789, 50.123456789)
    p2 = Point(4.123456789, 20.123456789)
    s1 = GeoSeries([p1, p2, None])
    assert "POINT (10.12346 50.12346)" in repr(s1)

    # geographic coordinates 4326
    s3 = GeoSeries([p1, p2], crs=4326)
    assert "POINT (10.12346 50.12346)" in repr(s3)

    # projected coordinates
    p1 = Point(3000.123456789, 3000.123456789)
    p2 = Point(4000.123456789, 4000.123456789)
    s2 = GeoSeries([p1, p2, None])
    assert "POINT (3000.123 3000.123)" in repr(s2)

    # projected geographic coordinate
    s4 = GeoSeries([p1, p2], crs=3857)
    assert "POINT (3000.123 3000.123)" in repr(s4)

    geopandas.options.display_precision = 1
    assert "POINT (10.1 50.1)" in repr(s1)

    geopandas.options.display_precision = 9
    assert "POINT (10.123456789 50.123456789)" in repr(s1)


def test_repr_all_missing():
    # https://github.com/geopandas/geopandas/issues/1195
    s = GeoSeries([None, None, None])
    assert "None" in repr(s)
    df = GeoDataFrame({"a": [1, 2, 3], "geometry": s})
    assert "None" in repr(df)
    assert "geometry" in df._repr_html_()


def test_repr_empty():
    # https://github.com/geopandas/geopandas/issues/1195
    s = GeoSeries([])
    assert repr(s) == "GeoSeries([], dtype: geometry)"
    df = GeoDataFrame({"a": [], "geometry": s})
    assert "Empty GeoDataFrame" in repr(df)
    # https://github.com/geopandas/geopandas/issues/1184
    assert "geometry" in df._repr_html_()


def test_repr_linearring():
    # https://github.com/geopandas/geopandas/pull/2689
    # specifically, checking internal shapely/pygeos/wkt/wkb conversions
    # preserve LinearRing
    s = GeoSeries([LinearRing([(0, 0), (1, 1), (1, -1)])])
    assert "LINEARRING" in str(s.iloc[0])  # shapely scalar repr
    assert "LINEARRING" in str(s)  # GeoSeries repr

    # check something coercible to linearring is not converted
    s2 = GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, -1)]),
            LineString([(0, 0), (1, 1), (1, -1), (0, 0)]),
        ]
    )
    assert "LINEARRING" not in str(s2)


def test_indexing(s, df):
    # accessing scalar from the geometry (column)
    exp = Point(1, 1)
    assert s[1] == exp
    assert s.loc[1] == exp
    assert s.iloc[1] == exp
    assert df.loc[1, "geometry"] == exp
    assert df.iloc[1, 0] == exp

    # multiple values
    exp = GeoSeries([Point(2, 2), Point(0, 0)], index=[2, 0])
    assert_geoseries_equal(s.loc[[2, 0]], exp)
    assert_geoseries_equal(s.iloc[[2, 0]], exp)
    assert_geoseries_equal(s.reindex([2, 0]), exp)
    assert_geoseries_equal(df.loc[[2, 0], "geometry"], exp)
    # TODO here iloc does not return a GeoSeries
    assert_series_equal(
        df.iloc[[2, 0], 0], exp, check_series_type=False, check_names=False
    )

    # boolean indexing
    exp = GeoSeries([Point(0, 0), Point(2, 2)], index=[0, 2])
    mask = np.array([True, False, True])
    assert_geoseries_equal(s[mask], exp)
    assert_geoseries_equal(s.loc[mask], exp)
    assert_geoseries_equal(df[mask]["geometry"], exp)
    assert_geoseries_equal(df.loc[mask, "geometry"], exp)

    # slices
    s.index = [1, 2, 3]
    exp = GeoSeries([Point(1, 1), Point(2, 2)], index=[2, 3])
    assert_series_equal(s[1:], exp)
    assert_series_equal(s.iloc[1:], exp)
    assert_series_equal(s.loc[2:], exp)


def test_reindex(s, df):
    # GeoSeries reindex
    res = s.reindex([1, 2, 3])
    exp = GeoSeries([Point(1, 1), Point(2, 2), None], index=[1, 2, 3])
    assert_geoseries_equal(res, exp)

    # GeoDataFrame reindex index
    res = df.reindex(index=[1, 2, 3])
    assert_geoseries_equal(res.geometry, exp)

    # GeoDataFrame reindex columns
    res = df.reindex(columns=["value1", "geometry"])
    assert isinstance(res, GeoDataFrame)
    assert isinstance(res.geometry, GeoSeries)
    assert_frame_equal(res, df[["value1", "geometry"]])

    res = df.reindex(columns=["value1", "value2"])
    assert type(res) == pd.DataFrame
    assert_frame_equal(res, df[["value1", "value2"]])


def test_take(s, df):
    inds = np.array([0, 2])

    # GeoSeries take
    result = s.take(inds)
    expected = s.iloc[[0, 2]]
    assert isinstance(result, GeoSeries)
    assert_geoseries_equal(result, expected)

    # GeoDataFrame take axis 0
    result = df.take(inds, axis=0)
    expected = df.iloc[[0, 2], :]
    assert isinstance(result, GeoDataFrame)
    assert_geodataframe_equal(result, expected)

    # GeoDataFrame take axis 1
    df = df.reindex(columns=["value1", "value2", "geometry"])  # ensure consistent order
    result = df.take(inds, axis=1)
    expected = df[["value1", "geometry"]]
    assert isinstance(result, GeoDataFrame)
    assert_geodataframe_equal(result, expected)

    result = df.take(np.array([0, 1]), axis=1)
    expected = df[["value1", "value2"]]
    assert isinstance(result, pd.DataFrame)
    assert_frame_equal(result, expected)


def test_take_empty(s, df):
    # ensure that index type is preserved in an empty take
    # https://github.com/geopandas/geopandas/issues/1190
    inds = np.array([], dtype="int64")

    # use non-default index
    df.index = pd.date_range("2012-01-01", periods=len(df))

    result = df.take(inds, axis=0)
    assert isinstance(result, GeoDataFrame)
    assert result.shape == (0, 3)
    assert isinstance(result.index, pd.DatetimeIndex)

    # the original bug report was an empty boolean mask
    for result in [df.loc[df["value1"] > 100], df[df["value1"] > 100]]:
        assert isinstance(result, GeoDataFrame)
        assert result.shape == (0, 3)
        assert isinstance(result.index, pd.DatetimeIndex)


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
    df2.loc[0, "geometry"] = Point(10, 10)
    assert_geoseries_equal(df2["geometry"], exp)

    df2 = df.copy()
    df2.iloc[0, 0] = Point(10, 10)
    assert_geoseries_equal(df2["geometry"], exp)


def test_assign(df):
    res = df.assign(new=1)
    exp = df.copy()
    exp["new"] = 1
    assert isinstance(res, GeoDataFrame)
    assert_frame_equal(res, exp)


def test_astype(s, df):
    # check geoseries functionality
    with pytest.raises(TypeError):
        s.astype(int)

    assert s.astype(str)[0] == "POINT (0 0)"

    res = s.astype(object)
    assert isinstance(res, pd.Series) and not isinstance(res, GeoSeries)
    assert res.dtype == object

    df = df.rename_geometry("geom_list")

    # check whether returned object is a geodataframe
    res = df.astype({"value1": float})
    assert isinstance(res, GeoDataFrame)

    # check whether returned object is a dataframe
    res = df.astype(str)
    assert isinstance(res, pd.DataFrame) and not isinstance(res, GeoDataFrame)

    res = df.astype({"geom_list": str})
    assert isinstance(res, pd.DataFrame) and not isinstance(res, GeoDataFrame)

    res = df.astype(object)
    assert isinstance(res, pd.DataFrame) and not isinstance(res, GeoDataFrame)
    assert res["geom_list"].dtype == object


def test_astype_invalid_geodataframe():
    # https://github.com/geopandas/geopandas/issues/1144
    # a GeoDataFrame without geometry column should not error in astype
    df = GeoDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res = df.astype(object)
    assert isinstance(res, pd.DataFrame) and not isinstance(res, GeoDataFrame)
    assert res["a"].dtype == object


def test_convert_dtypes(df):
    # https://github.com/geopandas/geopandas/issues/1870

    # Test geometry col is first col, first, geom_col_name=geometry
    # (order is important in concat, used internally)
    res1 = df.convert_dtypes()

    expected1 = GeoDataFrame(
        pd.DataFrame(df).convert_dtypes(), crs=df.crs, geometry=df.geometry.name
    )

    # Checking type and metadata are right
    assert_geodataframe_equal(expected1, res1)

    # Test geom last, geom_col_name=geometry
    res2 = df[["value1", "value2", "geometry"]].convert_dtypes()
    assert_geodataframe_equal(expected1[["value1", "value2", "geometry"]], res2)

    # Test again with crs set and custom geom col name
    df2 = df.set_crs(epsg=4326).rename_geometry("points")
    expected2 = GeoDataFrame(
        pd.DataFrame(df2).convert_dtypes(), crs=df2.crs, geometry=df2.geometry.name
    )
    res3 = df2.convert_dtypes()
    assert_geodataframe_equal(expected2, res3)

    # Test geom last, geom_col=geometry
    res4 = df2[["value1", "value2", "points"]].convert_dtypes()
    assert_geodataframe_equal(expected2[["value1", "value2", "points"]], res4)


def test_to_csv(df):
    exp = (
        "geometry,value1,value2\nPOINT (0 0),0,1\nPOINT (1 1),1,2\nPOINT (2 2),2,1\n"
    ).replace("\n", os.linesep)
    assert df.to_csv(index=False) == exp


@pytest.mark.filterwarnings(
    "ignore:Dropping of nuisance columns in DataFrame reductions"
)
def test_numerical_operations(s, df):
    # df methods ignore the geometry column
    exp = pd.Series([3, 4], index=["value1", "value2"])
    if not compat.PANDAS_GE_20:
        res = df.sum()
    else:
        res = df.sum(numeric_only=True)
    assert_series_equal(res, exp)

    # series methods raise error (not supported for geometry)
    with pytest.raises(TypeError):
        s.sum()

    with pytest.raises(TypeError):
        s.max()

    with pytest.raises((TypeError, ValueError)):
        # TODO: remove ValueError after pandas-dev/pandas#32749
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
    exp = GeoSeries([Point(0, 0), None, Point(2, 2)])
    assert_series_equal(res, exp)


def test_select_dtypes(df):
    res = df.select_dtypes(include=[np.number])
    exp = df[["value1", "value2"]]
    assert_frame_equal(res, exp)


def test_equals(s, df):
    # https://github.com/geopandas/geopandas/issues/1420
    s2 = s.copy()
    assert s.equals(s2) is True
    s2.iloc[0] = None
    assert s.equals(s2) is False

    df2 = df.copy()
    assert df.equals(df2) is True
    df2.loc[0, "geometry"] = Point(10, 10)
    assert df.equals(df2) is False
    df2 = df.copy()
    df2.loc[0, "value1"] = 10
    assert df.equals(df2) is False


# Missing values


def test_fillna_scalar(s, df):
    s2 = GeoSeries([Point(0, 0), None, Point(2, 2)])

    res = s2.fillna(Point(1, 1))
    assert_geoseries_equal(res, s)

    # allow np.nan although this does not change anything
    # https://github.com/geopandas/geopandas/issues/1149
    res = s2.fillna(np.nan)
    assert_geoseries_equal(res, s2)

    # raise exception if trying to fill missing geometry w/ non-geometry
    df2 = df.copy()
    df2["geometry"] = s2
    res = df2.fillna(Point(1, 1))
    assert_geodataframe_equal(res, df)
    with pytest.raises((NotImplementedError, TypeError)):  # GH2351
        df2.fillna(0)

    # allow non-geometry fill value if there are no missing values
    # https://github.com/geopandas/geopandas/issues/1149
    df3 = df.copy()
    df3.loc[0, "value1"] = np.nan
    res = df3.fillna(0)
    assert_geodataframe_equal(res.astype({"value1": "int64"}), df)


def test_fillna_series(s):
    # fill na with another GeoSeries
    s2 = GeoSeries([Point(0, 0), None, Point(2, 2)])

    # check na filled with the same index
    res = s2.fillna(GeoSeries([Point(1, 1)] * 3))
    assert_geoseries_equal(res, s)

    # check na filled based on index, not position
    index = [3, 2, 1]
    res = s2.fillna(GeoSeries([Point(i, i) for i in index], index=index))
    assert_geoseries_equal(res, s)

    # check na filled but the input length is different
    res = s2.fillna(GeoSeries([Point(1, 1)], index=[1]))
    assert_geoseries_equal(res, s)

    # check na filled but the inputting index is different
    res = s2.fillna(GeoSeries([Point(1, 1)], index=[9]))
    assert_geoseries_equal(res, s2)


def test_dropna():
    s2 = GeoSeries([Point(0, 0), None, Point(2, 2)])
    res = s2.dropna()
    exp = s2.loc[[0, 2]]
    assert_geoseries_equal(res, exp)


@pytest.mark.parametrize("NA", [None, np.nan])
def test_isna(NA):
    s2 = GeoSeries([Point(0, 0), NA, Point(2, 2)], index=[2, 4, 5], name="tt")
    exp = pd.Series([False, True, False], index=[2, 4, 5], name="tt")
    res = s2.isnull()
    assert type(res) == pd.Series
    assert_series_equal(res, exp)
    res = s2.isna()
    assert_series_equal(res, exp)
    res = s2.notnull()
    assert_series_equal(res, ~exp)
    res = s2.notna()
    assert_series_equal(res, ~exp)


# Any / all


def test_any_all():
    empty = GeometryCollection([])
    s = GeoSeries([empty, Point(1, 1)])
    assert not s.all()
    assert s.any()

    s = GeoSeries([Point(1, 1), Point(1, 1)])
    assert s.all()
    assert s.any()

    s = GeoSeries([empty, empty])
    assert not s.all()
    assert not s.any()


# Groupby / algos


def test_sort_values():
    s = GeoSeries([Point(0, 0), Point(2, 2), Point(0, 2)])
    res = s.sort_values()
    assert res.index.tolist() == [0, 2, 1]
    res2 = s.sort_values(ascending=False)
    assert res2.index.tolist() == [1, 2, 0]

    # empty geoseries
    assert_geoseries_equal(s.iloc[:0].sort_values(), s.iloc[:0])


def test_sort_values_empty_missing():
    s = GeoSeries([Point(0, 0), None, Point(), Point(1, 1)])
    # default: NA sorts last, empty first
    res = s.sort_values()
    assert res.index.tolist() == [2, 0, 3, 1]

    # descending: NA sorts last, empty last
    res = s.sort_values(ascending=False)
    assert res.index.tolist() == [3, 0, 2, 1]

    # NAs first, empty first after NAs
    res = s.sort_values(na_position="first")
    assert res.index.tolist() == [1, 2, 0, 3]

    # NAs first, descending with empyt last
    res = s.sort_values(ascending=False, na_position="first")
    assert res.index.tolist() == [1, 3, 0, 2]

    # all missing / empty
    s = GeoSeries([None, None, None])
    res = s.sort_values()
    assert res.index.tolist() == [0, 1, 2]

    s = GeoSeries([Point(), Point(), Point()])
    res = s.sort_values()
    assert res.index.tolist() == [0, 1, 2]

    s = GeoSeries([Point(), None, Point()])
    res = s.sort_values()
    assert res.index.tolist() == [0, 2, 1]


def test_unique():
    s = GeoSeries([Point(0, 0), Point(0, 0), Point(2, 2)])
    exp = from_shapely([Point(0, 0), Point(2, 2)])
    # TODO should have specialized GeometryArray assert method
    assert_array_equal(s.unique(), exp)


def pd14_compat_index(index):
    if compat.PANDAS_GE_14:
        return from_shapely(index)
    else:
        return index


def test_value_counts():
    # each object is considered unique
    s = GeoSeries([Point(0, 0), Point(1, 1), Point(0, 0)])
    res = s.value_counts()
    if compat.PANDAS_GE_20:
        name = "count"
    else:
        name = None
    with compat.ignore_shapely2_warnings():
        exp = pd.Series(
            [2, 1], index=pd14_compat_index([Point(0, 0), Point(1, 1)]), name=name
        )
    assert_series_equal(res, exp)
    # Check crs doesn't make a difference - note it is not kept in output index anyway
    s2 = GeoSeries([Point(0, 0), Point(1, 1), Point(0, 0)], crs="EPSG:4326")
    res2 = s2.value_counts()
    assert_series_equal(res2, exp)
    if compat.PANDAS_GE_14:
        # TODO should/ can we fix CRS being lost
        assert s2.value_counts().index.array.crs is None

    # check mixed geometry
    s3 = GeoSeries([Point(0, 0), LineString([[1, 1], [2, 2]]), Point(0, 0)])
    res3 = s3.value_counts()
    index = pd14_compat_index([Point(0, 0), LineString([[1, 1], [2, 2]])])
    with compat.ignore_shapely2_warnings():
        exp3 = pd.Series([2, 1], index=index, name=name)
    assert_series_equal(res3, exp3)

    # check None is handled
    s4 = GeoSeries([Point(0, 0), None, Point(0, 0)])
    res4 = s4.value_counts(dropna=True)
    with compat.ignore_shapely2_warnings():
        exp4_dropna = pd.Series([2], index=pd14_compat_index([Point(0, 0)]), name=name)
    assert_series_equal(res4, exp4_dropna)
    with compat.ignore_shapely2_warnings():
        exp4_keepna = pd.Series(
            [2, 1], index=pd14_compat_index([Point(0, 0), None]), name=name
        )
    res4_keepna = s4.value_counts(dropna=False)
    assert_series_equal(res4_keepna, exp4_keepna)


@pytest.mark.xfail(strict=False)
def test_drop_duplicates_series():
    # duplicated does not yet use EA machinery
    # (https://github.com/pandas-dev/pandas/issues/27264)
    # but relies on unstable hashing of unhashable objects in numpy array
    # giving flaky test (https://github.com/pandas-dev/pandas/issues/27035)
    dups = GeoSeries([Point(0, 0), Point(0, 0)])
    dropped = dups.drop_duplicates()
    assert len(dropped) == 1


@pytest.mark.xfail(strict=False)
def test_drop_duplicates_frame():
    # duplicated does not yet use EA machinery, see above
    gdf_len = 3
    dup_gdf = GeoDataFrame(
        {"geometry": [Point(0, 0) for _ in range(gdf_len)], "value1": range(gdf_len)}
    )
    dropped_geometry = dup_gdf.drop_duplicates(subset="geometry")
    assert len(dropped_geometry) == 1
    dropped_all = dup_gdf.drop_duplicates()
    assert len(dropped_all) == gdf_len


def test_groupby(df):
    # counts work fine
    res = df.groupby("value2").count()
    exp = pd.DataFrame(
        {"geometry": [2, 1], "value1": [2, 1], "value2": [1, 2]}
    ).set_index("value2")
    assert_frame_equal(res, exp)

    # reductions ignore geometry column
    if not compat.PANDAS_GE_20:
        res = df.groupby("value2").sum()
    else:
        res = df.groupby("value2").sum(numeric_only=True)
    exp = pd.DataFrame({"value1": [2, 1], "value2": [1, 2]}, dtype="int64").set_index(
        "value2"
    )
    assert_frame_equal(res, exp)

    # applying on the geometry column
    res = df.groupby("value2")["geometry"].apply(lambda x: x.unary_union)

    exp = GeoSeries(
        [shapely.geometry.MultiPoint([(0, 0), (2, 2)]), Point(1, 1)],
        index=pd.Index([1, 2], name="value2"),
        name="geometry",
    )
    assert_series_equal(res, exp)

    # apply on geometry column not resulting in new geometry
    res = df.groupby("value2")["geometry"].apply(lambda x: x.unary_union.area)
    exp = pd.Series([0.0, 0.0], index=pd.Index([1, 2], name="value2"), name="geometry")

    assert_series_equal(res, exp)


def test_groupby_groups(df):
    g = df.groupby("value2")
    res = g.get_group(1)
    assert isinstance(res, GeoDataFrame)
    exp = df.loc[[0, 2]]
    assert_frame_equal(res, exp)


@pytest.mark.skip_no_sindex
@pytest.mark.skipif(
    compat.PANDAS_GE_13 and not compat.PANDAS_GE_14,
    reason="this was broken in pandas 1.3.5 (GH-2294)",
)
@pytest.mark.parametrize("crs", [None, "EPSG:4326"])
def test_groupby_metadata(crs):
    # https://github.com/geopandas/geopandas/issues/2294
    df = GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1), Point(0, 0)],
            "value1": np.arange(3, dtype="int64"),
            "value2": np.array([1, 2, 1], dtype="int64"),
        },
        crs=crs,
    )

    # dummy test asserting we can access the crs
    def func(group):
        assert isinstance(group, GeoDataFrame)
        assert group.crs == crs

    df.groupby("value2").apply(func)

    # actual test with functionality
    res = df.groupby("value2").apply(
        lambda x: geopandas.sjoin(x, x[["geometry", "value1"]], how="inner")
    )

    expected = (
        df.take([0, 2, 0, 2, 1])
        .set_index("value2", drop=False, append=True)
        .swaplevel()
        .rename(columns={"value1": "value1_left"})
        .assign(value1_right=[0, 0, 2, 2, 1])
    )
    assert_geodataframe_equal(res.drop(columns=["index_right"]), expected)


def test_apply(s):
    # function that returns geometry preserves GeoSeries class
    def geom_func(geom):
        assert isinstance(geom, Point)
        return geom

    result = s.apply(geom_func)
    assert isinstance(result, GeoSeries)
    assert_geoseries_equal(result, s)

    # function that returns non-geometry results in Series
    def numeric_func(geom):
        assert isinstance(geom, Point)
        return geom.x

    result = s.apply(numeric_func)
    assert not isinstance(result, GeoSeries)
    assert_series_equal(result, pd.Series([0.0, 1.0, 2.0]))


def test_apply_loc_len1(df):
    # subset of len 1 with loc -> bug in pandas with inconsistent Block ndim
    # resulting in bug in apply
    # https://github.com/geopandas/geopandas/issues/1078
    subset = df.loc[[0], "geometry"]
    result = subset.apply(lambda geom: geom.is_empty)
    expected = subset.is_empty
    np.testing.assert_allclose(result, expected)


def test_apply_convert_dtypes_keyword(s):
    # ensure the convert_dtypes keyword is accepted
    res = s.apply(lambda x: x, convert_dtype=True, args=())
    assert_geoseries_equal(res, s)


@pytest.mark.parametrize("crs", [None, "EPSG:4326"])
def test_apply_no_geometry_result(df, crs):
    if crs:
        df = df.set_crs(crs)
    result = df.apply(lambda col: col.astype(str), axis=0)
    assert type(result) is pd.DataFrame
    expected = df.astype(str)
    assert_frame_equal(result, expected)

    result = df.apply(lambda col: col.astype(str), axis=1)
    assert type(result) is pd.DataFrame
    assert_frame_equal(result, expected)


def test_apply_preserves_geom_col_name(df):
    df = df.rename_geometry("geom")
    result = df.apply(lambda col: col, axis=0)
    assert result.geometry.name == "geom"


def test_df_apply_returning_series(df):
    # https://github.com/geopandas/geopandas/issues/2283
    result = df.apply(lambda row: row.geometry, axis=1)
    assert_geoseries_equal(result, df.geometry, check_crs=False)

    result = df.apply(lambda row: row.value1, axis=1)
    assert_series_equal(result, df["value1"].rename(None))
    # https://github.com/geopandas/geopandas/issues/2480
    result = df.apply(lambda x: float("NaN"), axis=1)
    assert result.dtype == "float64"
    # assert list of nones is not promoted to GeometryDtype
    result = df.apply(lambda x: None, axis=1)
    assert result.dtype == "object"


def test_df_apply_geometry_dtypes(df):
    # https://github.com/geopandas/geopandas/issues/1852
    apply_types = []

    def get_dtypes(srs):
        apply_types.append((srs.name, type(srs)))

    df["geom2"] = df.geometry
    df.apply(get_dtypes)
    expected = [
        ("geometry", GeoSeries),
        ("value1", pd.Series),
        ("value2", pd.Series),
        ("geom2", GeoSeries),
    ]
    assert apply_types == expected


def test_pivot(df):
    # https://github.com/geopandas/geopandas/issues/2057
    # pivot failing due to creating a MultiIndex
    result = df.pivot(columns="value1")
    expected = GeoDataFrame(pd.DataFrame(df).pivot(columns="value1"))
    # TODO assert_geodataframe_equal crashes
    assert isinstance(result, GeoDataFrame)
    assert_frame_equal(result, expected)


def test_preserve_attrs(df):
    # https://github.com/geopandas/geopandas/issues/1654
    df.attrs["name"] = "my_name"
    attrs = {"name": "my_name"}
    assert df.attrs == attrs

    # preserve attrs in indexing operations
    for subset in [df[:2], df[df["value1"] > 2], df[["value2", "geometry"]]]:
        assert df.attrs == attrs

    # preserve attrs in methods
    df2 = df.reset_index()
    assert df2.attrs == attrs

    # https://github.com/geopandas/geopandas/issues/1875
    df3 = df2.explode(index_parts=True)
    assert df3.attrs == attrs


@pytest.mark.skipif(not compat.PANDAS_GE_12, reason="attrs introduced in pandas 1.0")
def test_preserve_flags(df):
    # https://github.com/geopandas/geopandas/issues/1654
    df = df.set_flags(allows_duplicate_labels=False)
    assert df.flags.allows_duplicate_labels is False

    # preserve flags in indexing operations
    for subset in [df[:2], df[df["value1"] > 2], df[["value2", "geometry"]]]:
        assert df.flags.allows_duplicate_labels is False

    # preserve attrs in methods
    df2 = df.reset_index()
    assert df2.flags.allows_duplicate_labels is False

    # it is honored for operations that introduce duplicate labels
    with pytest.raises(ValueError):
        df.reindex([0, 0, 1])

    with pytest.raises(ValueError):
        df[["value1", "value1", "geometry"]]

    with pytest.raises(ValueError):
        pd.concat([df, df])
