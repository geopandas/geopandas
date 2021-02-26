import numpy as np
import pandas as pd

import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_025

from pandas.testing import assert_frame_equal
import pytest


@pytest.fixture
def nybb_polydf():
    nybb_filename = geopandas.datasets.get_path("nybb")
    nybb_polydf = read_file(nybb_filename)
    nybb_polydf = nybb_polydf[["geometry", "BoroName", "BoroCode"]]
    nybb_polydf = nybb_polydf.rename(columns={"geometry": "myshapes"})
    nybb_polydf = nybb_polydf.set_geometry("myshapes")
    nybb_polydf["manhattan_bronx"] = 5
    nybb_polydf.loc[3:4, "manhattan_bronx"] = 6
    return nybb_polydf


@pytest.fixture
def merged_shapes(nybb_polydf):
    # Merged geometry
    manhattan_bronx = nybb_polydf.loc[3:4]
    others = nybb_polydf.loc[0:2]

    collapsed = [others.geometry.unary_union, manhattan_bronx.geometry.unary_union]
    merged_shapes = GeoDataFrame(
        {"myshapes": collapsed},
        geometry="myshapes",
        index=pd.Index([5, 6], name="manhattan_bronx"),
        crs=nybb_polydf.crs,
    )

    return merged_shapes


@pytest.fixture
def first(merged_shapes):
    first = merged_shapes.copy()
    first["BoroName"] = ["Staten Island", "Manhattan"]
    first["BoroCode"] = [5, 1]
    return first


@pytest.fixture
def expected_mean(merged_shapes):
    test_mean = merged_shapes.copy()
    test_mean["BoroCode"] = [4, 1.5]
    return test_mean


def test_geom_dissolve(nybb_polydf, first):
    test = nybb_polydf.dissolve("manhattan_bronx")
    assert test.geometry.name == "myshapes"
    assert test.geom_almost_equals(first).all()


def test_dissolve_retains_existing_crs(nybb_polydf):
    assert nybb_polydf.crs is not None
    test = nybb_polydf.dissolve("manhattan_bronx")
    assert test.crs is not None


def test_dissolve_retains_nonexisting_crs(nybb_polydf):
    nybb_polydf.crs = None
    test = nybb_polydf.dissolve("manhattan_bronx")
    assert test.crs is None


def first_dissolve(nybb_polydf, first):
    test = nybb_polydf.dissolve("manhattan_bronx")
    assert_frame_equal(first, test, check_column_type=False)


def test_mean_dissolve(nybb_polydf, first, expected_mean):
    test = nybb_polydf.dissolve("manhattan_bronx", aggfunc="mean")
    assert_frame_equal(expected_mean, test, check_column_type=False)

    test = nybb_polydf.dissolve("manhattan_bronx", aggfunc=np.mean)
    assert_frame_equal(expected_mean, test, check_column_type=False)


def test_multicolumn_dissolve(nybb_polydf, first):
    multi = nybb_polydf.copy()
    multi["dup_col"] = multi.manhattan_bronx
    multi_test = multi.dissolve(["manhattan_bronx", "dup_col"], aggfunc="first")

    first_copy = first.copy()
    first_copy["dup_col"] = first_copy.index
    first_copy = first_copy.set_index([first_copy.index, "dup_col"])

    assert_frame_equal(multi_test, first_copy, check_column_type=False)


def test_reset_index(nybb_polydf, first):
    test = nybb_polydf.dissolve("manhattan_bronx", as_index=False)
    comparison = first.reset_index()
    assert_frame_equal(comparison, test, check_column_type=False)


def test_dissolve_none(nybb_polydf):
    test = nybb_polydf.dissolve(by=None)
    expected = GeoDataFrame(
        {
            nybb_polydf.geometry.name: [nybb_polydf.geometry.unary_union],
            "BoroName": ["Staten Island"],
            "BoroCode": [5],
            "manhattan_bronx": [5],
        },
        geometry=nybb_polydf.geometry.name,
        crs=nybb_polydf.crs,
    )
    assert_frame_equal(expected, test, check_column_type=False)


def test_dissolve_none_mean(nybb_polydf):
    test = nybb_polydf.dissolve(aggfunc="mean")
    expected = GeoDataFrame(
        {
            nybb_polydf.geometry.name: [nybb_polydf.geometry.unary_union],
            "BoroCode": [3.0],
            "manhattan_bronx": [5.4],
        },
        geometry=nybb_polydf.geometry.name,
        crs=nybb_polydf.crs,
    )
    assert_frame_equal(expected, test, check_column_type=False)


@pytest.mark.skipif(
    not PANDAS_GE_025, reason="'observed' param behavior changed in pandas 0.25.0"
)
def test_dissolve_categorical():
    gdf = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b"]),
            "noncat": [1, 1, 1, 2],
            "to_agg": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
        }
    )

    # when observed=False we get an additional observation
    # that wasn't in the original data
    expected_gdf_observed_false = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b"]),
            "noncat": [1, 2, 1, 2],
            "geometry": geopandas.array.from_wkt(
                [
                    "MULTIPOINT (0 0, 1 1)",
                    None,
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
            "to_agg": [1, None, 3, 4],
        }
    ).set_index(["cat", "noncat"])

    # when observed=True we do not get any additional observations
    expected_gdf_observed_true = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["a", "b", "b"]),
            "noncat": [1, 1, 2],
            "geometry": geopandas.array.from_wkt(
                ["MULTIPOINT (0 0, 1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
            "to_agg": [1, 3, 4],
        }
    ).set_index(["cat", "noncat"])

    assert_frame_equal(expected_gdf_observed_false, gdf.dissolve(["cat", "noncat"]))
    assert_frame_equal(
        expected_gdf_observed_true, gdf.dissolve(["cat", "noncat"], observed=True)
    )
