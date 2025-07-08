import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

import shapely
from shapely.geometry import GeometryCollection, Point, Polygon, box

import geopandas
import geopandas._compat as compat
from geopandas import (
    GeoDataFrame,
    GeoSeries,
    points_from_xy,
    read_file,
    sjoin,
    sjoin_nearest,
)

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal


@pytest.fixture()
def dfs(request):
    polys1 = GeoSeries(
        [
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
            Polygon([(6, 0), (9, 0), (9, 3), (6, 3)]),
        ]
    )

    polys2 = GeoSeries(
        [
            Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),
            Polygon([(4, 4), (7, 4), (7, 7), (4, 7)]),
            Polygon([(7, 7), (10, 7), (10, 10), (7, 10)]),
        ]
    )

    df1 = GeoDataFrame({"geometry": polys1, "df1": [0, 1, 2]})
    df2 = GeoDataFrame({"geometry": polys2, "df2": [3, 4, 5]})

    if request.param == "string-index":
        df1.index = ["a", "b", "c"]
        df2.index = ["d", "e", "f"]

    if request.param == "named-index":
        df1.index.name = "df1_ix"
        df2.index.name = "df2_ix"

    if request.param == "multi-index":
        i1 = ["a", "b", "c"]
        i2 = ["d", "e", "f"]
        df1 = df1.set_index([i1, i2])
        df2 = df2.set_index([i2, i1])

    if request.param == "named-multi-index":
        i1 = ["a", "b", "c"]
        i2 = ["d", "e", "f"]
        df1 = df1.set_index([i1, i2])
        df2 = df2.set_index([i2, i1])
        df1.index.names = ["df1_ix1", "df1_ix2"]
        df2.index.names = ["df2_ix1", "df2_ix2"]

    # construction expected frames
    expected = {}

    part1 = df1.copy().reset_index().rename(columns={"index": "index_left"})
    part2 = (
        df2.copy()
        .iloc[[0, 1, 1, 2]]
        .reset_index()
        .rename(columns={"index": "index_right"})
    )
    part1["_merge"] = [0, 1, 2]
    part2["_merge"] = [0, 0, 1, 3]
    exp = pd.merge(part1, part2, on="_merge", how="outer")
    expected["intersects"] = exp.drop("_merge", axis=1).copy()

    part1 = df1.copy().reset_index().rename(columns={"index": "index_left"})
    part2 = df2.copy().reset_index().rename(columns={"index": "index_right"})
    part1["_merge"] = [0, 1, 2]
    part2["_merge"] = [0, 3, 3]
    exp = pd.merge(part1, part2, on="_merge", how="outer")
    expected["contains"] = exp.drop("_merge", axis=1).copy()

    part1["_merge"] = [0, 1, 2]
    part2["_merge"] = [3, 1, 3]
    exp = pd.merge(part1, part2, on="_merge", how="outer")
    expected["within"] = exp.drop("_merge", axis=1).copy()

    return [request.param, df1, df2, expected]


@pytest.fixture()
def dfs_shared_attribute():
    geo_left = [
        Point(0, 0),
        Point(1, 1),
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(5, 5),
        Point(6, 6),
        Point(7, 7),
    ]
    geo_right = [
        Point(0, 0),
        Point(1, 1),
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(5, 5),
        Point(6, 6),
        Point(7, 7),
    ]
    attr_tracker = ["A", "B", "C", "D", "E", "F", "G", "H"]

    left_gdf = geopandas.GeoDataFrame(
        {
            "geometry": geo_left,
            "attr_tracker": attr_tracker,
            "duplicate_column": [0, 1, 2, 3, 4, 5, 6, 7],
            "attr1": [True, True, True, True, True, True, True, True],
            "attr2": [True, True, True, True, True, True, True, True],
        }
    )

    right_gdf = geopandas.GeoDataFrame(
        {
            "geometry": geo_right,
            "duplicate_column": [0, 1, 2, 3, 4, 5, 6, 7],
            "attr1": [True, True, False, False, True, True, False, False],
            "attr2": [True, True, False, False, False, False, False, False],
        }
    )

    return left_gdf, right_gdf


class TestSpatialJoin:
    @pytest.mark.parametrize(
        "how, lsuffix, rsuffix, expected_cols",
        [
            ("left", "left", "right", {"col_left", "col_right", "index_right"}),
            ("inner", "left", "right", {"col_left", "col_right", "index_right"}),
            ("right", "left", "right", {"col_left", "col_right", "index_left"}),
            ("left", "lft", "rgt", {"col_lft", "col_rgt", "index_rgt"}),
            ("inner", "lft", "rgt", {"col_lft", "col_rgt", "index_rgt"}),
            ("right", "lft", "rgt", {"col_lft", "col_rgt", "index_lft"}),
        ],
    )
    def test_suffixes(self, how: str, lsuffix: str, rsuffix: str, expected_cols):
        left = GeoDataFrame({"col": [1], "geometry": [Point(0, 0)]})
        right = GeoDataFrame({"col": [1], "geometry": [Point(0, 0)]})
        joined = sjoin(left, right, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
        assert set(joined.columns) == expected_cols | {"geometry"}

    @pytest.mark.skipif(not compat.HAS_PYPROJ, reason="pyproj not available")
    @pytest.mark.parametrize("dfs", ["default-index", "string-index"], indirect=True)
    def test_crs_mismatch(self, dfs):
        index, df1, df2, expected = dfs
        df1.crs = "epsg:4326"
        with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
            sjoin(df1, df2)

    @pytest.mark.parametrize("dfs", ["default-index"], indirect=True)
    def test_unknown_kwargs(self, dfs):
        _, df1, df2, _ = dfs
        with pytest.raises(
            TypeError,
            match=r"sjoin\(\) got an unexpected keyword argument 'extra_param'",
        ):
            sjoin(df1, df2, extra_param="test")

    @pytest.mark.parametrize(
        "dfs",
        [
            "default-index",
            "string-index",
            "named-index",
            "multi-index",
            "named-multi-index",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("predicate", ["intersects", "contains", "within"])
    def test_inner(self, predicate, dfs):
        index, df1, df2, expected = dfs

        res = sjoin(df1, df2, how="inner", predicate=predicate)
        exp = expected[predicate].dropna().copy()
        exp = exp.drop("geometry_y", axis=1).rename(columns={"geometry_x": "geometry"})
        exp[["df1", "df2"]] = exp[["df1", "df2"]].astype("int64")
        if index == "default-index":
            exp[["index_left", "index_right"]] = exp[
                ["index_left", "index_right"]
            ].astype("int64")
        if index == "named-index":
            exp[["df1_ix", "df2_ix"]] = exp[["df1_ix", "df2_ix"]].astype("int64")
            exp = exp.set_index("df1_ix")
        if index in ["default-index", "string-index"]:
            exp = exp.set_index("index_left")
            exp.index.name = None
        if index == "multi-index":
            exp = exp.set_index(["level_0_x", "level_1_x"]).rename(
                columns={"level_0_y": "index_right0", "level_1_y": "index_right1"}
            )
            exp.index.names = df1.index.names
        if index == "named-multi-index":
            exp = exp.set_index(["df1_ix1", "df1_ix2"])
        assert_frame_equal(res, exp)

    @pytest.mark.parametrize(
        "dfs",
        [
            "default-index",
            "string-index",
            "named-index",
            "multi-index",
            "named-multi-index",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("predicate", ["intersects", "contains", "within"])
    def test_left(self, predicate, dfs):
        index, df1, df2, expected = dfs

        res = sjoin(df1, df2, how="left", predicate=predicate)

        if index in ["default-index", "string-index"]:
            exp = expected[predicate].dropna(subset=["index_left"]).copy()
        elif index == "named-index":
            exp = expected[predicate].dropna(subset=["df1_ix"]).copy()
        elif index == "multi-index":
            exp = expected[predicate].dropna(subset=["level_0_x"]).copy()
        elif index == "named-multi-index":
            exp = expected[predicate].dropna(subset=["df1_ix1"]).copy()
        exp = exp.drop("geometry_y", axis=1).rename(columns={"geometry_x": "geometry"})
        exp["df1"] = exp["df1"].astype("int64")
        if index == "default-index":
            exp["index_left"] = exp["index_left"].astype("int64")
            # TODO: in result the dtype is object
            res["index_right"] = res["index_right"].astype(float)
        elif index == "named-index":
            exp[["df1_ix"]] = exp[["df1_ix"]].astype("int64")
            exp = exp.set_index("df1_ix")
        if index in ["default-index", "string-index"]:
            exp = exp.set_index("index_left")
            exp.index.name = None
        if index == "multi-index":
            exp = exp.set_index(["level_0_x", "level_1_x"]).rename(
                columns={"level_0_y": "index_right0", "level_1_y": "index_right1"}
            )
            exp.index.names = df1.index.names
        if index == "named-multi-index":
            exp = exp.set_index(["df1_ix1", "df1_ix2"])

        assert_frame_equal(res, exp)

    def test_empty_join(self):
        # Check joins resulting in empty gdfs.
        polygons = geopandas.GeoDataFrame(
            {
                "col2": [1, 2],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            }
        )
        not_in = geopandas.GeoDataFrame({"col1": [1], "geometry": [Point(-0.5, 0.5)]})
        empty = sjoin(not_in, polygons, how="left", predicate="intersects")
        assert empty.index_right.isnull().all()
        empty = sjoin(not_in, polygons, how="right", predicate="intersects")
        assert empty.index_left.isnull().all()
        empty = sjoin(not_in, polygons, how="inner", predicate="intersects")
        assert empty.empty

    @pytest.mark.parametrize(
        "predicate",
        [
            "contains",
            "contains_properly",
            "covered_by",
            "covers",
            "crosses",
            "intersects",
            "touches",
            "within",
        ],
    )
    @pytest.mark.parametrize(
        "empty",
        [
            GeoDataFrame(geometry=[GeometryCollection(), GeometryCollection()]),
            GeoDataFrame(geometry=GeoSeries()),
        ],
    )
    def test_join_with_empty(self, predicate, empty):
        # Check joins with empty geometry columns/dataframes.
        polygons = geopandas.GeoDataFrame(
            {
                "col2": [1, 2],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            }
        )
        result = sjoin(empty, polygons, how="left", predicate=predicate)
        assert result.index_right.isnull().all()
        result = sjoin(empty, polygons, how="right", predicate=predicate)
        assert result.index_left.isnull().all()
        result = sjoin(empty, polygons, how="inner", predicate=predicate)
        assert result.empty

    @pytest.mark.parametrize("dfs", ["default-index", "string-index"], indirect=True)
    def test_sjoin_invalid_args(self, dfs):
        index, df1, df2, expected = dfs

        with pytest.raises(ValueError, match="'left_df' should be GeoDataFrame"):
            sjoin(df1.geometry, df2)

        with pytest.raises(ValueError, match="'right_df' should be GeoDataFrame"):
            sjoin(df1, df2.geometry)

    @pytest.mark.parametrize(
        "dfs",
        [
            "default-index",
            "string-index",
            "named-index",
            "multi-index",
            "named-multi-index",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("predicate", ["intersects", "contains", "within"])
    def test_right(self, predicate, dfs):
        index, df1, df2, expected = dfs

        res = sjoin(df1, df2, how="right", predicate=predicate)

        if index in ["default-index", "string-index"]:
            exp = expected[predicate].dropna(subset=["index_right"]).copy()
        elif index == "named-index":
            exp = expected[predicate].dropna(subset=["df2_ix"]).copy()
        elif index == "multi-index":
            exp = expected[predicate].dropna(subset=["level_0_y"]).copy()
        elif index == "named-multi-index":
            exp = expected[predicate].dropna(subset=["df2_ix1"]).copy()
        exp = exp.drop("geometry_x", axis=1).rename(columns={"geometry_y": "geometry"})
        exp["df2"] = exp["df2"].astype("int64")
        if index == "default-index":
            exp["index_right"] = exp["index_right"].astype("int64")
            res["index_left"] = res["index_left"].astype(float)
        elif index == "named-index":
            exp[["df2_ix"]] = exp[["df2_ix"]].astype("int64")
            exp = exp.set_index("df2_ix")
        if index in ["default-index", "string-index"]:
            exp = exp.set_index("index_right")
            exp = exp.reindex(columns=res.columns)
            exp.index.name = None
        if index == "multi-index":
            exp = exp.set_index(["level_0_y", "level_1_y"]).rename(
                columns={"level_0_x": "index_left0", "level_1_x": "index_left1"}
            )
            exp.index.names = df2.index.names
        if index == "named-multi-index":
            exp = exp.set_index(["df2_ix1", "df2_ix2"])

        if predicate == "within":
            exp = exp.sort_index()

        assert_frame_equal(res, exp, check_index_type=False)

    @pytest.mark.skipif(not compat.GEOS_GE_310, reason="`dwithin` requires GEOS 3.10")
    @pytest.mark.parametrize("how", ["inner"])
    @pytest.mark.parametrize(
        "geo_left, geo_right, expected_left, expected_right, distance",
        [
            (
                # Distance is number, 2x1
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1)],
                [0, 1],
                [0, 0],
                math.sqrt(2),
            ),
            # Distance is number, 2x2
            (
                [Point(0, 0), Point(1, 1)],
                [Point(0, 0), Point(1, 1)],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                math.sqrt(2),
            ),
            # Distance is array, matches len(left)
            (
                [Point(0, 0), Point(0, 0), Point(-1, -1)],
                [Point(1, 1)],
                [1, 2],
                [0, 0],
                [0, math.sqrt(2), math.sqrt(8)],
            ),
            # Distance is np.array, matches len(left),
            # inner join sorts the right GeoDataFrame
            (
                [Point(0, 0), Point(0, 0), Point(-1, -1)],
                [Point(1, 1), Point(0.5, 0.5)],
                [1, 2, 1, 2],
                [1, 1, 0, 0],
                np.array([0, math.sqrt(2), math.sqrt(8)]),
            ),
        ],
    )
    def test_sjoin_dwithin(
        self,
        geo_left,
        geo_right,
        expected_left: Sequence[int],
        expected_right: Sequence[int],
        distance,
        how,
    ):
        left = geopandas.GeoDataFrame({"geometry": geo_left})
        right = geopandas.GeoDataFrame({"geometry": geo_right})
        expected_gdf = left.iloc[expected_left].copy()
        expected_gdf["index_right"] = expected_right
        joined = sjoin(left, right, how=how, predicate="dwithin", distance=distance)
        assert_frame_equal(expected_gdf.sort_index(), joined.sort_index())

    # GH3239
    @pytest.mark.parametrize(
        "predicate",
        [
            "contains",
            "contains_properly",
            "covered_by",
            "covers",
            "crosses",
            "intersects",
            "touches",
            "within",
        ],
    )
    def test_sjoin_left_order(self, predicate):
        # a set of points in random order -> that order should be preserved
        # with a left join
        pts = GeoDataFrame(
            geometry=points_from_xy([0.1, 0.4, 0.3, 0.7], [0.8, 0.6, 0.9, 0.1])
        )
        polys = GeoDataFrame(
            {"id": [1, 2, 3, 4]},
            geometry=[
                box(0, 0, 0.5, 0.5),
                box(0, 0.5, 0.5, 1),
                box(0.5, 0, 1, 0.5),
                box(0.5, 0.5, 1, 1),
            ],
        )

        joined = sjoin(pts, polys, predicate=predicate, how="left")
        assert_index_equal(joined.index, pts.index)

    def test_sjoin_shared_attribute(self, naturalearth_lowres, naturalearth_cities):
        countries = read_file(naturalearth_lowres)
        cities = read_file(naturalearth_cities)
        countries = countries[["geometry", "name"]].rename(columns={"name": "country"})

        # Add first letter of country/city as an attribute column to be compared
        countries["firstLetter"] = countries["country"].astype(str).str[0]
        cities["firstLetter"] = cities["name"].astype(str).str[0]

        result = sjoin(cities, countries, on_attribute="firstLetter")
        assert (
            result["country"].astype(str).str[0] == result["name"].astype(str).str[0]
        ).all()
        assert result.shape == (23, 5)

    @pytest.mark.parametrize(
        "attr1_key_change_dict, attr2_key_change_dict",
        [
            pytest.param(
                {True: "merge", False: "no_merge"},
                {True: "merge", False: "no_merge"},
                id="merge on string attributes",
            ),
            pytest.param(
                {True: 2, False: 1},
                {True: 2, False: 1},
                id="merge on integer attributes",
            ),
            pytest.param(
                {True: True, False: False},
                {True: True, False: False},
                id="merge on boolean attributes",
            ),
            pytest.param(
                {True: True, False: False},
                {True: "merge", False: "no_merge"},
                id="merge on mixed attributes",
            ),
        ],
    )
    def test_sjoin_multiple_attributes_datatypes(
        self, dfs_shared_attribute, attr1_key_change_dict, attr2_key_change_dict
    ):
        left_gdf, right_gdf = dfs_shared_attribute
        left_gdf["attr1"] = left_gdf["attr1"].map(attr1_key_change_dict)
        left_gdf["attr2"] = left_gdf["attr2"].map(attr2_key_change_dict)
        right_gdf["attr1"] = right_gdf["attr1"].map(attr1_key_change_dict)
        right_gdf["attr2"] = right_gdf["attr2"].map(attr2_key_change_dict)

        joined = sjoin(left_gdf, right_gdf, on_attribute=("attr1", "attr2"))
        assert (["A", "B"] == joined["attr_tracker"].values).all()

    def test_sjoin_multiple_attributes_check_header(self, dfs_shared_attribute):
        left_gdf, right_gdf = dfs_shared_attribute
        joined = sjoin(left_gdf, right_gdf, on_attribute=["attr1"])

        assert (["A", "B", "E", "F"] == joined["attr_tracker"].values).all()
        assert {"attr2_left", "attr2_right", "attr1"}.issubset(joined.columns)
        assert "attr1_left" not in joined

    def test_sjoin_error_column_does_not_exist(self, dfs_shared_attribute):
        left_gdf, right_gdf = dfs_shared_attribute
        right_gdf_dropped_attr = right_gdf.drop("attr1", axis=1)
        left_gdf_dropped_attr = left_gdf.drop("attr1", axis=1)

        with pytest.raises(
            ValueError,
            match="Expected column attr1 is missing from the right dataframe.",
        ):
            sjoin(left_gdf, right_gdf_dropped_attr, on_attribute="attr1")

        with pytest.raises(
            ValueError,
            match="Expected column attr1 is missing from the left dataframe.",
        ):
            sjoin(left_gdf_dropped_attr, right_gdf, on_attribute="attr1")

        with pytest.raises(
            ValueError,
            match="Expected column attr1 is missing from both of the dataframes.",
        ):
            sjoin(left_gdf_dropped_attr, right_gdf_dropped_attr, on_attribute="attr1")

    def test_sjoin_error_use_geometry_column(self, dfs_shared_attribute):
        left_gdf, right_gdf = dfs_shared_attribute
        with pytest.raises(
            ValueError,
            match="Active geometry column cannot be used as an input for "
            "on_attribute parameter.",
        ):
            sjoin(left_gdf, right_gdf, on_attribute="geometry")
        with pytest.raises(
            ValueError,
            match="Active geometry column cannot be used as an input for "
            "on_attribute parameter.",
        ):
            sjoin(left_gdf, right_gdf, on_attribute=["attr1", "geometry"])


class TestIndexNames:
    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_preserve_index_names(self, how):
        # preserve names of both left and right index
        geoms = [Point(1, 1), Point(2, 2)]
        df1 = GeoDataFrame({"geometry": geoms}, index=pd.Index([1, 2], name="myidx1"))
        df2 = GeoDataFrame(
            {"geometry": geoms}, index=pd.Index(["a", "b"], name="myidx2")
        )
        result = sjoin(df1, df2, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {"myidx1": [1, 2], "geometry": geoms, "myidx2": ["a", "b"]}
            ).set_index("myidx1")
        else:
            # right join
            expected = GeoDataFrame(
                {"myidx2": ["a", "b"], "myidx1": [1, 2], "geometry": geoms},
            ).set_index("myidx2")
        assert_geodataframe_equal(result, expected)

        # but also add suffixes if both left and right have the same index
        df1.index.name = "myidx"
        df2.index.name = "myidx"
        result = sjoin(df1, df2, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {"myidx_left": [1, 2], "geometry": geoms, "myidx_right": ["a", "b"]}
            ).set_index("myidx_left")
        else:
            # right join
            expected = GeoDataFrame(
                {"myidx_right": ["a", "b"], "myidx_left": [1, 2], "geometry": geoms},
            ).set_index("myidx_right")
        assert_geodataframe_equal(result, expected)

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_preserve_index_names_multiindex(self, how):
        # preserve names of both left and right index
        geoms = [Point(1, 1), Point(2, 2)]
        df1 = GeoDataFrame(
            {"geometry": geoms},
            index=pd.MultiIndex.from_tuples(
                [("a", 1), ("b", 2)], names=["myidx1", "level2"]
            ),
        )
        df2 = GeoDataFrame(
            {"geometry": geoms},
            index=pd.MultiIndex.from_tuples(
                [("c", 3), ("d", 4)], names=["myidx2", None]
            ),
        )
        result = sjoin(df1, df2, how=how)
        expected_base = GeoDataFrame(
            {
                "myidx1": ["a", "b"],
                "level2": [1, 2],
                "geometry": geoms,
                "myidx2": ["c", "d"],
                "index_right1": [3, 4],
            }
        )
        if how in ("inner", "left"):
            expected = expected_base.set_index(["myidx1", "level2"])
        else:
            # right join
            expected = expected_base.set_index(["myidx2", "index_right1"])
            # if it was originally None, that is preserved
            expected.index.names = ["myidx2", None]
        assert_geodataframe_equal(result, expected)

        # but also add suffixes if both left and right have the same index
        df1.index.names = ["myidx", "level2"]
        df2.index.names = ["myidx", None]
        result = sjoin(df1, df2, how=how)
        expected_base = GeoDataFrame(
            {
                "myidx_left": ["a", "b"],
                "level2": [1, 2],
                "geometry": geoms,
                "myidx_right": ["c", "d"],
                "index_right1": [3, 4],
            }
        )
        if how in ("inner", "left"):
            expected = expected_base.set_index(["myidx_left", "level2"])
        else:
            # right join
            expected = expected_base.set_index(["myidx_right", "index_right1"])
            # if it was originally None, that is preserved
            expected.index.names = ["myidx_right", None]
        assert_geodataframe_equal(result, expected)

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_duplicate_column_index_name(self, how):
        # case where a left column and the right index have the same name or the
        # other way around -> correctly add suffix or preserve index name
        geoms = [Point(1, 1), Point(2, 2)]
        df1 = GeoDataFrame({"myidx": [1, 2], "geometry": geoms})
        df2 = GeoDataFrame(
            {"geometry": geoms}, index=pd.Index(["a", "b"], name="myidx")
        )
        result = sjoin(df1, df2, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {"myidx_left": [1, 2], "geometry": geoms, "myidx_right": ["a", "b"]}
            )
        else:
            # right join
            expected = GeoDataFrame(
                {"index_left": [0, 1], "myidx_left": [1, 2], "geometry": geoms},
                index=pd.Index(["a", "b"], name="myidx_right"),
            )
        assert_geodataframe_equal(result, expected)

        result = sjoin(df2, df1, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {"geometry": geoms, "index_right": [0, 1], "myidx_right": [1, 2]},
                index=pd.Index(["a", "b"], name="myidx_left"),
            )
        else:
            # right join
            expected = GeoDataFrame(
                {"myidx_left": ["a", "b"], "myidx_right": [1, 2], "geometry": geoms},
            )
        assert_geodataframe_equal(result, expected)

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_duplicate_column_index_name_multiindex(self, how):
        # case where a left column and the right index have the same name or the
        # other way around -> correctly add suffix or preserve index name
        geoms = [Point(1, 1), Point(2, 2)]
        df1 = GeoDataFrame({"myidx": [1, 2], "geometry": geoms})
        df2 = GeoDataFrame(
            {"geometry": geoms},
            index=pd.MultiIndex.from_tuples(
                [("a", 1), ("b", 2)], names=["myidx", "level2"]
            ),
        )
        result = sjoin(df1, df2, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {
                    "myidx_left": [1, 2],
                    "geometry": geoms,
                    "myidx_right": ["a", "b"],
                    "level2": [1, 2],
                }
            )
        else:
            # right join
            expected = GeoDataFrame(
                {"index_left": [0, 1], "myidx_left": [1, 2], "geometry": geoms},
                index=pd.MultiIndex.from_tuples(
                    [("a", 1), ("b", 2)], names=["myidx_right", "level2"]
                ),
            )
        assert_geodataframe_equal(result, expected)

        result = sjoin(df2, df1, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {"geometry": geoms, "index_right": [0, 1], "myidx_right": [1, 2]},
                index=pd.MultiIndex.from_tuples(
                    [("a", 1), ("b", 2)], names=["myidx_left", "level2"]
                ),
            )
        else:
            # right join
            expected = GeoDataFrame(
                {
                    "myidx_left": ["a", "b"],
                    "level2": [1, 2],
                    "myidx_right": [1, 2],
                    "geometry": geoms,
                },
            )
        assert_geodataframe_equal(result, expected)

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_conflicting_column_index_name(self, how):
        # test case where the auto-generated index name conflicts
        geoms = [Point(1, 1), Point(2, 2)]
        df1 = GeoDataFrame({"index_right": [1, 2], "geometry": geoms})
        df2 = GeoDataFrame({"geometry": geoms})
        with pytest.raises(ValueError, match="'index_right' cannot be a column name"):
            sjoin(df1, df2, how=how)

    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_conflicting_column_with_suffix(self, how):
        # test case where the auto-generated index name conflicts
        geoms = [Point(1, 1), Point(2, 2)]
        df1 = GeoDataFrame(
            {"column": [1, 2], "column_right": ["a", "b"], "geometry": geoms}
        )
        df2 = GeoDataFrame({"column": [0.1, 0.2], "geometry": geoms})

        result = sjoin(df1, df2, how=how)
        if how in ("inner", "left"):
            expected = GeoDataFrame(
                {1: [1, 2], 2: ["a", "b"], 3: geoms, 4: [0, 1], 5: [0.1, 0.2]}
            )
            expected.columns = [
                "column_left",
                "column_right",
                "geometry",
                "index_right",
                "column_right",
            ]
        else:
            # right join
            expected = GeoDataFrame(
                {1: [0, 1], 2: [1, 2], 3: ["a", "b"], 4: [0.1, 0.2], 5: geoms}
            )
            expected.columns = [
                "index_left",
                "column_left",
                "column_right",
                "column_right",
                "geometry",
            ]
        expected = expected.set_geometry("geometry")
        assert_geodataframe_equal(result, expected)


@pytest.mark.usefixtures("_setup_class_nybb_filename")
class TestSpatialJoinNYBB:
    def setup_method(self):
        self.polydf = read_file(self.nybb_filename)
        self.crs = self.polydf.crs
        N = 20
        b = [int(x) for x in self.polydf.total_bounds]
        self.pointdf = GeoDataFrame(
            [
                {"geometry": Point(x, y), "pointattr1": x + y, "pointattr2": x - y}
                for x, y in zip(
                    range(b[0], b[2], int((b[2] - b[0]) / N)),
                    range(b[1], b[3], int((b[3] - b[1]) / N)),
                )
            ],
            crs=self.crs,
        )

    def test_geometry_name(self):
        # test sjoin is working with other geometry name
        polydf_original_geom_name = self.polydf.geometry.name
        self.polydf = self.polydf.rename(columns={"geometry": "new_geom"}).set_geometry(
            "new_geom"
        )
        assert polydf_original_geom_name != self.polydf.geometry.name
        res = sjoin(self.polydf, self.pointdf, how="left")
        assert self.polydf.geometry.name == res.geometry.name

    def test_sjoin_left(self):
        df = sjoin(self.pointdf, self.polydf, how="left")
        assert df.shape == (21, 8)
        for i, row in df.iterrows():
            assert row.geometry.geom_type == "Point"
        assert "pointattr1" in df.columns
        assert "BoroCode" in df.columns

    def test_sjoin_right(self):
        # the inverse of left
        df = sjoin(self.pointdf, self.polydf, how="right")
        df2 = sjoin(self.polydf, self.pointdf, how="left")
        assert df.shape == (12, 8)
        assert df.shape == df2.shape
        for i, row in df.iterrows():
            assert row.geometry.geom_type == "MultiPolygon"
        for i, row in df2.iterrows():
            assert row.geometry.geom_type == "MultiPolygon"

    def test_sjoin_inner(self):
        df = sjoin(self.pointdf, self.polydf, how="inner")
        assert df.shape == (11, 8)

    def test_sjoin_predicate(self):
        # points within polygons
        df = sjoin(self.pointdf, self.polydf, how="left", predicate="within")
        assert df.shape == (21, 8)
        assert df.loc[1]["BoroName"] == "Staten Island"

        # points contain polygons? never happens so we should have nulls
        df = sjoin(self.pointdf, self.polydf, how="left", predicate="contains")
        assert df.shape == (21, 8)
        assert np.isnan(df.loc[1]["Shape_Area"])

    def test_sjoin_bad_predicate(self):
        # AttributeError: 'Point' object has no attribute 'spandex'
        with pytest.raises(ValueError):
            sjoin(self.pointdf, self.polydf, how="left", predicate="spandex")

    def test_sjoin_duplicate_column_name(self):
        pointdf2 = self.pointdf.rename(columns={"pointattr1": "Shape_Area"})
        df = sjoin(pointdf2, self.polydf, how="left")
        assert "Shape_Area_left" in df.columns
        assert "Shape_Area_right" in df.columns

    @pytest.mark.parametrize("how", ["left", "right", "inner"])
    def test_sjoin_named_index(self, how):
        # original index names should be unchanged
        pointdf2 = self.pointdf.copy()
        pointdf2.index.name = "pointid"
        polydf = self.polydf.copy()
        polydf.index.name = "polyid"

        res = sjoin(pointdf2, polydf, how=how)
        assert pointdf2.index.name == "pointid"
        assert polydf.index.name == "polyid"

        # original index name should pass through to result
        if how == "right":
            assert res.index.name == "polyid"
        else:  # how == "left", how == "inner"
            assert res.index.name == "pointid"

    def test_sjoin_values(self):
        # GH190
        self.polydf.index = [1, 3, 4, 5, 6]
        df = sjoin(self.pointdf, self.polydf, how="left")
        assert df.shape == (21, 8)
        df = sjoin(self.polydf, self.pointdf, how="left")
        assert df.shape == (12, 8)

    @pytest.mark.xfail
    def test_no_overlapping_geometry(self):
        # Note: these tests are for correctly returning GeoDataFrame
        # when result of the join is empty

        df_inner = sjoin(self.pointdf.iloc[17:], self.polydf, how="inner")
        df_left = sjoin(self.pointdf.iloc[17:], self.polydf, how="left")
        df_right = sjoin(self.pointdf.iloc[17:], self.polydf, how="right")

        expected_inner_df = pd.concat(
            [
                self.pointdf.iloc[:0],
                pd.Series(name="index_right", dtype="int64"),
                self.polydf.drop("geometry", axis=1).iloc[:0],
            ],
            axis=1,
        )

        expected_inner = GeoDataFrame(expected_inner_df)

        expected_right_df = pd.concat(
            [
                self.pointdf.drop("geometry", axis=1).iloc[:0],
                pd.concat(
                    [
                        pd.Series(name="index_left", dtype="int64"),
                        pd.Series(name="index_right", dtype="int64"),
                    ],
                    axis=1,
                ),
                self.polydf,
            ],
            axis=1,
        )

        expected_right = GeoDataFrame(expected_right_df).set_index("index_right")

        expected_left_df = pd.concat(
            [
                self.pointdf.iloc[17:],
                pd.Series(name="index_right", dtype="int64"),
                self.polydf.iloc[:0].drop("geometry", axis=1),
            ],
            axis=1,
        )

        expected_left = GeoDataFrame(expected_left_df)

        assert expected_inner.equals(df_inner)
        assert expected_right.equals(df_right)
        assert expected_left.equals(df_left)

    @pytest.mark.skip("Not implemented")
    def test_sjoin_outer(self):
        df = sjoin(self.pointdf, self.polydf, how="outer")
        assert df.shape == (21, 8)

    def test_sjoin_empty_geometries(self):
        # https://github.com/geopandas/geopandas/issues/944
        empty = GeoDataFrame(geometry=[GeometryCollection()] * 3, crs=self.crs)
        df = sjoin(pd.concat([self.pointdf, empty]), self.polydf, how="left")
        assert df.shape == (24, 8)
        df2 = sjoin(self.pointdf, pd.concat([self.polydf, empty]), how="left")
        assert df2.shape == (21, 8)

    @pytest.mark.parametrize("predicate", ["intersects", "within", "contains"])
    def test_sjoin_no_valid_geoms(self, predicate):
        """Tests a completely empty GeoDataFrame."""
        empty = GeoDataFrame(geometry=[], crs=self.pointdf.crs)
        assert sjoin(self.pointdf, empty, how="inner", predicate=predicate).empty
        assert sjoin(self.pointdf, empty, how="right", predicate=predicate).empty
        assert sjoin(empty, self.pointdf, how="inner", predicate=predicate).empty
        assert sjoin(empty, self.pointdf, how="left", predicate=predicate).empty

    def test_empty_sjoin_return_duplicated_columns(self, nybb_filename):
        nybb = geopandas.read_file(nybb_filename)
        nybb2 = nybb.copy()
        nybb2.geometry = nybb2.translate(200000)  # to get non-overlapping

        result = geopandas.sjoin(nybb, nybb2)

        assert "BoroCode_right" in result.columns
        assert "BoroCode_left" in result.columns


@pytest.fixture
def world(naturalearth_lowres):
    return read_file(naturalearth_lowres)


@pytest.fixture
def cities(naturalearth_cities):
    return read_file(naturalearth_cities)


def test_sjoin_inner(world, cities):
    # GH637
    countries = world[["geometry", "name"]]
    countries = countries.rename(columns={"name": "country"})
    cities_with_country = sjoin(cities, countries, how="inner", predicate="intersects")
    assert cities_with_country.shape == (213, 4)


class TestNearest:
    @pytest.mark.parametrize(
        "how_kwargs", ({}, {"how": "inner"}, {"how": "left"}, {"how": "right"})
    )
    def test_allowed_hows(self, how_kwargs):
        left = geopandas.GeoDataFrame({"geometry": []})
        right = geopandas.GeoDataFrame({"geometry": []})
        sjoin_nearest(left, right, **how_kwargs)  # no error

    @pytest.mark.parametrize("how", ("outer", "abcde"))
    def test_invalid_hows(self, how: str):
        left = geopandas.GeoDataFrame({"geometry": []})
        right = geopandas.GeoDataFrame({"geometry": []})
        with pytest.raises(ValueError, match="`how` was"):
            sjoin_nearest(left, right, how=how)

    @pytest.mark.parametrize("distance_col", (None, "distance"))
    def test_empty_right_df_how_left(self, distance_col: str):
        # all records from left and no results from right
        left = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({"geometry": []})
        joined = sjoin_nearest(
            left,
            right,
            how="left",
            distance_col=distance_col,
        )
        assert_geoseries_equal(joined["geometry"], left["geometry"])
        assert joined["index_right"].isna().all()
        if distance_col is not None:
            assert joined[distance_col].isna().all()

    @pytest.mark.parametrize("distance_col", (None, "distance"))
    def test_empty_right_df_how_right(self, distance_col: str):
        # no records in joined
        left = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({"geometry": []})
        joined = sjoin_nearest(
            left,
            right,
            how="right",
            distance_col=distance_col,
        )
        assert joined.empty
        if distance_col is not None:
            assert distance_col in joined

    @pytest.mark.parametrize("how", ["inner", "left"])
    @pytest.mark.parametrize("distance_col", (None, "distance"))
    def test_empty_left_df(self, how, distance_col: str):
        right = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        left = geopandas.GeoDataFrame({"geometry": []})
        joined = sjoin_nearest(left, right, how=how, distance_col=distance_col)
        assert joined.empty
        if distance_col is not None:
            assert distance_col in joined

    @pytest.mark.parametrize("distance_col", (None, "distance"))
    def test_empty_left_df_how_right(self, distance_col: str):
        right = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        left = geopandas.GeoDataFrame({"geometry": []})
        joined = sjoin_nearest(
            left,
            right,
            how="right",
            distance_col=distance_col,
        )
        assert_geoseries_equal(joined["geometry"], right["geometry"])
        assert joined["index_left"].isna().all()
        if distance_col is not None:
            assert joined[distance_col].isna().all()

    @pytest.mark.parametrize("how", ["inner", "left"])
    def test_empty_join_due_to_max_distance(self, how):
        # after applying max_distance the join comes back empty
        # (as in NaN in the joined columns)
        left = geopandas.GeoDataFrame({"geometry": [Point(0, 0)]})
        right = geopandas.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2)]})
        joined = sjoin_nearest(
            left,
            right,
            how=how,
            max_distance=1,
            distance_col="distances",
        )
        expected = left.copy()
        expected["index_right"] = [np.nan]
        expected["distances"] = [np.nan]
        if how == "inner":
            expected = expected.dropna()
            expected["index_right"] = expected["index_right"].astype("int64")
        assert_geodataframe_equal(joined, expected)

    def test_empty_join_due_to_max_distance_how_right(self):
        # after applying max_distance the join comes back empty
        # (as in NaN in the joined columns)
        left = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({"geometry": [Point(2, 2)]})
        joined = sjoin_nearest(
            left,
            right,
            how="right",
            max_distance=1,
            distance_col="distances",
        )
        expected = right.copy()
        expected["index_left"] = [np.nan]
        expected["distances"] = [np.nan]
        expected = expected[["index_left", "geometry", "distances"]]
        assert_geodataframe_equal(joined, expected)

    @pytest.mark.parametrize("how", ["inner", "left"])
    def test_max_distance(self, how):
        left = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        right = geopandas.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2)]})
        joined = sjoin_nearest(
            left,
            right,
            how=how,
            max_distance=1,
            distance_col="distances",
        )
        expected = left.copy()
        expected["index_right"] = [np.nan, 0]
        expected["distances"] = [np.nan, 0]
        if how == "inner":
            expected = expected.dropna()
            expected["index_right"] = expected["index_right"].astype("int64")
        assert_geodataframe_equal(joined, expected)

    def test_max_distance_how_right(self):
        left = geopandas.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2)]})
        right = geopandas.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        joined = sjoin_nearest(
            left,
            right,
            how="right",
            max_distance=1,
            distance_col="distances",
        )
        expected = right.copy()
        expected["index_left"] = [np.nan, 0]
        expected["distances"] = [np.nan, 0]
        expected = expected[["index_left", "geometry", "distances"]]
        assert_geodataframe_equal(joined, expected)

    @pytest.mark.parametrize("how", ["inner", "left"])
    @pytest.mark.parametrize(
        "geo_left, geo_right, expected_left, expected_right, distances",
        [
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1)],
                [0, 1],
                [0, 0],
                [math.sqrt(2), 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0, 0)],
                [0, 1],
                [1, 0],
                [0, 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0, 0), Point(0, 0)],
                [0, 0, 1],
                [1, 2, 0],
                [0, 0, 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0, 0), Point(2, 2)],
                [0, 1],
                [1, 0],
                [0, 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0.25, 1)],
                [0, 1],
                [1, 0],
                [math.sqrt(0.25**2 + 1), 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(-10, -10), Point(100, 100)],
                [0, 1],
                [0, 0],
                [math.sqrt(10**2 + 10**2), math.sqrt(11**2 + 11**2)],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(x, y) for x, y in zip(np.arange(10), np.arange(10))],
                [0, 1],
                [0, 1],
                [0, 0],
            ),
            (
                [Point(0, 0), Point(1, 1), Point(0, 0)],
                [Point(1.1, 1.1), Point(0, 0)],
                [0, 1, 2],
                [1, 0, 1],
                [0, np.sqrt(0.1**2 + 0.1**2), 0],
            ),
        ],
    )
    def test_sjoin_nearest_left(
        self,
        geo_left,
        geo_right,
        expected_left: Sequence[int],
        expected_right: Sequence[int],
        distances: Sequence[float],
        how,
    ):
        left = geopandas.GeoDataFrame({"geometry": geo_left})
        right = geopandas.GeoDataFrame({"geometry": geo_right})
        expected_gdf = left.iloc[expected_left].copy()
        expected_gdf["index_right"] = expected_right
        # without distance col
        joined = sjoin_nearest(left, right, how=how)
        # inner / left join give a different row order
        check_like = how == "inner"
        assert_geodataframe_equal(expected_gdf, joined, check_like=check_like)
        # with distance col
        expected_gdf["distance_col"] = np.array(distances, dtype=float)
        joined = sjoin_nearest(left, right, how=how, distance_col="distance_col")
        assert_geodataframe_equal(expected_gdf, joined, check_like=check_like)

    @pytest.mark.parametrize(
        "geo_left, geo_right, expected_left, expected_right, distances",
        [
            ([Point(0, 0), Point(1, 1)], [Point(1, 1)], [1], [0], [0]),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0, 0)],
                [1, 0],
                [0, 1],
                [0, 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0, 0), Point(0, 0)],
                [1, 0, 0],
                [0, 1, 2],
                [0, 0, 0],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0, 0), Point(2, 2)],
                [1, 0, 1],
                [0, 1, 2],
                [0, 0, math.sqrt(2)],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(1, 1), Point(0.25, 1)],
                [1, 1],
                [0, 1],
                [0, 0.75],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(-10, -10), Point(100, 100)],
                [0, 1],
                [0, 1],
                [math.sqrt(10**2 + 10**2), math.sqrt(99**2 + 99**2)],
            ),
            (
                [Point(0, 0), Point(1, 1)],
                [Point(x, y) for x, y in zip(np.arange(10), np.arange(10))],
                [0, 1] + [1] * 8,
                list(range(10)),
                [0, 0] + [np.sqrt(x**2 + x**2) for x in np.arange(1, 9)],
            ),
            (
                [Point(0, 0), Point(1, 1), Point(0, 0)],
                [Point(1.1, 1.1), Point(0, 0)],
                [1, 0, 2],
                [0, 1, 1],
                [np.sqrt(0.1**2 + 0.1**2), 0, 0],
            ),
        ],
    )
    def test_sjoin_nearest_right(
        self,
        geo_left,
        geo_right,
        expected_left: Sequence[int],
        expected_right: Sequence[int],
        distances: Sequence[float],
    ):
        left = geopandas.GeoDataFrame({"geometry": geo_left})
        right = geopandas.GeoDataFrame({"geometry": geo_right})
        expected_gdf = right.iloc[expected_right].copy()
        expected_gdf["index_left"] = expected_left
        expected_gdf = expected_gdf[["index_left", "geometry"]]
        # without distance col
        joined = sjoin_nearest(left, right, how="right")
        assert_geodataframe_equal(expected_gdf, joined)
        # with distance col
        expected_gdf["distance_col"] = np.array(distances, dtype=float)
        joined = sjoin_nearest(left, right, how="right", distance_col="distance_col")
        assert_geodataframe_equal(expected_gdf, joined)

    @pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
    def test_sjoin_nearest_inner(self, naturalearth_lowres, naturalearth_cities):
        # check equivalency of left and inner join
        countries = read_file(naturalearth_lowres)
        cities = read_file(naturalearth_cities)
        countries = countries[["geometry", "name"]].rename(columns={"name": "country"})

        # default: inner and left give the same result
        result1 = sjoin_nearest(cities, countries, distance_col="dist")
        assert result1.shape[0] == cities.shape[0]
        result2 = sjoin_nearest(cities, countries, distance_col="dist", how="inner")
        assert_geodataframe_equal(result2, result1)
        result3 = sjoin_nearest(cities, countries, distance_col="dist", how="left")
        assert_geodataframe_equal(result3, result1, check_like=True)

        # with max_distance: rows that go above are dropped in case of inner
        result4 = sjoin_nearest(cities, countries, distance_col="dist", max_distance=1)
        assert_geodataframe_equal(
            result4, result1[result1["dist"] < 1], check_like=True
        )
        result5 = sjoin_nearest(
            cities, countries, distance_col="dist", max_distance=1, how="left"
        )
        assert result5.shape[0] == cities.shape[0]
        result5 = result5.dropna()
        result5["index_right"] = result5["index_right"].astype("int64")
        assert_geodataframe_equal(result5, result4, check_like=True)

    @pytest.mark.parametrize(
        "max_distance,expected", [(None, [1, 3, 3, 1, 2]), (1.1, [3, 3, 1, 2])]
    )
    def test_sjoin_nearest_exclusive(self, max_distance, expected):
        geoms = shapely.points(np.arange(3), np.arange(3))
        geoms = np.append(geoms, [Point(1, 2)])

        df = geopandas.GeoDataFrame({"geometry": geoms})
        result = df.sjoin_nearest(
            df, max_distance=max_distance, distance_col="dist", exclusive=True
        )

        assert_series_equal(
            result["index_right"].reset_index(drop=True),
            pd.Series(expected),
            check_names=False,
        )

        if max_distance:
            assert result["dist"].max() <= max_distance
