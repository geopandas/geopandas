from distutils.version import LooseVersion

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, GeometryCollection

import geopandas
from geopandas import GeoDataFrame, GeoSeries, read_file, sindex, sjoin

from pandas.testing import assert_frame_equal
import pytest


pytestmark = pytest.mark.skipif(
    not sindex.has_sindex(), reason="sjoin requires spatial index"
)


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


class TestSpatialJoin:
    @pytest.mark.parametrize("dfs", ["default-index", "string-index"], indirect=True)
    def test_crs_mismatch(self, dfs):
        index, df1, df2, expected = dfs
        df1.crs = "epsg:4326"
        with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
            sjoin(df1, df2)

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
    @pytest.mark.parametrize("op", ["intersects", "contains", "within"])
    def test_inner(self, op, dfs):
        index, df1, df2, expected = dfs

        res = sjoin(df1, df2, how="inner", op=op)

        exp = expected[op].dropna().copy()
        exp = exp.drop("geometry_y", axis=1).rename(columns={"geometry_x": "geometry"})
        exp[["df1", "df2"]] = exp[["df1", "df2"]].astype("int64")
        if index == "default-index":
            exp[["index_left", "index_right"]] = exp[
                ["index_left", "index_right"]
            ].astype("int64")
        if index == "named-index":
            exp[["df1_ix", "df2_ix"]] = exp[["df1_ix", "df2_ix"]].astype("int64")
            exp = exp.set_index("df1_ix").rename(columns={"df2_ix": "index_right"})
        if index in ["default-index", "string-index"]:
            exp = exp.set_index("index_left")
            exp.index.name = None
        if index == "multi-index":
            exp = exp.set_index(["level_0_x", "level_1_x"]).rename(
                columns={"level_0_y": "index_right0", "level_1_y": "index_right1"}
            )
            exp.index.names = df1.index.names
        if index == "named-multi-index":
            exp = exp.set_index(["df1_ix1", "df1_ix2"]).rename(
                columns={"df2_ix1": "index_right0", "df2_ix2": "index_right1"}
            )
            exp.index.names = df1.index.names

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
    @pytest.mark.parametrize("op", ["intersects", "contains", "within"])
    def test_left(self, op, dfs):
        index, df1, df2, expected = dfs

        res = sjoin(df1, df2, how="left", op=op)

        if index in ["default-index", "string-index"]:
            exp = expected[op].dropna(subset=["index_left"]).copy()
        elif index == "named-index":
            exp = expected[op].dropna(subset=["df1_ix"]).copy()
        elif index == "multi-index":
            exp = expected[op].dropna(subset=["level_0_x"]).copy()
        elif index == "named-multi-index":
            exp = expected[op].dropna(subset=["df1_ix1"]).copy()
        exp = exp.drop("geometry_y", axis=1).rename(columns={"geometry_x": "geometry"})
        exp["df1"] = exp["df1"].astype("int64")
        if index == "default-index":
            exp["index_left"] = exp["index_left"].astype("int64")
            # TODO: in result the dtype is object
            res["index_right"] = res["index_right"].astype(float)
        elif index == "named-index":
            exp[["df1_ix"]] = exp[["df1_ix"]].astype("int64")
            exp = exp.set_index("df1_ix").rename(columns={"df2_ix": "index_right"})
        if index in ["default-index", "string-index"]:
            exp = exp.set_index("index_left")
            exp.index.name = None
        if index == "multi-index":
            exp = exp.set_index(["level_0_x", "level_1_x"]).rename(
                columns={"level_0_y": "index_right0", "level_1_y": "index_right1"}
            )
            exp.index.names = df1.index.names
        if index == "named-multi-index":
            exp = exp.set_index(["df1_ix1", "df1_ix2"]).rename(
                columns={"df2_ix1": "index_right0", "df2_ix2": "index_right1"}
            )
            exp.index.names = df1.index.names

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
        empty = sjoin(not_in, polygons, how="left", op="intersects")
        assert empty.index_right.isnull().all()
        empty = sjoin(not_in, polygons, how="right", op="intersects")
        assert empty.index_left.isnull().all()
        empty = sjoin(not_in, polygons, how="inner", op="intersects")
        assert empty.empty

    @pytest.mark.parametrize("op", ["intersects", "contains", "within"])
    @pytest.mark.parametrize(
        "empty",
        [
            GeoDataFrame(geometry=[GeometryCollection(), GeometryCollection()]),
            GeoDataFrame(geometry=GeoSeries()),
        ],
    )
    def test_join_with_empty(self, op, empty):
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
        result = sjoin(empty, polygons, how="left", op=op)
        assert result.index_right.isnull().all()
        result = sjoin(empty, polygons, how="right", op=op)
        assert result.index_left.isnull().all()
        result = sjoin(empty, polygons, how="inner", op=op)
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
    @pytest.mark.parametrize("op", ["intersects", "contains", "within"])
    def test_right(self, op, dfs):
        index, df1, df2, expected = dfs

        res = sjoin(df1, df2, how="right", op=op)

        if index in ["default-index", "string-index"]:
            exp = expected[op].dropna(subset=["index_right"]).copy()
        elif index == "named-index":
            exp = expected[op].dropna(subset=["df2_ix"]).copy()
        elif index == "multi-index":
            exp = expected[op].dropna(subset=["level_0_y"]).copy()
        elif index == "named-multi-index":
            exp = expected[op].dropna(subset=["df2_ix1"]).copy()
        exp = exp.drop("geometry_x", axis=1).rename(columns={"geometry_y": "geometry"})
        exp["df2"] = exp["df2"].astype("int64")
        if index == "default-index":
            exp["index_right"] = exp["index_right"].astype("int64")
            res["index_left"] = res["index_left"].astype(float)
        elif index == "named-index":
            exp[["df2_ix"]] = exp[["df2_ix"]].astype("int64")
            exp = exp.set_index("df2_ix").rename(columns={"df1_ix": "index_left"})
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
            exp = exp.set_index(["df2_ix1", "df2_ix2"]).rename(
                columns={"df1_ix1": "index_left0", "df1_ix2": "index_left1"}
            )
            exp.index.names = df2.index.names

        # GH 1364 fix of behaviour was done in pandas 1.1.0
        if op == "within" and str(pd.__version__) >= LooseVersion("1.1.0"):
            exp = exp.sort_index()

        assert_frame_equal(res, exp, check_index_type=False)


class TestSpatialJoinNYBB:
    def setup_method(self):
        nybb_filename = geopandas.datasets.get_path("nybb")
        self.polydf = read_file(nybb_filename)
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
            assert row.geometry.type == "Point"
        assert "pointattr1" in df.columns
        assert "BoroCode" in df.columns

    def test_sjoin_right(self):
        # the inverse of left
        df = sjoin(self.pointdf, self.polydf, how="right")
        df2 = sjoin(self.polydf, self.pointdf, how="left")
        assert df.shape == (12, 8)
        assert df.shape == df2.shape
        for i, row in df.iterrows():
            assert row.geometry.type == "MultiPolygon"
        for i, row in df2.iterrows():
            assert row.geometry.type == "MultiPolygon"

    def test_sjoin_inner(self):
        df = sjoin(self.pointdf, self.polydf, how="inner")
        assert df.shape == (11, 8)

    def test_sjoin_op(self):
        # points within polygons
        df = sjoin(self.pointdf, self.polydf, how="left", op="within")
        assert df.shape == (21, 8)
        assert df.loc[1]["BoroName"] == "Staten Island"

        # points contain polygons? never happens so we should have nulls
        df = sjoin(self.pointdf, self.polydf, how="left", op="contains")
        assert df.shape == (21, 8)
        assert np.isnan(df.loc[1]["Shape_Area"])

    def test_sjoin_bad_op(self):
        # AttributeError: 'Point' object has no attribute 'spandex'
        with pytest.raises(ValueError):
            sjoin(self.pointdf, self.polydf, how="left", op="spandex")

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
        empty = GeoDataFrame(geometry=[GeometryCollection()] * 3)
        df = sjoin(self.pointdf.append(empty), self.polydf, how="left")
        assert df.shape == (24, 8)
        df2 = sjoin(self.pointdf, self.polydf.append(empty), how="left")
        assert df2.shape == (21, 8)

    @pytest.mark.parametrize("op", ["intersects", "within", "contains"])
    def test_sjoin_no_valid_geoms(self, op):
        """Tests a completely empty GeoDataFrame."""
        empty = GeoDataFrame(geometry=[], crs=self.pointdf.crs)
        assert sjoin(self.pointdf, empty, how="inner", op=op).empty
        assert sjoin(self.pointdf, empty, how="right", op=op).empty
        assert sjoin(empty, self.pointdf, how="inner", op=op).empty
        assert sjoin(empty, self.pointdf, how="left", op=op).empty


class TestSpatialJoinNaturalEarth:
    def setup_method(self):
        world_path = geopandas.datasets.get_path("naturalearth_lowres")
        cities_path = geopandas.datasets.get_path("naturalearth_cities")
        self.world = read_file(world_path)
        self.cities = read_file(cities_path)

    def test_sjoin_inner(self):
        # GH637
        countries = self.world[["geometry", "name"]]
        countries = countries.rename(columns={"name": "country"})
        cities_with_country = sjoin(
            self.cities, countries, how="inner", op="intersects"
        )
        assert cities_with_country.shape == (172, 4)
