from warnings import warn

from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from shapely.geometry import Point, Polygon, GeometryCollection

import geopandas
from geopandas import GeoDataFrame, GeoSeries, base, read_file, sjoin, sjoin_nearest
from geopandas.tools.sjoin import RTREE_VERSION

import pytest


# save rtree version
RTREE_VERSION_NEWER_OR_EQAL_094 = RTREE_VERSION >= LooseVersion("0.9.4")


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
    part1 = (
        df1.copy()
        .iloc[[0, 0, 1, 2]]
        .reset_index()
        .rename(columns={"index": "index_left"})
    )
    part2 = (
        df2.copy()
        .iloc[[0, 1, 1, 1, 2]]
        .reset_index()
        .rename(columns={"index": "index_right"})
    )
    part1["_merge"] = [0, 1, 2, 3]
    part2["_merge"] = [0, 1, 2, 3, 4]
    exp = pd.merge(part1, part2, on="_merge", how="outer")
    exp = exp.drop("_merge", axis=1).copy()

    return [request.param, df1, df2, exp]


@pytest.mark.skipif(not base.HAS_SINDEX, reason="Rtree absent, skipping")
class TestSpatialJoin:
    @pytest.mark.parametrize("dfs", ["default-index", "string-index"], indirect=True)
    def test_crs_mismatch(self, dfs):
        index, df1, df2, expected = dfs
        df1.crs = "epsg:4326"
        with pytest.warns(UserWarning):
            sjoin_nearest(df1, df2)

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
    def test_inner(self, dfs):
        index, df1, df2, expected = dfs

        res = sjoin_nearest(df1, df2, how="inner")
        exp = expected.dropna().copy()
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
    def test_left(self, dfs):
        index, df1, df2, expected = dfs

        res = sjoin_nearest(df1, df2, how="left")

        if index in ["default-index", "string-index"]:
            exp = expected.dropna(subset=["index_left"]).copy()
        elif index == "named-index":
            exp = expected.dropna(subset=["df1_ix"]).copy()
        elif index == "multi-index":
            exp = expected.dropna(subset=["level_0_x"]).copy()
        elif index == "named-multi-index":
            exp = expected.dropna(subset=["df1_ix1"]).copy()
        exp = exp.drop("geometry_y", axis=1).rename(columns={"geometry_x": "geometry"})
        exp["df1"] = exp["df1"].astype("int64")
        if index == "default-index":
            exp["index_left"] = exp["index_left"].astype("int64")
            # dtypes for index may get changed if there are non-finite values
            if res["index_right"].dtype != exp["index_right"].dtype:
                res["index_right"] = res["index_right"].astype(exp["index_right"].dtype)
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
        result = sjoin_nearest(empty, polygons, how="left")
        assert result.index_right.isnull().all()
        result = sjoin_nearest(empty, polygons, how="right")
        assert result.index_left.isnull().all()
        result = sjoin_nearest(empty, polygons, how="inner")
        assert result.empty

    @pytest.mark.parametrize("dfs", ["default-index", "string-index"], indirect=True)
    def test_sjoin_invalid_args(self, dfs):
        index, df1, df2, expected = dfs

        with pytest.raises(ValueError, match="'left_df' should be GeoDataFrame"):
            sjoin_nearest(df1.geometry, df2)

        with pytest.raises(ValueError, match="'right_df' should be GeoDataFrame"):
            sjoin_nearest(df1, df2.geometry)

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
    def test_right(self, dfs):
        index, df1, df2, expected = dfs

        res = sjoin_nearest(df1, df2, how="right")

        if index in ["default-index", "string-index"]:
            exp = expected.dropna(subset=["index_right"]).copy()
        elif index == "named-index":
            exp = expected.dropna(subset=["df2_ix"]).copy()
        elif index == "multi-index":
            exp = expected.dropna(subset=["level_0_y"]).copy()
        elif index == "named-multi-index":
            exp = expected.dropna(subset=["df2_ix1"]).copy()
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
        assert_frame_equal(res, exp, check_index_type=False)


@pytest.mark.skipif(base.HAS_SINDEX, reason="Rtree present, skipping")
class TestNoRtree:
    """Tests that an error is raised when no rtree is present and spatial joins
     are called.
    """

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

    def test_no_rtree(self):
        with pytest.raises(RuntimeError):
            sjoin_nearest(self.pointdf, self.polydf)


@pytest.mark.skipif(not base.HAS_SINDEX, reason="Rtree absent, skipping")
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
        res = sjoin_nearest(self.polydf, self.pointdf, how="left")
        assert self.polydf.geometry.name == res.geometry.name

    def test_sjoin_left(self):
        df = sjoin_nearest(self.pointdf, self.polydf, how="left")
        assert df.shape == (21, 8)
        for i, row in df.iterrows():
            assert row.geometry.type == "Point"
        assert "pointattr1" in df.columns
        assert "BoroCode" in df.columns

    def test_sjoin_right(self):
        # the inverse of left
        df = sjoin_nearest(self.pointdf, self.polydf, how="right")
        assert df.shape == (22, 8)
        for i, row in df.iterrows():
            assert row.geometry.type == "MultiPolygon"

    def test_sjoin_inner(self):
        df = sjoin_nearest(self.pointdf, self.polydf, how="inner")
        assert df.shape == (21, 8)

    def test_sjoin_points_in_poly(self):
        # points within polygons
        df = sjoin_nearest(self.pointdf, self.polydf, how="left")
        assert df.shape == (21, 8)
        assert df.loc[1]["BoroName"] == "Staten Island"

    def test_sjoin_nearest_zero_radius_equals_sjoin_intersect(self):
        # results should be equal since search_radius=0 is really intersection
        sjn = sjoin_nearest(self.pointdf, self.polydf, how="left", search_radius=0)
        sj = sjoin(self.pointdf, self.polydf, how="left")
        assert_frame_equal(sjn, sj)

    def test_sjoin_duplicate_column_name(self):
        pointdf2 = self.pointdf.rename(columns={"pointattr1": "Shape_Area"})
        df = sjoin_nearest(pointdf2, self.polydf, how="left")
        assert "Shape_Area_left" in df.columns
        assert "Shape_Area_right" in df.columns

    @pytest.mark.parametrize("how", ["left", "right", "inner"])
    def test_sjoin_named_index(self, how):
        # original index names should be unchanged
        pointdf2 = self.pointdf.copy()
        pointdf2.index.name = "pointid"
        polydf = self.polydf.copy()
        polydf.index.name = "polyid"

        res = sjoin_nearest(pointdf2, polydf, how=how)
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
        df = sjoin_nearest(self.pointdf, self.polydf, how="left")
        assert df.shape == (21, 8)
        df = sjoin_nearest(self.polydf, self.pointdf, how="left")
        assert df.shape == (12, 8)

    def test_no_matches_in_radius_geometry(self):
        # Note: these tests are for correctly returning GeoDataFrame
        # when result of the join is empty

        df_inner = sjoin_nearest(
            self.pointdf.iloc[17:], self.polydf, how="inner", search_radius=0
        )
        df_left = sjoin_nearest(
            self.pointdf.iloc[17:], self.polydf, how="left", search_radius=0
        )
        df_right = sjoin_nearest(
            self.pointdf.iloc[17:], self.polydf, how="right", search_radius=0
        )

        expected_inner_df = pd.concat(
            [
                self.pointdf.iloc[:0],
                pd.Series(name="index_right", dtype="int64"),
                self.polydf.drop("geometry", axis=1).iloc[:0],
            ],
            axis=1,
        )

        expected_inner = GeoDataFrame(expected_inner_df, crs="epsg:4326")

        data_from_right = (
            self.pointdf.iloc[0 : self.polydf.shape[0]]
            .drop("geometry", axis=1)
            .applymap(lambda x: np.nan)
        )
        expected_right_df = pd.concat(
            [
                pd.Series(name="index_left", dtype="int64"),
                pd.Series(data=data_from_right.index, name="index_right"),
                data_from_right,
                self.polydf,
            ],
            axis=1,
        )

        expected_right = GeoDataFrame(expected_right_df, crs="epsg:4326").set_index(
            "index_right"
        )

        expected_left_df = pd.concat(
            [
                self.pointdf.iloc[17:],
                pd.Series(name="index_right", dtype="int64"),
                self.polydf.iloc[:0].drop("geometry", axis=1),
            ],
            axis=1,
        )

        expected_left = GeoDataFrame(expected_left_df, crs="epsg:4326")

        # check results
        assert expected_inner.equals(df_inner)
        assert expected_right.equals(df_right)
        assert expected_left.equals(df_left)

    @pytest.mark.parametrize("how", ["howdy", "riiiight"])
    def test_invalid_hows(self, how):
        with pytest.raises(ValueError):
            sjoin_nearest(self.pointdf, self.polydf, how=how)

    def test_sjoin_empty_geometries(self):
        # https://github.com/geopandas/geopandas/issues/944
        empty = GeoDataFrame(geometry=[GeometryCollection()] * 3)
        empty.crs = self.pointdf.crs  # just to avoid warnings clutter

        df = sjoin_nearest(self.pointdf, empty, how="left")
        assert df.shape == (self.pointdf.shape[0], 4)
        df1 = sjoin_nearest(empty, self.pointdf, how="left")
        assert df1.shape == (empty.shape[0], 4)
        df2 = sjoin_nearest(
            self.pointdf.append(empty, sort=False), self.polydf, how="left"
        )
        assert df2.shape == (self.pointdf.append(empty, sort=False).shape[0], 8)
        df3 = sjoin_nearest(
            self.pointdf, self.polydf.append(empty, sort=False), how="left"
        )
        assert df3.shape == (self.pointdf.shape[0], 8)

    def test_join_suffix_valid(self):
        for lsuffix in ["test", ""]:
            pointdf = self.pointdf.copy()
            polydf = self.polydf.copy()
            # add column with invalid name
            polydf["index_left"] = None
            sjoin_nearest(pointdf, polydf, lsuffix=lsuffix)
        for rsuffix in ["test", ""]:
            pointdf = self.pointdf.copy()
            polydf = self.polydf.copy()
            # add column with invalid name
            polydf["index_right"] = None
            sjoin_nearest(pointdf, polydf, rsuffix=rsuffix)

    def test_join_suffix_invalid(self):
        for lsuffix in ["left", "test", ""]:
            pointdf = self.pointdf.copy()
            polydf = self.polydf.copy()
            # add column with invalid name
            polydf["index_%s" % lsuffix] = None
            with pytest.raises(ValueError):
                sjoin_nearest(pointdf, polydf, lsuffix=lsuffix)
        for rsuffix in ["right", "test", ""]:
            pointdf = self.pointdf.copy()
            polydf = self.polydf.copy()
            # add column with invalid name
            polydf["index_%s" % rsuffix] = None
            with pytest.raises(ValueError):
                sjoin_nearest(pointdf, polydf, rsuffix=rsuffix)


@pytest.mark.skipif(not base.HAS_SINDEX, reason="Rtree absent, skipping")
class TestSearchRadius:
    """Tests that involve the search_radius parameter.
    """

    def setup_method(self):
        # load dataset
        world_path = geopandas.datasets.get_path("naturalearth_lowres")
        cities_path = geopandas.datasets.get_path("naturalearth_cities")
        world = read_file(world_path)
        countries = world[["geometry", "name"]].rename(columns={"name": "country"})
        cities = read_file(cities_path)
        # save datasets
        self.countries = countries
        self.cities = cities
        # comopute sjoin_nearest with no filters, will be re-used many times
        self.sjn_None = {}
        for how in ["left", "inner", "right"]:
            self.sjn_None[how] = sjoin_nearest(
                self.cities, self.countries, how=how, nearest_distances=True
            )

    # -------------------------- search_radius=None ----------------------------
    def test_radius_None_how_inner(self):
        sjn = self.sjn_None["inner"]
        sj = sjoin(self.cities, self.countries, how="inner",)
        # most results are the same as using op="intersects" with sjoin
        # but we have some new matches
        assert sjn.shape == (202, 5)  # empirically checked

        # drop dist > 0 and nearest_distances column and compare
        sjn_filtered = (
            sjn[sjn["nearest_distances"] == 0]
            .drop("nearest_distances", axis=1)
            .sort_values("name")
            .reset_index()
        )
        sj_filtered = sj.sort_values("name").reset_index()
        assert_frame_equal(sj_filtered, sjn_filtered, check_dtype=False)

    def test_radius_None_how_left(self):
        sjn = self.sjn_None["left"]
        sj = sjoin(self.cities, self.countries, how="left",)
        # most results are the same as using op="intersects" with sjoin
        # but we have some new matches that were manually checked
        assert sjn.shape == (self.cities.shape[0], 5)

        # drop dist > 0 and nearest_distances column and compare
        sjn_filtered = (
            sjn[sjn["nearest_distances"] == 0]
            .drop("nearest_distances", axis=1)
            .sort_values("name")
            .reset_index()
        )
        # need to drop na values in sjoin result
        sj_filtered = sj.dropna(axis=0).sort_values("name").reset_index()
        assert_frame_equal(sj_filtered, sjn_filtered, check_dtype=False)

    def test_radius_None_how_right(self):
        sjn = self.sjn_None["right"]
        sj = sjoin(self.cities, self.countries, how="right",)
        # most results are the same as using op="intersects" with sjoin
        # but we have some new matches
        assert sjn.shape == (212, 5)  # empirically checked
        # drop dist > 0 and nearest_distances column and compare
        sjn_filtered = (
            sjn[sjn["nearest_distances"] == 0]
            .drop("nearest_distances", axis=1)
            .sort_values("name")
            .reset_index()
        )
        # need to drop na values in sjoin result
        sj_filtered = sj.dropna(axis=0).sort_values("name").reset_index()

        assert_frame_equal(sj_filtered, sjn_filtered, check_dtype=False)

    # -------------------------- search_radius=0 -------------------------------

    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    def test_radius_0(self, how):
        """All results with search_radius=0 should be exactly the same as sjoin.
        """
        sjn = sjoin_nearest(self.cities, self.countries, how=how, search_radius=0)
        sj = sjoin(self.cities, self.countries, how=how)
        sjn = sjn.sort_values("name").reset_index()
        sj = sj.sort_values("name").reset_index()

        assert_frame_equal(sjn, sj, check_dtype=False)

    # -------------------- other search_radius tests --------------------------
    # radii chosen empirically for this dataset
    @pytest.mark.parametrize("search_radius", [20, 100])
    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    def test_radius_large(self, how, search_radius):
        """A large radius should be equivalent to radius=None
        """
        sjn_None = self.sjn_None[how].drop("nearest_distances", axis=1)
        sjn = sjoin_nearest(
            self.cities, self.countries, how=how, search_radius=search_radius,
        )
        assert_frame_equal(sjn_None, sjn)

    # radii chosen empirically for this dataset
    @pytest.mark.parametrize("search_radius", [0, 5, 15])
    @pytest.mark.parametrize("how", ["left", "inner"])
    def test_radius_small(self, how, search_radius):
        """
        A small radius should give different results than radius=None.
        Specifically, it should be missing only results that were outside of that
         radius.
        """
        sjn_None = self.sjn_None[how]
        sjn = sjoin_nearest(
            self.cities,
            self.countries,
            how=how,
            search_radius=search_radius,
            nearest_distances=True,
        )
        # filter sjn_None to remove results outside radius
        sjn_None_filtered = (
            sjn_None[sjn_None["nearest_distances"] <= search_radius]
            .sort_values("name")
            .reset_index()
        )
        # need to drop na values in sjoin result
        sjn_filtered = sjn.dropna(axis=0).sort_values("name").reset_index()
        # skip checking dtypes
        assert_frame_equal(sjn_None_filtered, sjn_filtered, check_dtype=False)

    @pytest.mark.parametrize("search_radius", [0, 5, 15])
    def test_radius_small_how_right(self, search_radius):
        """
        Same idea as test_radius_small, except here we can check specifically that
         matches with nearest_distances > search_radius in sjn_None
         show up as NaN in sjn.
        """
        sjn_None = self.sjn_None["right"]
        sjn = sjoin_nearest(
            self.cities,
            self.countries,
            how="right",
            search_radius=search_radius,
            nearest_distances=True,
        )
        # filter sjn_None to by radius or NaN
        sjn_None = sjn_None[
            (sjn_None["nearest_distances"] > search_radius)
            | (sjn_None.isnull().any(axis=1))
        ]
        # filter sjn by null
        sjn = sjn[sjn.isnull().any(axis=1)]
        # need to drop na values in sjoin result
        # we can't guarantee that they are equal, but we do know that any country
        # that didn't match in sjn should have distance > search_radius in sjn_None
        # or also be NaN in sjn_None (i.e. it wasn't closest to any city)
        sjn_countries = set(sjn["country"])
        sjn_None_countries = set(sjn_None["country"])
        assert sjn_countries.issubset(sjn_None_countries)

    @pytest.mark.parametrize("search_radius", [-1])
    def test_invalid_radius(self, search_radius):
        with pytest.raises(ValueError):
            sjoin_nearest(
                self.cities,
                self.countries,
                how="right",
                search_radius=search_radius,
                nearest_distances=True,
            )


@pytest.mark.skipif(not base.HAS_SINDEX, reason="Rtree absent, skipping")
class TestMaxSearchNeighbors:
    """Tests that involve the max_search_neighbors parameter.
    """

    def setup_method(self):
        world_path = geopandas.datasets.get_path("naturalearth_lowres")
        cities_path = geopandas.datasets.get_path("naturalearth_cities")
        world = read_file(world_path)
        countries = world[["geometry", "name"]]
        self.countries = countries.rename(columns={"name": "country"})
        self.cities = read_file(cities_path)
        self.sjn_None = {}
        for how in ["left", "inner", "right"]:
            self.sjn_None[how] = sjoin_nearest(
                self.cities, self.countries, how=how, nearest_distances=True
            )

    @pytest.mark.parametrize("n", [-1, 1.5, 1.0])
    def test_invalid_max_search_neighbors(self, n):
        with pytest.raises(ValueError):
            sjoin_nearest(
                self.cities,
                self.countries,
                how="right",
                nearest_distances=True,
                max_search_neighbors=n,
            )

    # n chosen empirically for this dataset
    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("how", ["left", "inner"])
    def test_neighbors_small(self, how, n):
        """
        A small n may or may not be different than None but should be the same
         for large n.
        For this dataset n=1 gives different results but n>=2 does not
        This tests checks that for n=1, if we get "unexpected" matches in sjn,
         the distance be larger than the match in sjn_None
         (i.e., sjn_None is "right")
        """
        sjn_None = self.sjn_None[how]
        if RTREE_VERSION_NEWER_OR_EQAL_094:
            with pytest.warns(UserWarning) as warnings:
                sjn = sjoin_nearest(
                    self.cities,
                    self.countries,
                    how=how,
                    nearest_distances=True,
                    max_search_neighbors=n,
                )
                assert not warnings
                warn("any warning")  # just to pass test
        else:
            with pytest.warns(UserWarning) as warnings:
                sjn = sjoin_nearest(
                    self.cities,
                    self.countries,
                    how=how,
                    nearest_distances=True,
                    max_search_neighbors=n,
                )

        # get difference between the resulsts
        diff = pd.concat([sjn_None, sjn]).drop_duplicates(
            subset=["country", "name"], keep=False
        )
        # check that in every case, sjn_None is "right"
        for name in diff["name"]:
            sjn_None_dist = sjn_None.loc[sjn_None["name"] == name, "nearest_distances"]
            sjn_dist = sjn.loc[sjn["name"] == name, "nearest_distances"]
            assert sjn_None_dist.squeeze() < sjn_dist.squeeze()

    # n chosen empirically for this dataset
    @pytest.mark.skipif(not RTREE_VERSION_NEWER_OR_EQAL_094, reason="Rtree <= 0.9.4")
    @pytest.mark.parametrize("n", [2, 10, 50])
    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    def test_neighbors_large(self, how, n):
        """
        A small n may or may or may not be different than None but, large n
         should be the same as n=None.
        For this dataset n=1 gives different results but n>=2 does not.
        This tests checks that the result is the same for n>=2.
        """
        sjn_None = self.sjn_None[how]
        sjn = sjoin_nearest(
            self.cities,
            self.countries,
            how=how,
            nearest_distances=True,
            max_search_neighbors=n,
        )
        # check that in every case, sjn_None is "right"
        assert_frame_equal(
            sjn.sort_values("name").reset_index(),
            sjn_None.sort_values("name").reset_index(),
        )

    # n chosen empirically for this dataset
    @pytest.mark.skipif(RTREE_VERSION_NEWER_OR_EQAL_094, reason="Rtree > 0.9.4")
    @pytest.mark.parametrize("n", [2, 10, 50])
    def test_old_rtree_warning(self, n):
        """
        Checks that using max_search_neighbors raises a warning on old rtree
         versions.
        """
        if n is None:
            with pytest.warns(None) as raised_warnings:
                sjoin_nearest(
                    self.cities.iloc[:2],
                    self.countries.iloc[:2],
                    max_search_neighbors=n,
                )
            assert not raised_warnings  # fails if any warnings were raised
        else:
            with pytest.warns(UserWarning):
                sjoin_nearest(
                    self.cities.iloc[:2],
                    self.countries.iloc[:2],
                    max_search_neighbors=n,
                )


@pytest.mark.skipif(not base.HAS_SINDEX, reason="Rtree absent, skipping")
class TestNeighborsAndRadius:
    """Tests that use both max_search_neighbors and search_radius.
    """

    def setup_method(self):
        world_path = geopandas.datasets.get_path("naturalearth_lowres")
        cities_path = geopandas.datasets.get_path("naturalearth_cities")
        world = read_file(world_path)
        countries = world[["geometry", "name"]]
        self.countries = countries.rename(columns={"name": "country"})
        self.cities = read_file(cities_path)
        self.sjn_None = {}
        for how in ["left", "inner", "right"]:
            self.sjn_None[how] = sjoin_nearest(
                self.cities, self.countries, how=how, nearest_distances=True
            )

    # radii chosen empirically for this dataset
    # n chosen empirically for this dataset
    @pytest.mark.parametrize("search_radius", [20, 100])
    @pytest.mark.parametrize("n", [5, 10, 50])
    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    @pytest.mark.skipif(not RTREE_VERSION_NEWER_OR_EQAL_094, reason="Rtree <= 0.9.4")
    def test_combined_params_new_rtree(self, how, search_radius, n):
        """A small n may or may or may not be different than None but, large n
        should be the same as n=None.
        For this dataset n=1 gives different results but n>=2 does not.
        This tests checks that the result is the same for n>=2 AND a large
         search radius.
        """
        sjn_None = self.sjn_None[how].drop("nearest_distances", axis=1)
        sjn = sjoin_nearest(
            self.cities,
            self.countries,
            how=how,
            search_radius=search_radius,
            max_search_neighbors=n,
        )
        assert_frame_equal(sjn_None, sjn)

    # radii chosen empirically for this dataset
    # n chosen empirically for this dataset
    @pytest.mark.parametrize("search_radius", [20, 100])
    @pytest.mark.parametrize("n", [10, 50])
    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    def test_combined_params_old_rtree(self, how, search_radius, n):
        """Same as test_combined_params_new_rtree, but the minimum n is larger
         due to bugs in rtree<0.9.4.
        """
        sjn_None = self.sjn_None[how].drop("nearest_distances", axis=1)
        sjn = sjoin_nearest(
            self.cities,
            self.countries,
            how=how,
            search_radius=search_radius,
            max_search_neighbors=n,
        )
        assert_frame_equal(sjn_None, sjn)
