import os
import shutil
import tempfile

import pandas as pd

from shapely.geometry import Point

import geopandas
from geopandas import GeoDataFrame, read_file, GeoSeries

from geopandas.tests.util import PACKAGE_DIR


class TestDataFrame:
    def setup_method(self):
        N = 10

        nybb_filename = geopandas.datasets.get_path("nybb")
        self.df = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = "epsg:4326"
        self.df2 = GeoDataFrame(
            [
                {
                    "geometry": Point(x, y),
                    "value1": x + y,
                    "value2": x * y,
                }
                for x, y in zip(range(N), range(N))
            ],
            crs=self.crs,
        )
        self.df2["geometry2"] = self.df2["geometry"]  # want geometry2 to be a geoseries
        self.df3 = read_file(
            os.path.join(PACKAGE_DIR, "geopandas", "tests", "data", "null_geom.geojson")
        )

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        assert type(self.df2) is GeoDataFrame
        assert self.df2.crs == self.crs
        print(type(self.df2["geometry2"]))
        print(type(self.df2["geometry"]))

    def test_getitem_passing(self):
        assert type(self.df2[["value1", "value2"]]) is pd.DataFrame
        assert type(self.df2["geometry"]) is GeoSeries
        assert type(self.df2["value1"]) is pd.Series
        assert type(self.df2[["geometry"]]) is GeoDataFrame
        assert type(self.df2[["value1"]]) is pd.DataFrame

    def test_getitem(self):
        assert type(self.df2["geometry2"]) is GeoSeries
        assert type(self.df2[["geometry2"]]) is GeoDataFrame

    def test_loc(self):
        assert type(self.df2.loc[:, ["value1", "value2"]]) is pd.DataFrame
        assert type(self.df2.loc[:, "geometry"]) is GeoSeries
        assert type(self.df2.loc[:, "geometry2"]) is GeoSeries
        assert type(self.df2["value1"]) is pd.Series
        assert type(self.df2.loc[:, ["geometry"]]) is GeoDataFrame
        assert type(self.df2.loc[:, ["value1"]]) is pd.DataFrame
        assert type(self.df2.loc[:, ["geometry2"]]) is GeoDataFrame

    def test_squeeze(self):
        assert type(self.df2[["geometry"]].squeeze()) is GeoSeries
        assert type(self.df2[["geometry2"]].squeeze()) is GeoSeries

    def test_to_frame(self):
        assert type(self.df2["geometry"].to_frame()) is GeoDataFrame
        assert type(self.df2["geometry2"].to_frame()) is GeoDataFrame

    def test_iloc(self):
        assert type(self.df2.iloc[:, 1:3]) is pd.DataFrame
        assert type(self.df2.iloc[:, 0]) is GeoSeries
        assert type(self.df2.iloc[:, 3]) is GeoSeries
        assert type(self.df2.iloc[:, 1]) is pd.Series
        assert type(self.df2.iloc[:, [0]]) is GeoDataFrame
        assert type(self.df2.iloc[:, [1]]) is pd.DataFrame
        assert type(self.df2.iloc[:, [3]]) is GeoDataFrame

    def test_reindex(self):
        df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))

        df_reindex = df.reindex(columns=["name"])
        print(df)
        print(df_reindex)
        assert type(df_reindex) is pd.DataFrame

    def test_drop(self):
        assert type(self.df.drop(columns="geometry")) is pd.DataFrame

    def test_apply(self):
        assert type(self.df2["geometry"].apply(lambda x: x)) is GeoSeries
        assert type(self.df2["geometry2"].apply(lambda x: x)) is GeoSeries
        assert type(self.df2["value1"].apply(lambda x: x)) is pd.Series

        assert type(self.df2.apply(lambda x: x)) is GeoDataFrame
        assert type(self.df2[["value1", "value2"]].apply(lambda x: x)) is pd.DataFrame
