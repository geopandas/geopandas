import pandas as pd

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries


class TestDataFrame:
    def setup_method(self):
        N = 10
        self.crs = "epsg:4326"
        self.df = GeoDataFrame(
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
        self.df["geometry2"] = self.df["geometry"]  # want geometry2 to be a geoseries

    @staticmethod
    def _check_metadata(gdf, geometry_column_name="geometry", crs="epsg:4326"):

        assert gdf._geometry_column_name == geometry_column_name
        assert gdf.crs == crs

    def test_getitem(self):
        assert type(self.df[["value1", "value2"]]) is pd.DataFrame
        assert type(self.df["geometry"]) is GeoSeries
        assert type(self.df["geometry2"]) is GeoSeries
        assert type(self.df["value1"]) is pd.Series
        assert type(self.df[["geometry"]]) is GeoDataFrame
        assert type(self.df[["value1"]]) is pd.DataFrame
        assert type(self.df[["geometry2"]]) is GeoDataFrame
        self._check_metadata(self.df[["geometry2"]], geometry_column_name="geometry2")

    def test_loc(self):
        assert type(self.df.loc[:, ["value1", "value2"]]) is pd.DataFrame
        assert type(self.df.loc[:, "geometry"]) is GeoSeries
        assert type(self.df.loc[:, "geometry2"]) is GeoSeries
        assert type(self.df["value1"]) is pd.Series
        assert type(self.df.loc[:, ["geometry"]]) is GeoDataFrame
        assert type(self.df.loc[:, ["value1"]]) is pd.DataFrame
        assert type(self.df.loc[:, ["geometry2"]]) is GeoDataFrame
        self._check_metadata(self.df.loc[:, ["geometry2"]], "geometry2")

    def test_squeeze(self):
        assert type(self.df[["geometry"]].squeeze()) is GeoSeries
        assert type(self.df[["geometry2"]].squeeze()) is GeoSeries

    def test_to_frame(self):
        res1 = self.df["geometry"].to_frame()
        assert type(res1) is GeoDataFrame
        self._check_metadata(res1)
        res2 = self.df["geometry2"].to_frame()
        assert type(res2) is GeoDataFrame
        self._check_metadata(res2, geometry_column_name="geometry2")
        assert type(self.df["value1"].to_frame()) is pd.DataFrame

    def test_iloc(self):
        assert type(self.df.iloc[:, 1:3]) is pd.DataFrame
        assert type(self.df.iloc[:, 0]) is GeoSeries
        assert type(self.df.iloc[:, 3]) is GeoSeries
        assert type(self.df.iloc[:, 1]) is pd.Series
        res1 = self.df.iloc[:, [0]]
        assert type(res1) is GeoDataFrame
        self._check_metadata(res1)
        assert type(self.df.iloc[:, [1]]) is pd.DataFrame
        res2 = self.df.iloc[:, [3]]
        assert type(res2) is GeoDataFrame
        self._check_metadata(res2, geometry_column_name="geometry2")

    def test_reindex(self):
        res1 = self.df.reindex(columns=["value1"])
        assert type(res1) is pd.DataFrame
        res2 = self.df.reindex(columns=["geometry"])
        assert type(res2) is GeoDataFrame
        self._check_metadata(res2)
        res3 = self.df.reindex(columns=["geometry2"])
        assert type(res3) is GeoDataFrame
        self._check_metadata(res3, geometry_column_name="geometry2")

    def test_drop(self):
        assert type(self.df.drop(columns=["geometry", "geometry2"])) is pd.DataFrame
        res1 = self.df.drop(columns=["value1", "value2", "geometry2"])
        assert type(res1) is GeoDataFrame
        self._check_metadata(res1)
        res2 = self.df.drop(columns=["value1", "value2", "geometry"])
        assert type(res2) is GeoDataFrame
        self._check_metadata(res2, geometry_column_name="geometry2")

    def test_apply(self):
        assert type(self.df["geometry"].apply(lambda x: x)) is GeoSeries
        assert type(self.df["geometry2"].apply(lambda x: x)) is GeoSeries
        assert type(self.df["value1"].apply(lambda x: x)) is pd.Series

        assert type(self.df.apply(lambda x: x)) is GeoDataFrame
        assert type(self.df[["value1", "value2"]].apply(lambda x: x)) is pd.DataFrame
