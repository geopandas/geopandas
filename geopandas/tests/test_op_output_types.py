import pandas as pd
import pyproj
import pytest

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries

crsgs_osgb = pyproj.CRS(27700)
crs_wgs = pyproj.CRS(27700)
# TODO need to check that behaviour works with non default geom col name


class TestDataFrameMethodReturnTypes:
    def setup_method(self):
        N = 10

        self.df = GeoDataFrame(
            [
                {
                    "value1": x + y,
                    "value2": x * y,
                    "geometry_temp": Point(x, y),  # rename this col in tests
                }
                for x, y in zip(range(N), range(N))
            ],
            crs=crs_wgs,
            geometry="geometry_temp",
        )
        # want geometry2 to be a GeoSeries not Series, test behaviour of non geom col
        self.df["geometry2"] = self.df["geometry_temp"].set_crs(
            crsgs_osgb, allow_override=True
        )

    @staticmethod
    def _check_metadata_gdf(gdf, geo_col_name="geometry", crs=crs_wgs):

        assert gdf._geometry_column_name == geo_col_name
        assert gdf.geometry.name == geo_col_name
        assert gdf.crs == crs

    @staticmethod
    def _check_metadata_gs(gs, name="geometry", crs=crs_wgs):
        assert gs.name == name
        assert gs.crs == crs

    def _check_standard_df(self, results, geo_col_name, method=None):
        for i in results:
            print(i.columns if isinstance(i, pd.DataFrame) else i.name)

        assert type(results[0]) is pd.DataFrame
        assert type(results[1]) is GeoDataFrame
        self._check_metadata_gdf(results[1], geo_col_name=geo_col_name)
        assert type(results[2]) is GeoDataFrame
        self._check_metadata_gdf(results[2], geo_col_name=geo_col_name)
        # Non geometry column is not treated specially, we get a DataFrame
        # Not sure this is the most desirable behaviour, but it is at least consistent
        # If there is only one geometry column left, returning a gdf is not ambiguous,
        # but if there are 2 geom cols, and non are the set geom col, behaviour is
        # unclear. Therefore opting to return dataframes.
        # assert type(results[3]) is pd.DataFrame
        # # Exception is if there is only one column in the dataframe, then seems
        # # wrong to return a dataframe. Not implemented as inconsistent with old
        # # __getitem__ behaviour.
        # assert type(results[4]) is pd.DataFrame
        # losing geom col still returns gdf - TODO different from logic in text above

        # End goal is to mimic getitem completely, that's not happening right now
        exceptions_for_now = ["getitem", "apply"]

        if method in exceptions_for_now:
            assert type(results[3]) is pd.DataFrame
        else:
            assert type(results[3]) is GeoDataFrame

        if method in exceptions_for_now:
            assert type(results[4]) is pd.DataFrame
        else:
            assert type(results[4]) is GeoDataFrame
        # self._check_metadata_gs(results[4], crsgs_osgb)
        assert type(results[5]) is pd.DataFrame

    def _check_standard_srs(self, results, geo_col_name):
        assert type(results[0]) is GeoSeries
        self._check_metadata_gs(results[0], geo_col_name)
        assert type(results[1]) is GeoSeries
        self._check_metadata_gs(results[1], name="geometry2")
        assert type(results[2]) is pd.Series
        assert results[2].name == "value1"

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_getitem(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        self._check_standard_df(
            [
                df[["value1", "value2"]],
                df[[set_geom_col_name, "geometry2"]],
                df[[set_geom_col_name]],
                df[["geometry2", "value1"]],
                df[["geometry2"]],
                df[["value1"]],
            ],
            geo_col_name=set_geom_col_name,
            method="getitem",
        )
        self._check_standard_srs(
            [df[set_geom_col_name], df["geometry2"], df["value1"]],
            geo_col_name=set_geom_col_name,
        )

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_loc(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        self._check_standard_df(
            [
                df.loc[:, ["value1", "value2"]],
                df.loc[:, [set_geom_col_name, "geometry2"]],
                df.loc[:, [set_geom_col_name]],
                df.loc[:, ["geometry2", "value1"]],
                df.loc[:, ["geometry2"]],
                df.loc[:, ["value1"]],
            ],
            geo_col_name=set_geom_col_name,
        )
        self._check_standard_df(
            [
                df.loc[1:5, ["value1", "value2"]],
                df.loc[1:5, [set_geom_col_name, "geometry2"]],
                df.loc[1:5, [set_geom_col_name]],
                df.loc[1:5, ["geometry2", "value1"]],
                df.loc[1:5, ["geometry2"]],
                df.loc[1:5, ["value1"]],
            ],
            geo_col_name=set_geom_col_name,
        )
        self._check_standard_srs(
            [
                df.loc[:, set_geom_col_name],
                df.loc[:, "geometry2"],
                df.loc[:, "value1"],
            ],
            geo_col_name=set_geom_col_name,
        )

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_iloc(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        self._check_standard_df(
            [
                df.iloc[:, 0:2],
                df.iloc[:, 2:4],
                df.iloc[:, [2]],
                df.iloc[:, [3, 0]],
                df.iloc[:, [3]],
                df.iloc[:, [0]],
            ],
            geo_col_name=set_geom_col_name,
        )
        self._check_standard_srs(
            [df.iloc[:, 2], df.iloc[:, 3], df.iloc[:, 0]],
            geo_col_name=set_geom_col_name,
        )

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_squeeze(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        res1 = df[[set_geom_col_name]].squeeze()
        assert type(res1) is GeoSeries
        self._check_metadata_gs(res1, name=set_geom_col_name)

        res2 = self.df[["geometry2"]].squeeze()
        # Not ideal behaviour, but to change this need to agree to change __getitem__
        assert type(res2) is pd.Series

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_to_frame(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        res1 = df[set_geom_col_name].to_frame()
        assert type(res1) is GeoDataFrame
        if set_geom_col_name == "geometry":  # -> should be set_geom_col_name always
            self._check_metadata_gdf(res1, geo_col_name=set_geom_col_name)
        res2 = df["geometry2"].to_frame()
        assert type(res2) is GeoDataFrame
        # TODO this reflects current behaviour, but we should fix
        #  GeoSeries._constructor_expanddim so this doesn't happen
        assert (
            res2._geometry_column_name == "geometry"
        )  # -> should be set_geom_col_name
        assert res2.crs is None  # -> should be self.osgb
        # also res2.geometry should not crash because geometry isn't set
        assert type(self.df["value1"].to_frame()) is pd.DataFrame

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_reindex(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        self._check_standard_df(
            [
                df.reindex(columns=["value1", "value2"]),
                df.reindex(columns=[set_geom_col_name, "geometry2"]),
                df.reindex(columns=[set_geom_col_name]),
                df.reindex(columns=["geometry2", "value1"]),
                df.reindex(columns=["geometry2"]),
                df.reindex(columns=["value1"]),
            ],
            geo_col_name=set_geom_col_name,
        )

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_drop(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        self._check_standard_df(
            [
                df.drop(columns=[set_geom_col_name, "geometry2"]),
                df.drop(columns=["value1", "value2"]),
                df.drop(columns=["value1", "value2", "geometry2"]),
                df.drop(columns=[set_geom_col_name, "value2"]),
                df.drop(columns=["value1", "value2", set_geom_col_name]),
                df.drop(columns=["geometry2", "value2", set_geom_col_name]),
            ],
            geo_col_name=set_geom_col_name,
        )

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_apply(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)
        self._check_standard_srs(
            [
                df[set_geom_col_name].apply(lambda x: x),
                df["geometry2"].apply(lambda x: x),
                df["value1"].apply(lambda x: x),
            ],
            geo_col_name=set_geom_col_name,
        )

        assert type(df[set_geom_col_name].apply(lambda x: str(x))) is pd.Series
        assert type(df["geometry2"].apply(lambda x: str(x))) is pd.Series

        self._check_standard_df(
            [
                df[["value1", "value2"]].apply(lambda x: x),
                df[[set_geom_col_name, "geometry2"]].apply(lambda x: x),
                df[[set_geom_col_name]].apply(lambda x: x),
                df[["geometry2", "value1"]].apply(lambda x: x),
                df[["geometry2"]].apply(lambda x: x),
                df[["value1"]].apply(lambda x: x),
            ],
            geo_col_name=set_geom_col_name,
            method="apply",
        )
        self._check_standard_df(
            [
                df[["value1", "value2"]].apply(lambda x: x, axis=1),
                df[[set_geom_col_name, "geometry2"]].apply(lambda x: x, axis=1),
                df[[set_geom_col_name]].apply(lambda x: x, axis=1),
                df[["geometry2", "value1"]].apply(lambda x: x, axis=1),
                df[["geometry2"]].apply(lambda x: x, axis=1),
                df[["value1"]].apply(lambda x: x, axis=1),
            ],
            geo_col_name=set_geom_col_name,
            method="apply",
        )

    @pytest.mark.parametrize("set_geom_col_name", ["geometry", "points"])
    def test_convert_dtypes(self, set_geom_col_name):
        df = self.df.rename_geometry(set_geom_col_name)

        # convert_dtypes also relies on constructor_expanddim, so crs and geom col
        # are lost right now. #TODO fix this
        assert type(df[["value1", "value2"]].convert_dtypes()) is pd.DataFrame
        assert (
            type(df[[set_geom_col_name, "geometry2"]].convert_dtypes()) is GeoDataFrame
        )
        assert type(df[[set_geom_col_name]].convert_dtypes()) is GeoDataFrame
        assert type(df[["geometry2", "value1"]].convert_dtypes()) is pd.DataFrame
        assert type(df[["geometry2"]].convert_dtypes()) is pd.DataFrame
        assert type(df[["value1"]].convert_dtypes()) is pd.DataFrame
