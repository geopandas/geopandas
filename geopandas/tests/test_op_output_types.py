import pandas as pd
import pyproj
import pytest

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat

crsgs_osgb = pyproj.CRS(27700)
crs_wgs = pyproj.CRS(27700)


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
    df["geometry2"] = df[geo_name].set_crs(crsgs_osgb, allow_override=True)
    return df


class TestDataFrameMethodReturnTypes:
    @staticmethod
    def _check_metadata_gdf(gdf, geo_name="geometry", crs=crs_wgs):

        assert gdf._geometry_column_name == geo_name
        assert gdf.geometry.name == geo_name
        assert gdf.crs == crs

    @staticmethod
    def _check_metadata_gs(gs, name="geometry", crs=crs_wgs):
        assert gs.name == name
        assert gs.crs == crs

    def _check_standard_df(self, results, geo_name, method=None):
        for i in results:
            print(i.columns if isinstance(i, pd.DataFrame) else i.name)

        assert type(results[0]) is pd.DataFrame
        assert type(results[1]) is GeoDataFrame
        self._check_metadata_gdf(results[1], geo_name=geo_name)
        assert type(results[2]) is GeoDataFrame
        self._check_metadata_gdf(results[2], geo_name=geo_name)
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

    def _check_standard_srs(self, results, geo_name):
        assert type(results[0]) is GeoSeries
        self._check_metadata_gs(results[0], geo_name)
        assert type(results[1]) is GeoSeries
        self._check_metadata_gs(results[1], name="geometry2")
        assert type(results[2]) is pd.Series
        assert results[2].name == "value1"

    def _assert_object(self, result, expected_type, geo_name="geometry", crs=crs_wgs):
        """Helper method to make tests easier to read. Checks result is of the expected
        type. If result is a GeoDataFrame or GeoSeries, checks geo_name
        and crs match. If geo_name is None, then we expect a GeoDataFrame
        where the geometry column is invalid/ isn't set
        """
        assert type(result) is expected_type

        if expected_type == GeoDataFrame:
            if geo_name is not None:
                self._check_metadata_gdf(result, geo_name=geo_name, crs=crs)
        elif expected_type == GeoSeries:
            self._check_metadata_gs(result, name=geo_name, crs=crs)

    def test_getitem(self, df):
        geo_name = df.geometry.name
        self._assert_object(df[["value1", "value2"]], pd.DataFrame)
        self._assert_object(df[[geo_name, "geometry2"]], GeoDataFrame, geo_name)
        self._assert_object(df[[geo_name]], GeoDataFrame, geo_name)
        self._assert_object(df[["geometry2", "value1"]], pd.DataFrame)
        self._assert_object(df[["geometry2"]], pd.DataFrame)
        self._assert_object(df[["value1"]], pd.DataFrame)
        # Series
        self._assert_object(df[geo_name], GeoSeries, geo_name)
        self._assert_object(df["geometry2"], GeoSeries, "geometry2")
        self._assert_object(df["value1"], pd.Series)

    def test_loc(self, df):
        geo_name = df.geometry.name

        self._assert_object(df.loc[:, ["value1", "value2"]], pd.DataFrame)
        self._assert_object(df.loc[:, [geo_name, "geometry2"]], GeoDataFrame, geo_name)
        self._assert_object(df.loc[:, [geo_name]], GeoDataFrame, geo_name)
        self._assert_object(df.loc[:, ["geometry2", "value1"]], GeoDataFrame, None)
        self._assert_object(df.loc[:, ["geometry2"]], GeoDataFrame, None)
        # Ideally this would mirror __getitem__ and below would be true
        # self._assert_object(df.loc[:, ["geometry2", "value1"]], pd.DataFrame)
        # self._assert_object(df.loc[:, ["geometry2"]], pd.DataFrame)
        self._assert_object(df.loc[:, ["value1"]], pd.DataFrame)
        # Series
        self._assert_object(df.loc[:, geo_name], GeoSeries, geo_name)
        self._assert_object(df.loc[:, "geometry2"], GeoSeries, "geometry2")
        self._assert_object(df.loc[:, "value1"], pd.Series)

    def test_iloc(self, df):
        geo_name = df.geometry.name
        self._assert_object(df.iloc[:, 0:2], pd.DataFrame)
        self._assert_object(df.iloc[:, 2:4], GeoDataFrame, geo_name)
        self._assert_object(df.iloc[:, [2]], GeoDataFrame, geo_name)
        self._assert_object(df.iloc[:, [3, 0]], GeoDataFrame, None)
        self._assert_object(df.iloc[:, [3]], GeoDataFrame, None)
        # Ideally this would mirror __getitem__ and below would be true
        # self._assert_object(df[["geometry2", "value1"]], pd.DataFrame)
        # self._assert_object(df[["geometry2"]], pd.DataFrame)
        self._assert_object(df.iloc[:, [0]], pd.DataFrame)
        # Series
        self._assert_object(df.iloc[:, 2], GeoSeries, geo_name)
        self._assert_object(df.iloc[:, 3], GeoSeries, "geometry2")
        self._assert_object(df.iloc[:, 0], pd.Series)

    def test_squeeze(self, df):
        geo_name = df.geometry.name
        self._assert_object(df[[geo_name]].squeeze(), GeoSeries, geo_name)

        # Not ideal behaviour, but this is consistent with __getitem__
        self._assert_object(df[["geometry2"]].squeeze(), pd.Series)

    def test_to_frame(self, df):
        geo_name = df.geometry.name
        # This shouldn't be a specical case,
        # need to update GeoSeries._constructor_expanddim
        if geo_name == "geometry":
            self._assert_object(df[geo_name].to_frame(), GeoDataFrame, geo_name)
        assert df[geo_name].to_frame()._geometry_column_name == "geometry"

        res2 = df["geometry2"].to_frame()
        # TODO this reflects current behaviour, but we should fix
        #  GeoSeries._constructor_expanddim so this doesn't happen
        assert type(res2) is GeoDataFrame
        assert res2._geometry_column_name == "geometry"  # -> should be geo_name
        assert res2.crs is None  # -> should be self.osgb
        # also res2.geometry should not crash because geometry isn't set
        self._assert_object(df["value1"].to_frame(), pd.DataFrame)

    def test_reindex(self, df):
        geo_name = df.geometry.name
        self._check_standard_df(
            [
                df.reindex(columns=["value1", "value2"]),
                df.reindex(columns=[geo_name, "geometry2"]),
                df.reindex(columns=[geo_name]),
                df.reindex(columns=["geometry2", "value1"]),
                df.reindex(columns=["geometry2"]),
                df.reindex(columns=["value1"]),
            ],
            geo_name=geo_name,
        )

    def test_drop(self, df):
        geo_name = df.geometry.name
        self._check_standard_df(
            [
                df.drop(columns=[geo_name, "geometry2"]),
                df.drop(columns=["value1", "value2"]),
                df.drop(columns=["value1", "value2", "geometry2"]),
                df.drop(columns=[geo_name, "value2"]),
                df.drop(columns=["value1", "value2", geo_name]),
                df.drop(columns=["geometry2", "value2", geo_name]),
            ],
            geo_name=geo_name,
        )

    def test_apply(self, df):
        geo_name = df.geometry.name
        self._check_standard_srs(
            [
                df[geo_name].apply(lambda x: x),
                df["geometry2"].apply(lambda x: x),
                df["value1"].apply(lambda x: x),
            ],
            geo_name=geo_name,
        )

        assert type(df[geo_name].apply(lambda x: str(x))) is pd.Series
        assert type(df["geometry2"].apply(lambda x: str(x))) is pd.Series

        self._check_standard_df(
            [
                df[["value1", "value2"]].apply(lambda x: x),
                df[[geo_name, "geometry2"]].apply(lambda x: x),
                df[[geo_name]].apply(lambda x: x),
                df[["geometry2", "value1"]].apply(lambda x: x),
                df[["geometry2"]].apply(lambda x: x),
                df[["value1"]].apply(lambda x: x),
            ],
            geo_name=geo_name,
            method="apply",
        )
        self._check_standard_df(
            [
                df[["value1", "value2"]].apply(lambda x: x, axis=1),
                df[[geo_name, "geometry2"]].apply(lambda x: x, axis=1),
                df[[geo_name]].apply(lambda x: x, axis=1),
                df[["geometry2", "value1"]].apply(lambda x: x, axis=1),
                df[["geometry2"]].apply(lambda x: x, axis=1),
                df[["value1"]].apply(lambda x: x, axis=1),
            ],
            geo_name=geo_name,
            method="apply",
        )

    @pytest.mark.skipif(
        not compat.PANDAS_GE_10, reason="Convert dtypes new in pandas 1.0"
    )
    def test_convert_dtypes(self, df):
        geo_name = df.geometry.name

        # convert_dtypes also relies on constructor_expanddim, so crs and geom col
        # are lost right now. #TODO fix this
        assert type(df[["value1", "value2"]].convert_dtypes()) is pd.DataFrame
        assert type(df[[geo_name, "geometry2"]].convert_dtypes()) is GeoDataFrame
        assert type(df[[geo_name]].convert_dtypes()) is GeoDataFrame
        assert type(df[["geometry2", "value1"]].convert_dtypes()) is pd.DataFrame
        assert type(df[["geometry2"]].convert_dtypes()) is pd.DataFrame
        assert type(df[["value1"]].convert_dtypes()) is pd.DataFrame
