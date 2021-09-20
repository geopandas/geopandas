import pandas as pd
import pyproj
import pytest

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries

crs_osgb = pyproj.CRS(27700)
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
    df["geometry2"] = df[geo_name].set_crs(crs_osgb, allow_override=True)
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
        where the geometry column is invalid/ isn't set. This is never desirable,
        but is a reality of this first stage of implementation.
        """
        assert type(result) is expected_type

        if expected_type == GeoDataFrame:
            if geo_name is not None:
                self._check_metadata_gdf(result, geo_name=geo_name, crs=crs)
            else:
                with pytest.raises(AttributeError, match="No geometry data set yet"):
                    result.geometry.name  # be explicit that geometry is invalid here
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
        # self._assert_object(df.iloc[:, [3, 0]], pd.DataFrame)
        # self._assert_object(df.iloc[:, [3]], pd.DataFrame)
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
        # TODO this reflects current behaviour, but we should fix
        #  GeoSeries._constructor_expanddim so this doesn't happen
        res1 = df[geo_name].to_frame()
        if geo_name == "geometry":  # -> this should be doable for any geo_name
            self._assert_object(res1, GeoDataFrame, geo_name)
        assert res1._geometry_column_name == "geometry"  # -> should be geo_name

        res2 = df["geometry2"].to_frame()

        assert type(res2) is GeoDataFrame
        assert res2._geometry_column_name == "geometry"  # -> should be geometry2
        assert res2.crs is None  # -> should be crs_osgb
        # also res2.geometry should not crash because geometry isn't set
        self._assert_object(df["value1"].to_frame(), pd.DataFrame)

    def test_reindex(self, df):
        geo_name = df.geometry.name

        test_func = self._assert_object  # easier to read without black line wraps
        test_func(df.reindex(columns=["value1", "value2"]), pd.DataFrame)
        test_func(df.reindex(columns=[geo_name, "geometry2"]), GeoDataFrame, geo_name)
        test_func(df.reindex(columns=[geo_name]), GeoDataFrame, geo_name)
        test_func(df.reindex(columns=["geometry2", "value1"]), GeoDataFrame, None)
        test_func(df.reindex(columns=["geometry2"]), GeoDataFrame, None)
        # Ideally this would mirror __getitem__ and below would be true
        # test_func(df.reindex(columns=["geometry2", "value1"]), pd.DataFrame)
        # test_func(df.reindex(columns=["geometry2"]), pd.DataFrame)
        test_func(df.reindex(columns=["value1"]), pd.DataFrame)

    def test_drop(self, df):
        geo_name = df.geometry.name
        test_func = self._assert_object  # easier to read without black line wraps
        test_func(df.drop(columns=[geo_name, "geometry2"]), pd.DataFrame)
        test_func(df.drop(columns=["value1", "value2"]), GeoDataFrame, geo_name)
        cols = ["value1", "value2", "geometry2"]
        test_func(df.drop(columns=cols), GeoDataFrame, geo_name)
        test_func(df.drop(columns=[geo_name, "value2"]), GeoDataFrame, None)
        test_func(df.drop(columns=["value1", "value2", geo_name]), GeoDataFrame, None)
        # Ideally this would mirror __getitem__ and below would be true
        # test_func(df.drop(columns=[geo_name, "value2"]), pd.DataFrame)
        # test_func(df.drop(columns=["value1", "value2", geo_name]), pd.DataFrame)
        test_func(df.drop(columns=["geometry2", "value2", geo_name]), pd.DataFrame)

    def test_apply(self, df):
        geo_name = df.geometry.name
        test_func = self._assert_object  # easier to read without black line wraps

        def identity(x):
            return x

        # axis = 0
        test_func(df[["value1", "value2"]].apply(identity), pd.DataFrame)
        test_func(df[[geo_name, "geometry2"]].apply(identity), GeoDataFrame, geo_name)
        test_func(df[[geo_name]].apply(identity), GeoDataFrame, geo_name)
        test_func(df[["geometry2", "value1"]].apply(identity), pd.DataFrame)
        test_func(df[["geometry2"]].apply(identity), pd.DataFrame)
        test_func(df[["value1"]].apply(identity), pd.DataFrame)

        # axis = 0, Series
        test_func(df[geo_name].apply(identity), GeoSeries, geo_name)
        test_func(df["geometry2"].apply(identity), GeoSeries, "geometry2")
        test_func(df["value1"].apply(identity), pd.Series)

        # axis =0, Series, no longer geometry
        test_func(df[geo_name].apply(lambda x: str(x)), pd.Series)
        test_func(df["geometry2"].apply(lambda x: str(x)), pd.Series)

        # axis = 1
        test_func(df[["value1", "value2"]].apply(identity, axis=1), pd.DataFrame)
        test_func(
            df[[geo_name, "geometry2"]].apply(identity, axis=1), GeoDataFrame, geo_name
        )
        test_func(df[[geo_name]].apply(identity, axis=1), GeoDataFrame, geo_name)
        test_func(df[["geometry2", "value1"]].apply(identity, axis=1), pd.DataFrame)
        test_func(df[["geometry2"]].apply(identity, axis=1), pd.DataFrame)
        test_func(df[["value1"]].apply(identity, axis=1), pd.DataFrame)
