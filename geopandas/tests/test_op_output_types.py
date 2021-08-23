import pandas as pd
import pyproj

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
                    "geometry": Point(x, y),
                }
                for x, y in zip(range(N), range(N))
            ],
            crs=crs_wgs,
        )
        # want geometry2 to be a GeoSeries not Series, test behaviour of non geom col
        self.df["geometry2"] = self.df["geometry"].set_crs(
            crsgs_osgb, allow_override=True
        )

    @staticmethod
    def _check_metadata_gdf(gdf, geometry_column_name="geometry", crs=crs_wgs):

        assert gdf._geometry_column_name == geometry_column_name
        assert gdf.geometry.name == geometry_column_name
        assert gdf.crs == crs

    @staticmethod
    def _check_metadata_gs(gs, name="geometry", crs=crs_wgs):
        assert gs.name == name
        assert gs.crs == crs

    def _check_standard_df(self, results, method=None):
        for i in results:
            print(i.columns if isinstance(i, pd.DataFrame) else i.name)

        assert type(results[0]) is pd.DataFrame
        assert type(results[1]) is GeoDataFrame
        self._check_metadata_gdf(results[1])
        assert type(results[2]) is GeoDataFrame
        self._check_metadata_gdf(results[2])
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
            assert type(results[3]) is pd.DataFrame
        else:
            assert type(results[3]) is GeoDataFrame
        # self._check_metadata_gs(results[4], crsgs_osgb)
        assert type(results[5]) is pd.DataFrame

    def _check_standard_srs(self, results):
        assert type(results[0]) is GeoSeries
        self._check_metadata_gs(results[0])
        assert type(results[1]) is GeoSeries
        self._check_metadata_gs(results[1], name="geometry2")
        assert type(results[2]) is pd.Series
        assert results[2].name == "value1"

    def test_getitem(self):
        self._check_standard_df(
            [
                self.df[["value1", "value2"]],
                self.df[["geometry", "geometry2"]],
                self.df[["geometry"]],
                self.df[["geometry2", "value1"]],
                self.df[["geometry2"]],
                self.df[["value1"]],
            ],
            method="getitem",
        )
        self._check_standard_srs(
            [self.df["geometry"], self.df["geometry2"], self.df["value1"]]
        )

    def test_loc(self):
        self._check_standard_df(
            [
                self.df.loc[:, ["value1", "value2"]],
                self.df.loc[:, ["geometry", "geometry2"]],
                self.df.loc[:, ["geometry"]],
                self.df.loc[:, ["geometry2", "value1"]],
                self.df.loc[:, ["geometry2"]],
                self.df.loc[:, ["value1"]],
            ]
        )
        self._check_standard_df(
            [
                self.df.loc[1:5, ["value1", "value2"]],
                self.df.loc[1:5, ["geometry", "geometry2"]],
                self.df.loc[1:5, ["geometry"]],
                self.df.loc[1:5, ["geometry2", "value1"]],
                self.df.loc[1:5, ["geometry2"]],
                self.df.loc[1:5, ["value1"]],
            ]
        )
        self._check_standard_srs(
            [
                self.df.loc[:, "geometry"],
                self.df.loc[:, "geometry2"],
                self.df.loc[:, "value1"],
            ]
        )

    def test_iloc(self):
        self._check_standard_df(
            [
                self.df.iloc[:, 0:2],
                self.df.iloc[:, 2:4],
                self.df.iloc[:, [2]],
                self.df.iloc[:, [3, 0]],
                self.df.iloc[:, [3]],
                self.df.iloc[:, [0]],
            ]
        )
        self._check_standard_srs(
            [self.df.iloc[:, 2], self.df.iloc[:, 3], self.df.iloc[:, 0]]
        )

    def test_squeeze(self):
        res1 = self.df[["geometry"]].squeeze()
        assert type(res1) is GeoSeries
        self._check_metadata_gs(res1)
        res2 = self.df.rename_geometry("points")[["points"]].squeeze()
        assert type(res2) is GeoSeries
        self._check_metadata_gs(res2, name="points")

        res2 = self.df[["geometry2"]].squeeze()
        # Not ideal behaviour, but to change this need to agree to change __getitem__
        assert type(res2) is pd.Series

    def test_to_frame(self):
        res1 = self.df["geometry"].to_frame()
        assert type(res1) is GeoDataFrame
        self._check_metadata_gdf(res1, geometry_column_name="geometry")
        res2 = self.df["geometry2"].to_frame()
        assert type(res2) is GeoDataFrame
        # TODO this reflects current behaviour, but we should fix
        #  GeoSeries._constructor_expanddim so this doesn't happen
        assert res2._geometry_column_name == "geometry"  # -> should be geometry2
        assert res2.crs is None  # -> should be self.osgb
        # also res2.geometry should not crash because geometry isn't set
        assert type(self.df["value1"].to_frame()) is pd.DataFrame

    def test_reindex(self):
        self._check_standard_df(
            [
                self.df.reindex(columns=["value1", "value2"]),
                self.df.reindex(columns=["geometry", "geometry2"]),
                self.df.reindex(columns=["geometry"]),
                self.df.reindex(columns=["geometry2", "value1"]),
                self.df.reindex(columns=["geometry2"]),
                self.df.reindex(columns=["value1"]),
            ]
        )

    def test_drop(self):
        self._check_standard_df(
            [
                self.df.drop(columns=["geometry", "geometry2"]),
                self.df.drop(columns=["value1", "value2"]),
                self.df.drop(columns=["value1", "value2", "geometry2"]),
                self.df.drop(columns=["geometry", "value2"]),
                self.df.drop(columns=["value1", "value2", "geometry"]),
                self.df.drop(columns=["geometry2", "value2", "geometry"]),
            ]
        )

    def test_apply(self):
        self._check_standard_srs(
            [
                self.df["geometry"].apply(lambda x: x),
                self.df["geometry2"].apply(lambda x: x),
                self.df["value1"].apply(lambda x: x),
            ],
        )

        assert type(self.df["geometry"].apply(lambda x: str(x))) is pd.Series
        assert type(self.df["geometry2"].apply(lambda x: str(x))) is pd.Series

        self._check_standard_df(
            [
                self.df[["value1", "value2"]].apply(lambda x: x),
                self.df[["geometry", "geometry2"]].apply(lambda x: x),
                self.df[["geometry"]].apply(lambda x: x),
                self.df[["geometry2", "value1"]].apply(lambda x: x),
                self.df[["geometry2"]].apply(lambda x: x),
                self.df[["value1"]].apply(lambda x: x),
            ],
            method="apply",
        )
        self._check_standard_df(
            [
                self.df[["value1", "value2"]].apply(lambda x: x, axis=1),
                self.df[["geometry", "geometry2"]].apply(lambda x: x, axis=1),
                self.df[["geometry"]].apply(lambda x: x, axis=1),
                self.df[["geometry2", "value1"]].apply(lambda x: x, axis=1),
                self.df[["geometry2"]].apply(lambda x: x, axis=1),
                self.df[["value1"]].apply(lambda x: x, axis=1),
            ],
            method="apply",
        )

    def test_convert_dtypes(self):
        # convert_dtypes also relies on constructor_expanddim, so crs and geom col
        # are lost right now. #TODO fix this
        assert type(self.df[["value1", "value2"]].convert_dtypes()) is pd.DataFrame
        assert type(self.df[["geometry", "geometry2"]].convert_dtypes()) is GeoDataFrame
        assert type(self.df[["geometry"]].convert_dtypes()) is GeoDataFrame
        assert type(self.df[["geometry2", "value1"]].convert_dtypes()) is pd.DataFrame
        assert type(self.df[["geometry2"]].convert_dtypes()) is pd.DataFrame
        assert type(self.df[["value1"]].convert_dtypes()) is pd.DataFrame

        # self._check_standard_df(
        #     [
        #         self.df[["value1", "value2"]].convert_dtypes(),
        #         self.df[["geometry", "geometry2"]].convert_dtypes(),
        #         self.df[["geometry"]].convert_dtypes(),
        #         self.df[["geometry2", 'value1']].convert_dtypes(),
        #         self.df[["geometry2"]].convert_dtypes(),
        #         self.df[["value1"]].convert_dtypes()
        #     ]
        # )
