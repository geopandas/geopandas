import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries
from geopandas import _compat as compat


class TestMerging:
    def setup_method(self):

        self.gseries = GeoSeries([Point(i, i) for i in range(3)])
        self.series = pd.Series([1, 2, 3])
        self.gdf = GeoDataFrame({"geometry": self.gseries, "values": range(3)})
        self.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [0.1, 0.2, 0.3]})

    def _check_metadata(self, gdf, geometry_column_name="geometry", crs=None):

        assert gdf._geometry_column_name == geometry_column_name
        assert gdf.crs == crs

    def test_merge(self):

        res = self.gdf.merge(self.df, left_on="values", right_on="col1")

        # check result is a GeoDataFrame
        assert isinstance(res, GeoDataFrame)

        # check geometry property gives GeoSeries
        assert isinstance(res.geometry, GeoSeries)

        # check metadata
        self._check_metadata(res)

        # test that crs and other geometry name are preserved
        self.gdf.crs = "epsg:4326"
        self.gdf = self.gdf.rename(columns={"geometry": "points"}).set_geometry(
            "points"
        )
        res = self.gdf.merge(self.df, left_on="values", right_on="col1")
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res, "points", self.gdf.crs)

    def test_concat_axis0(self):
        # frame
        res = pd.concat([self.gdf, self.gdf])
        assert res.shape == (6, 2)
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)
        exp = GeoDataFrame(pd.concat([pd.DataFrame(self.gdf), pd.DataFrame(self.gdf)]))
        assert_geodataframe_equal(exp, res)

        # series
        res = pd.concat([self.gdf.geometry, self.gdf.geometry])
        assert res.shape == (6,)
        assert isinstance(res, GeoSeries)
        assert isinstance(res.geometry, GeoSeries)

    def test_concat_axis0_crs(self):

        # CRS not set for both GeoDataFrame
        res = pd.concat([self.gdf, self.gdf])
        self._check_metadata(res)

        # CRS set for both GeoDataFrame, same CRS
        res1 = pd.concat([self.gdf.set_crs("epsg:4326"), self.gdf.set_crs("epsg:4326")])
        self._check_metadata(res1, crs="epsg:4326")

        # CRS not set for one GeoDataFrame, but set for the other GeoDataFrame
        with pytest.warns(
            UserWarning, match=r"CRS not set for some of the concatenation inputs.*"
        ):
            res2 = pd.concat([self.gdf, self.gdf.set_crs("epsg:4326")])
            self._check_metadata(res2, crs="epsg:4326")

        # CRS set for both GeoDataFrame, different CRS
        with pytest.raises(
            ValueError, match=r"Cannot determine common CRS for concatenation inputs.*"
        ):
            pd.concat([self.gdf.set_crs("epsg:4326"), self.gdf.set_crs("epsg:4327")])

        # CRS not set for one GeoDataFrame, but set for the other GeoDataFrames,
        # same CRS
        with pytest.warns(
            UserWarning, match=r"CRS not set for some of the concatenation inputs.*"
        ):
            res3 = pd.concat(
                [self.gdf, self.gdf.set_crs("epsg:4326"), self.gdf.set_crs("epsg:4326")]
            )
            self._check_metadata(res3, crs="epsg:4326")

        # CRS not set for one GeoDataFrame, but set for the other GeoDataFrames,
        # different CRS
        with pytest.raises(
            ValueError, match=r"Cannot determine common CRS for concatenation inputs.*"
        ):
            pd.concat(
                [self.gdf, self.gdf.set_crs("epsg:4326"), self.gdf.set_crs("epsg:4327")]
            )

    def test_concat_axis1(self):

        res = pd.concat([self.gdf, self.df], axis=1)

        assert res.shape == (3, 4)
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)

    def test_concat_axis1_multiple_geodataframes(self):
        # https://github.com/geopandas/geopandas/issues/1230
        # Expect that concat should fail gracefully if duplicate column names belonging
        # to geometry columns are introduced.
        expected_err = (
            "GeoDataFrame does not support multiple columns using the geometry"
            " column name 'geometry'"
        )
        with pytest.raises(ValueError, match=expected_err):
            pd.concat([self.gdf, self.gdf], axis=1)

        # Check case is handled if custom geometry column name is used
        df2 = self.gdf.rename_geometry("geom")
        expected_err2 = (
            "Concat operation has resulted in multiple columns using the geometry "
            "column name 'geom'."
        )
        with pytest.raises(ValueError, match=expected_err2):
            pd.concat([df2, df2], axis=1)

        # Check that two geometry columns is fine, if they have different names
        res3 = pd.concat([df2.set_crs("epsg:4326"), self.gdf], axis=1)
        # check metadata comes from first df
        self._check_metadata(res3, geometry_column_name="geom", crs="epsg:4326")

    @pytest.mark.xfail(
        not compat.PANDAS_GE_11,
        reason="pandas <=1.0 hard codes concat([GeoSeries, GeoSeries]) -> "
        "DataFrame or Union[DataFrame, SparseDataFrame] in 0.25",
    )
    @pytest.mark.filterwarnings("ignore:Accessing CRS")
    def test_concat_axis1_geoseries(self):
        gseries2 = GeoSeries([Point(i, i) for i in range(3, 6)], crs="epsg:4326")
        result = pd.concat([gseries2, self.gseries], axis=1)
        # Note this is not consistent with concat([gdf, gdf], axis=1) where the
        # left metadata is set on the result. This is deliberate for now.
        assert type(result) is GeoDataFrame
        assert result._geometry_column_name is None
        assert_index_equal(pd.Index([0, 1]), result.columns)

        gseries2.name = "foo"
        result2 = pd.concat([gseries2, self.gseries], axis=1)
        assert type(result2) is GeoDataFrame
        assert result._geometry_column_name is None
        assert_index_equal(pd.Index(["foo", 0]), result2.columns)
