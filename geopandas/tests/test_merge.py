import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries


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
        # check metadata comes from first gdf
        res4 = pd.concat([self.gdf.set_crs("epsg:4326"), self.gdf], axis=0)
        # Note: this behaviour potentially does not make sense. If geom cols are
        # concatenated but have different CRS, then the CRS will be overridden.
        self._check_metadata(res4, crs="epsg:4326")

        # series
        res = pd.concat([self.gdf.geometry, self.gdf.geometry])
        assert res.shape == (6,)
        assert isinstance(res, GeoSeries)
        assert isinstance(res.geometry, GeoSeries)

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
