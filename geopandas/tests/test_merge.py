from __future__ import absolute_import

import pandas as pd
from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries


class TestMerging:

    def setup_method(self):

        self.gseries = GeoSeries([Point(i, i) for i in range(3)])
        self.series = pd.Series([1, 2, 3])
        self.gdf = GeoDataFrame({'geometry': self.gseries, 'values': range(3)})
        self.df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [0.1, 0.2, 0.3]})

    def _check_metadata(self, gdf, geometry_column_name='geometry', crs=None):

        assert gdf._geometry_column_name == geometry_column_name
        assert gdf.crs == crs

    def test_merge(self):

        res = self.gdf.merge(self.df, left_on='values', right_on='col1')

        # check result is a GeoDataFrame
        assert isinstance(res, GeoDataFrame)

        # check geometry property gives GeoSeries
        assert isinstance(res.geometry, GeoSeries)

        # check metadata
        self._check_metadata(res)

        ## test that crs and other geometry name are preserved
        self.gdf.crs = {'init' :'epsg:4326'}
        self.gdf = (self.gdf.rename(columns={'geometry': 'points'})
                            .set_geometry('points'))
        res = self.gdf.merge(self.df, left_on='values', right_on='col1')
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res, 'points', self.gdf.crs)

    def test_concat_axis0(self):

        res = pd.concat([self.gdf, self.gdf])

        assert res.shape == (6, 2)
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)

    def test_concat_axis1(self):

        res = pd.concat([self.gdf, self.df], axis=1)

        assert res.shape == (3, 4)
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)
