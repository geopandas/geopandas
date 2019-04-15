from shapely.geometry import Point

from geopandas import GeoDataFrame


class TestToCRS:

    def setup_method(self):
        N = 10
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry': Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)

    def test_transform(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert all(df2['geometry'].geom_almost_equals(utm['geometry'],
                                                      decimal=2))

    def test_transform_inplace(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        df2.to_crs(epsg=4326, inplace=True)
        assert all(df2['geometry'].geom_almost_equals(lonlat['geometry'],
                                                      decimal=2))

    def test_to_crs_geo_column_name(self):
        # Test to_crs() with different geometry column name (GH#339)
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        df2 = df2.rename(columns={'geometry': 'geom'})
        df2.set_geometry('geom', inplace=True)
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert lonlat.geometry.name == 'geom'
        assert utm.geometry.name == 'geom'
        assert all(df2.geometry.geom_almost_equals(utm.geometry, decimal=2))
