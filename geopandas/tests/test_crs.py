import numpy as np

from geopandas import GeoDataFrame, points_from_xy


def _create_df(x, y=None, crs=None):
    y = y or x
    x = np.asarray(x)
    y = np.asarray(y)

    return GeoDataFrame(
        {'geometry': points_from_xy(x, y), 'value1': x + y, 'value2': x * y},
        crs=crs)


class TestToCRS:

    def test_transform(self):
        df = _create_df(range(10), crs={'init': 'epsg:26918', 'no_defs': True})
        lonlat = df.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert all(df['geometry'].geom_almost_equals(utm['geometry'],
                                                     decimal=))

    def test_transform_inplace(self):
        df = _create_df(range(10), crs={'init': 'epsg:26918', 'no_defs': True})
        lonlat = df.to_crs(epsg=4326)
        df.to_crs(epsg=4326, inplace=True)
        assert all(df['geometry'].geom_almost_equals(lonlat['geometry'],
                                                     decimal=2))

    def test_to_crs_geo_column_name(self):
        # Test to_crs() with different geometry column name (GH#339)
        df = _create_df(range(10), crs={'init': 'epsg:26918', 'no_defs': True})
        df = df.rename(columns={'geometry': 'geom'})
        df.set_geometry('geom', inplace=True)
        lonlat = df.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert lonlat.geometry.name == 'geom'
        assert utm.geometry.name == 'geom'
        assert all(df.geometry.geom_almost_equals(utm.geometry, decimal=2))
