import numpy as np

from geopandas import GeoDataFrame, points_from_xy
from geopandas.testing import assert_geodataframe_equal


def _create_df(x, y=None, crs=None):
    y = y or x
    x = np.asarray(x)
    y = np.asarray(y)

    return GeoDataFrame(
        {'geometry': points_from_xy(x, y), 'value1': x + y, 'value2': x * y},
        crs=crs)


def df_epsg26918():
    # EPSG:26918
    # Center coordinates
    # -1683723.64 6689139.23
    return _create_df(x=range(-1683723, -1683723 + 10, 1),
                      y=range(6689139, 6689139 + 10, 1),
                      crs={'init': 'epsg:26918', 'no_defs': True})


class TestToCRS:

    def test_transform(self):
        df = df_epsg26918()
        lonlat = df.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert_geodataframe_equal(df, utm, check_less_precise=True)

    def test_transform_inplace(self):
        df = df_epsg26918()
        lonlat = df.to_crs(epsg=4326)
        df.to_crs(epsg=4326, inplace=True)
        assert_geodataframe_equal(df, lonlat, check_less_precise=True)

    def test_to_crs_geo_column_name(self):
        # Test to_crs() with different geometry column name (GH#339)
        df = df_epsg26918()
        df = df.rename(columns={'geometry': 'geom'})
        df.set_geometry('geom', inplace=True)
        lonlat = df.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert lonlat.geometry.name == 'geom'
        assert utm.geometry.name == 'geom'
        assert_geodataframe_equal(df, utm, check_less_precise=True)
