from __future__ import absolute_import

from collections import OrderedDict

import fiona

import geopandas
from geopandas import read_postgis, read_file

import pytest
from geopandas.tests.util import connect, create_db, validate_boro_df

from shapely.geometry.geo import box


class TestIO:
    def setup_method(self):
        nybb_zip_path = geopandas.datasets.get_path('nybb')
        self.df = read_file(nybb_zip_path)
        with fiona.open(nybb_zip_path) as f:
            self.crs = f.crs
            self.columns = list(f.meta["schema"]["properties"].keys())

    def test_read_postgis_default(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_postgis_custom_geom_col(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise pytest.skip()

        try:
            sql = """SELECT
                     borocode, boroname, shape_leng, shape_area,
                     geom AS __geometry__
                     FROM nybb;"""
            df = read_postgis(sql, con, geom_col='__geometry__')
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_file(self):
        df = self.df.rename(columns=lambda x: x.lower())
        validate_boro_df(df)
        assert df.crs == self.crs
        # get lower case columns, and exclude geometry column from comparison
        lower_columns = [c.lower() for c in self.columns]
        assert (df.columns[:-1] == lower_columns).all()

    @pytest.mark.web
    def test_remote_geojson_url(self):
        url = ("https://raw.githubusercontent.com/geopandas/geopandas/"
               "master/examples/null_geom.geojson")
        gdf = read_file(url)
        assert isinstance(gdf, geopandas.GeoDataFrame)

    def test_filtered_read_file(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = (1031051.7879884212, 224272.49231459625, 1047224.3104931959,
                244317.30894023244)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert full_df_shape != filtered_df_shape
        assert filtered_df_shape == (2, 5)

    def test_empty_shapefile(self, tmpdir):

        # create empty shapefile
        meta = {'crs': {},
                'crs_wkt': '',
                'driver': 'ESRI Shapefile',
                'schema':
                    {'geometry': 'Point',
                     'properties': OrderedDict([('A', 'int:9'),
                                                ('Z', 'float:24.15')])}}

        fname = str(tmpdir.join("test_empty.shp"))

        with fiona.drivers():
            with fiona.open(fname, 'w', **meta) as _:
                pass

        empty = read_file(fname)
        assert isinstance(empty, geopandas.GeoDataFrame)
        assert all(empty.columns == ['A', 'Z', 'geometry'])

    def test_filtered_read_file_with_gdf_boundary(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = geopandas.GeoDataFrame(data={'a': ['a']}, geometry=[(1031051.7879884212, 224272.49231459625, 1047224.3104931959,
                                                                    244317.30894023244)], crs=self.crs)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert full_df_shape != filtered_df_shape
        assert filtered_df_shape == (2, 5)

    def test_filtered_read_file_with_gdf_boundary_mismatched_crs(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = geopandas.GeoDataFrame(
            geometry=[box(1031051.7879884212, 224272.49231459625, 1047224.3104931959,
                          244317.30894023244)],
            crs=self.crs)
        bbox.to_crs(epsg=4436, inplace=True)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert full_df_shape != filtered_df_shape
        assert filtered_df_shape == (2, 5)
