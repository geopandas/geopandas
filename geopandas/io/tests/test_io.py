from __future__ import absolute_import

import fiona

import geopandas
from geopandas import read_postgis, read_file

import pytest
from geopandas.tests.util import connect, create_db, validate_boro_df


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
        geom_col = "the_geom"
        if con is None or not create_db(self.df, geom_col=geom_col):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con, geom_col=geom_col)
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_postgis_select_geom_as(self):
        con = connect('test_geopandas')
        orig_geom = "geom"
        out_geom = "the_geom"
        if con is None or not create_db(self.df, geom_col=orig_geom):
            raise pytest.skip()

        try:
            sql = """SELECT borocode, boroname, shape_leng, shape_area,
                     {} as {} FROM nybb;""".format(orig_geom, out_geom)
            df = read_postgis(sql, con, geom_col=out_geom)
        finally:
            con.close()

        validate_boro_df(df)

    # geopandas does not currently return use the SRID to set the frame CRS
    @pytest.mark.xfail
    def test_read_postgis_get_srid(self):
        crs = {"init": "epsg:4269"}
        df_reproj = self.df.to_crs(crs)
        created = create_db(df_reproj, srid=4269)
        con = connect('test_geopandas')
        if con is None or not created:
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(df)
        assert(df.crs == crs)

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
