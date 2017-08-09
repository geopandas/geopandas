from __future__ import absolute_import

import fiona

import geopandas
from geopandas import read_postgis, read_file
from geopandas.tests.util import connect, create_db, unittest, validate_boro_df


class TestIO(unittest.TestCase):
    def setUp(self):
        nybb_zip_path = geopandas.datasets.get_path('nybb')
        self.df = read_file(nybb_zip_path)
        with fiona.open(nybb_zip_path) as f:
            self.crs = f.crs
            self.columns = list(f.meta["schema"]["properties"].keys())

    def test_read_postgis_default(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(self, df)

    def test_read_postgis_custom_geom_col(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = """SELECT
                     borocode, boroname, shape_leng, shape_area,
                     geom AS __geometry__
                     FROM nybb;"""
            df = read_postgis(sql, con, geom_col='__geometry__')
        finally:
            con.close()

        validate_boro_df(self, df)

    def test_read_file(self):
        df = self.df.rename(columns=lambda x: x.lower())
        validate_boro_df(self, df)
        self.assert_(df.crs == self.crs)
        # get lower case columns, and exclude geometry column from comparison
        lower_columns = [c.lower() for c in self.columns]
        self.assert_((df.columns[:-1] == lower_columns).all())

    def test_filtered_read_file(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = (1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert(full_df_shape != filtered_df_shape)
        assert(filtered_df_shape == (2, 5))
