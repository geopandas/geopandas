from __future__ import absolute_import

import fiona

from geopandas import GeoDataFrame, read_postgis, read_file
from geopandas.tests.util import download_nybb, connect, create_db, \
     PANDAS_NEW_SQL_API, unittest, validate_boro_df


class TestIO(unittest.TestCase):
    def setUp(self):
        nybb_filename = download_nybb()
        path = '/nybb_14a_av/nybb.shp'
        vfs = 'zip://' + nybb_filename
        self.df = read_file(path, vfs=vfs)
        with fiona.open(path, vfs=vfs) as f:
            self.crs = f.crs

    def test_read_postgis_default(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            if PANDAS_NEW_SQL_API:
                # It's not really a connection, it's an engine
                con = con.connect()
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
            if PANDAS_NEW_SQL_API:
                # It's not really a connection, it's an engine
                con = con.connect()
            con.close()

        validate_boro_df(self, df)

    def test_read_file(self):
        df = self.df.rename(columns=lambda x: x.lower())
        validate_boro_df(self, df)
        self.assert_(df.crs == self.crs)
