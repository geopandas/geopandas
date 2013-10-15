import unittest

from geopandas import GeoDataFrame, read_postgis, read_file
import tests.util

class TestIO(unittest.TestCase):
    def setUp(self):
        nybb_filename = tests.util.download_nybb()
        self.df = read_file('/nybb_13a/nybb.shp', vfs='zip://' + nybb_filename)

    def test_read_postgis_default(self):
        con = tests.util.connect('test_geopandas')
        if con is None or not tests.util.create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        tests.util.validate_boro_df(self, df)

    def test_read_postgis_custom_geom_col(self):
        con = tests.util.connect('test_geopandas')
        if con is None or not tests.util.create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = """SELECT
                     borocode, boroname, shape_leng, shape_area,
                     geom AS __geometry__
                     FROM nybb;"""
            df = read_postgis(sql, con, geom_col='__geometry__')
        finally:
            con.close()

        tests.util.validate_boro_df(self, df)
