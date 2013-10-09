import unittest
import json
import os
import tempfile
import shutil
import urllib2

try:
    import psycopg2
    from psycopg2 import OperationalError
except ImportError:
    class OperationalError(Exception):
        pass

import numpy as np
from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame


class TestDataFrame(unittest.TestCase):

    def setUp(self):
        N = 10
        # Data from http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip
        # saved as geopandas/examples/nybb_13a.zip.
        if not os.path.exists(os.path.join('examples', 'nybb_13a.zip')):
            with open(os.path.join('examples', 'nybb_13a.zip'), 'w') as f:
                response = urllib2.urlopen('http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip')
                f.write(response.read())
        self.df = GeoDataFrame.from_file(
            '/nybb_13a/nybb.shp', vfs='zip://examples/nybb_13a.zip')
        self.tempdir = tempfile.mkdtemp()
        self.boros = np.array(['Staten Island', 'Queens', 'Brooklyn',
                               'Manhattan', 'Bronx'])
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)

        # Try to create the database, skip the db tests if something goes
        # wrong
        try:
            self._create_db()
            self.run_db_test = True
        except (NameError, OperationalError):
            # NameError is thrown if psycopg2 fails to import at top of file
            # OperationalError is thrown if we can't connect to the database
            self.run_db_test = False

    def _create_db(self):
        con = psycopg2.connect(dbname='test_geopandas')
        cursor = con.cursor()
        cursor.execute("DROP TABLE IF EXISTS nybb;")

        sql = """CREATE TABLE nybb (
            geom        geometry,
            borocode    integer,
            boroname    varchar(40),
            shape_leng  float,
            shape_area  float
        );"""
        cursor.execute(sql)

        for i, row in self.df.iterrows():
            sql = """INSERT INTO nybb VALUES (
                ST_GeometryFromText(%s), %s, %s, %s, %s 
            );"""
            cursor.execute(sql, (row['geometry'].wkt, 
                                 row['BoroCode'],
                                 row['BoroName'],
                                 row['Shape_Leng'],
                                 row['Shape_Area']))

        cursor.close()
        con.commit()
        con.close()


    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        self.assertTrue(type(self.df2) is GeoDataFrame)
        self.assertTrue(self.df2.crs == self.crs)

    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)

    def test_copy(self):
        df2 = self.df.copy()
        self.assertTrue(type(df2) is GeoDataFrame)
        self.assertEqual(self.df.crs, df2.crs)

    def test_to_file(self):
        """ Test to_file and from_file """
        tempfilename = os.path.join(self.tempdir, 'boros.shp')
        self.df.to_file(tempfilename)
        # Read layer back in?
        df = GeoDataFrame.from_file(tempfilename)
        self.assertTrue('geometry' in df)
        self.assertTrue(len(df) == 5)
        self.assertTrue(np.alltrue(df['BoroName'].values == self.boros))

    def test_mixed_types_to_file(self):
        """ Test that mixed geometry types raise error when writing to file """
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        s = GeoDataFrame({'geometry' : [Point(0, 0),
                                        Polygon([(0, 0), (1, 0), (1, 1)])]})
        with self.assertRaises(ValueError):
            s.to_file(tempfilename)

    def test_bool_index(self):
        # Find boros with 'B' in their name
        df = self.df[self.df['BoroName'].str.contains('B')]
        self.assertTrue(len(df) == 2)
        boros = df['BoroName'].values
        self.assertTrue('Brooklyn' in boros)
        self.assertTrue('Bronx' in boros)
        self.assertTrue(type(df) is GeoDataFrame)

    def test_transform(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        self.assertTrue(all(df2['geometry'].almost_equals(utm['geometry'], decimal=2)))

    def _validate_sql(self, df):
        # Make sure all the columns are there and the geometries
        # were properly loaded as MultiPolygons
        self.assertEqual(len(df), 5)
        columns = ('borocode', 'boroname', 'shape_leng', 'shape_area')
        for col in columns:
            self.assertTrue(col in df.columns, 'Column {} missing'.format(col))
        self.assertTrue(all(df['geometry'].type == 'MultiPolygon'))

    def test_read_postgis_default(self):
        if not self.run_db_test:
            raise unittest.case.SkipTest()

        with psycopg2.connect(dbname='test_geopandas') as con:
            sql = "SELECT * FROM nybb;"
            df = GeoDataFrame.read_postgis(sql, con)

        self._validate_sql(df)

    def test_read_postgis_custom_geom_col(self):
        if not self.run_db_test:
            raise unittest.case.SkipTest()

        with psycopg2.connect(dbname='test_geopandas') as con:
            sql = """SELECT
                     borocode, boroname, shape_leng, shape_area,
                     geom AS __geometry__
                     FROM nybb;"""
            df = GeoDataFrame.read_postgis(sql, con, geom_col='__geometry__')

        self._validate_sql(df)
