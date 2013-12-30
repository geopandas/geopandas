import json
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, read_file, GeoSeries
from .util import unittest, download_nybb, assert_geoseries_equal, connect, \
                  create_db, validate_boro_df


class TestDataFrame(unittest.TestCase):

    def setUp(self):
        N = 10

        nybb_filename = download_nybb()

        self.df = read_file('/nybb_13a/nybb.shp', vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.boros = np.array(['Staten Island', 'Queens', 'Brooklyn',
                               'Manhattan', 'Bronx'])
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        self.assertTrue(type(self.df2) is GeoDataFrame)
        self.assertTrue(self.df2.crs == self.crs)

    def test_different_geo_colname(self):
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        df = GeoDataFrame(data, crs=self.crs, geometry='location')
        locs = GeoSeries(data['location'], crs=self.crs)
        assert_geoseries_equal(df.geometry, locs)
        self.assert_('geometry' not in df)
        self.assertEqual(df.geometry.name, 'location')
        # internal implementation detail
        self.assertEqual(df._geometry_column_name, 'location')

        geom2 = [Point(x, y) for x, y in zip(range(5, 10), range(5))]
        df2 = df.set_geometry(geom2, crs='dummy_crs')
        self.assert_('geometry' in df2)
        self.assert_('location' in df2)
        self.assertEqual(df2.crs, 'dummy_crs')
        self.assertEqual(df2.geometry.crs, 'dummy_crs')
        # reset so it outputs okay
        df2.crs = df.crs
        assert_geoseries_equal(df2.geometry, GeoSeries(geom2, crs=df2.crs))
        # for right now, non-geometry comes back as series
        assert_geoseries_equal(df2['location'], df['location'],
                                  check_series_type=False, check_dtype=False)

    def test_geo_getitem(self):
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        df = GeoDataFrame(data, crs=self.crs, geometry='location')
        self.assert_(isinstance(df.geometry, GeoSeries))
        df['geometry'] = df["A"]
        self.assert_(isinstance(df.geometry, GeoSeries))
        self.assertEqual(df.geometry[0], data['location'][0])
        # good if this changed in the future
        self.assert_(not isinstance(df['geometry'], GeoSeries))
        self.assert_(isinstance(df['location'], GeoSeries))

        data["geometry"] = [Point(x + 1, y - 1) for x, y in zip(range(5), range(5))]
        df = GeoDataFrame(data, crs=self.crs)
        self.assert_(isinstance(df.geometry, GeoSeries))
        self.assert_(isinstance(df['geometry'], GeoSeries))
        # good if this changed in the future
        self.assert_(not isinstance(df['location'], GeoSeries))

    def test_geometry_property(self):
        assert_geoseries_equal(self.df.geometry, self.df['geometry'],
                                  check_dtype=True, check_index_type=True)

        df = self.df.copy()
        new_geom = [Point(x,y) for x, y in zip(range(len(self.df)),
                                               range(len(self.df)))]
        df.geometry = new_geom

        new_geom = GeoSeries(new_geom, index=df.index, crs=df.crs)
        assert_geoseries_equal(df.geometry, new_geom)
        assert_geoseries_equal(df['geometry'], new_geom)

        # new crs
        gs = GeoSeries(new_geom, crs="epsg:26018")
        df.geometry = gs
        self.assertEqual(df.crs, "epsg:26018")

    def test_geometry_property_errors(self):
        with self.assertRaises(AttributeError):
            df = self.df.copy()
            del df['geometry']
            df.geometry

        # list-like error
        with self.assertRaises(ValueError):
            df = self.df2.copy()
            df.geometry = 'value1'

        # list-like error
        with self.assertRaises(ValueError):
            df = self.df.copy()
            df.geometry = 'apple'

        # non-geometry error
        with self.assertRaises(TypeError):
            df = self.df.copy()
            df.geometry = range(df.shape[0])

        with self.assertRaises(KeyError):
            df = self.df.copy()
            del df['geometry']
            df['geometry']

        # ndim error
        with self.assertRaises(ValueError):
            df = self.df.copy()
            df.geometry = df

    def test_set_geometry(self):
        geom = GeoSeries([Point(x,y) for x,y in zip(range(5), range(5))])
        original_geom = self.df.geometry

        df2 = self.df.set_geometry(geom)
        self.assert_(self.df is not df2)
        assert_geoseries_equal(df2.geometry, geom)
        assert_geoseries_equal(self.df.geometry, original_geom)
        assert_geoseries_equal(self.df['geometry'], self.df.geometry)
        # unknown column
        with self.assertRaises(ValueError):
            self.df.set_geometry('nonexistent-column')

        # ndim error
        with self.assertRaises(ValueError):
            self.df.set_geometry(self.df)

        # new crs - setting should default to GeoSeries' crs
        gs = GeoSeries(geom, crs="epsg:26018")
        new_df = self.df.set_geometry(gs)
        self.assertEqual(new_df.crs, "epsg:26018")

        # explicit crs overrides self and dataframe
        new_df = self.df.set_geometry(gs, crs="epsg:27159")
        self.assertEqual(new_df.crs, "epsg:27159")
        self.assertEqual(new_df.geometry.crs, "epsg:27159")

        # Series should use dataframe's
        new_df = self.df.set_geometry(geom.values)
        self.assertEqual(new_df.crs, self.df.crs)
        self.assertEqual(new_df.geometry.crs, self.df.crs)

    def test_set_geometry_col(self):
        g = self.df.geometry
        g_simplified = g.simplify(100)
        self.df['simplified_geometry'] = g_simplified
        df2 = self.df.set_geometry('simplified_geometry')

        # Drop is false by default
        self.assert_('simplified_geometry' in df2)
        assert_geoseries_equal(df2.geometry, g_simplified)

        # If True, drops column and renames to geometry
        df3 = self.df.set_geometry('simplified_geometry', drop=True)
        self.assert_('simplified_geometry' not in df3)
        assert_geoseries_equal(df3.geometry, g_simplified)

    def test_set_geometry_inplace(self):
        geom = [Point(x,y) for x,y in zip(range(5), range(5))]
        ret = self.df.set_geometry(geom, inplace=True)
        self.assert_(ret is None)
        geom = GeoSeries(geom, index=self.df.index, crs=self.df.crs)
        assert_geoseries_equal(self.df.geometry, geom)

    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)

    def test_to_json_na(self):
        # Set a value as nan and make sure it's written
        self.df['Shape_Area'][self.df['BoroName']=='Queens'] = np.nan

        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(len(data['features']) == 5)
        for f in data['features']:
            props = f['properties']
            self.assertEqual(len(props), 4)
            if props['BoroName'] == 'Queens':
                self.assertTrue(props['Shape_Area'] is None)

    def test_to_json_dropna(self):
        self.df['Shape_Area'][self.df['BoroName']=='Queens'] = np.nan
        self.df['Shape_Leng'][self.df['BoroName']=='Bronx'] = np.nan

        text = self.df.to_json(na='drop')
        data = json.loads(text)
        self.assertEqual(len(data['features']), 5)
        for f in data['features']:
            props = f['properties']
            if props['BoroName'] == 'Queens':
                self.assertEqual(len(props), 3)
                self.assertTrue('Shape_Area' not in props)
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                self.assertTrue('Shape_Leng' in props)
            elif props['BoroName'] == 'Bronx':
                self.assertEqual(len(props), 3)
                self.assertTrue('Shape_Leng' not in props)
                self.assertTrue('Shape_Area' in props)
            else:
                self.assertEqual(len(props), 4)

    def test_to_json_keepna(self):
        self.df['Shape_Area'][self.df['BoroName']=='Queens'] = np.nan
        self.df['Shape_Leng'][self.df['BoroName']=='Bronx'] = np.nan

        text = self.df.to_json(na='keep')
        data = json.loads(text)
        self.assertEqual(len(data['features']), 5)
        for f in data['features']:
            props = f['properties']
            self.assertEqual(len(props), 4)
            if props['BoroName'] == 'Queens':
                self.assertTrue(np.isnan(props['Shape_Area']))
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                self.assertTrue('Shape_Leng' in props)
            elif props['BoroName'] == 'Bronx':
                self.assertTrue(np.isnan(props['Shape_Leng']))
                self.assertTrue('Shape_Area' in props)

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
        self.assertTrue(all(df2['geometry'].geom_almost_equals(utm['geometry'], decimal=2)))

    def test_from_postgis_default(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = "SELECT * FROM nybb;"
            df = GeoDataFrame.from_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(self, df)

    def test_from_postgis_custom_geom_col(self):
        con = connect('test_geopandas')
        if con is None or not create_db(self.df):
            raise unittest.case.SkipTest()

        try:
            sql = """SELECT
                     borocode, boroname, shape_leng, shape_area,
                     geom AS __geometry__
                     FROM nybb;"""
            df = GeoDataFrame.from_postgis(sql, con, geom_col='__geometry__')
        finally:
            con.close()

        validate_boro_df(self, df)

    def test_dataframe_to_geodataframe(self):
        df = pd.DataFrame({"A": range(len(self.df)), "location":
                           list(self.df.geometry)}, index=self.df.index)
        gf = df.set_geometry('location', crs=self.df.crs)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(gf, GeoDataFrame)
        assert_geoseries_equal(gf.geometry, self.df.geometry)
        self.assertEqual(gf.geometry.name, 'location')
        self.assert_('geometry' not in gf)

        gf2 = df.set_geometry('location', crs=self.df.crs, drop=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(gf2, GeoDataFrame)
        self.assertEqual(gf2.geometry.name, 'geometry')
        self.assert_('geometry' in gf2)
        self.assert_('location' not in gf2)
        self.assert_('location' in df)

        # should be a copy
        df.ix[0, "A"] = 100
        self.assertEqual(gf.ix[0, "A"], 0)
        self.assertEqual(gf2.ix[0, "A"], 0)

        with self.assertRaises(ValueError):
            df.set_geometry('location', inplace=True)
