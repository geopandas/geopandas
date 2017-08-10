from __future__ import absolute_import

import json
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from shapely.geometry import Point, Polygon

import fiona
import geopandas
from geopandas import GeoDataFrame, read_file, GeoSeries
from geopandas.tests.util import assert_geoseries_equal, connect, create_db, \
    PACKAGE_DIR, unittest, validate_boro_df


class TestDataFrame(unittest.TestCase):

    def setUp(self):
        N = 10

        nybb_filename = geopandas.datasets.get_path('nybb')

        self.df = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.boros = self.df['BoroName']
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry': Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)
        self.df3 = read_file(os.path.join(PACKAGE_DIR, 'examples', 'null_geom.geojson'))
        self.line_paths = self.df3['Name']

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
        self.assert_('location' in df2)
        self.assertEqual(df2.crs, 'dummy_crs')
        self.assertEqual(df2.geometry.crs, 'dummy_crs')
        # reset so it outputs okay
        df2.crs = df.crs
        assert_geoseries_equal(df2.geometry, GeoSeries(geom2, crs=df2.crs))

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
        new_geom = [Point(x, y) for x, y in zip(range(len(self.df)),
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
            df.geometry = list(range(df.shape[0]))

        with self.assertRaises(KeyError):
            df = self.df.copy()
            del df['geometry']
            df['geometry']

        # ndim error
        with self.assertRaises(ValueError):
            df = self.df.copy()
            df.geometry = df

    def test_set_geometry(self):
        geom = GeoSeries([Point(x, y) for x, y in zip(range(5), range(5))])
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
        geom = [Point(x, y) for x, y in zip(range(5), range(5))]
        ret = self.df.set_geometry(geom, inplace=True)
        self.assert_(ret is None)
        geom = GeoSeries(geom, index=self.df.index, crs=self.df.crs)
        assert_geoseries_equal(self.df.geometry, geom)

    def test_set_geometry_series(self):
        # Test when setting geometry with a Series that
        # alignment will occur
        #
        # Reverse the index order
        # Set the Series to be Point(i,i) where i is the index
        self.df.index = range(len(self.df)-1, -1, -1)

        d = {}
        for i in range(len(self.df)):
            d[i] = Point(i, i)
        g = GeoSeries(d)
        # At this point, the DataFrame index is [4,3,2,1,0] and the
        # GeoSeries index is [0,1,2,3,4]. Make sure set_geometry aligns
        # them to match indexes
        df = self.df.set_geometry(g)

        for i, r in df.iterrows():
            self.assertAlmostEqual(i, r['geometry'].x)
            self.assertAlmostEqual(i, r['geometry'].y)

    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)

    def test_to_json_geom_col(self):
        df = self.df.copy()
        df['geom'] = df['geometry']
        df['geometry'] = np.arange(len(df))
        df.set_geometry('geom', inplace=True)

        text = df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)

    def test_to_json_na(self):
        # Set a value as nan and make sure it's written
        self.df.loc[self.df['BoroName']=='Queens', 'Shape_Area'] = np.nan

        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(len(data['features']) == 5)
        for f in data['features']:
            props = f['properties']
            self.assertEqual(len(props), 4)
            if props['BoroName'] == 'Queens':
                self.assertTrue(props['Shape_Area'] is None)

    def test_to_json_bad_na(self):
        # Check that a bad na argument raises error
        with self.assertRaises(ValueError):
            text = self.df.to_json(na='garbage')

    def test_to_json_dropna(self):
        self.df.loc[self.df['BoroName']=='Queens', 'Shape_Area'] = np.nan
        self.df.loc[self.df['BoroName']=='Bronx', 'Shape_Leng'] = np.nan

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
        self.df.loc[self.df['BoroName']=='Queens', 'Shape_Area'] = np.nan
        self.df.loc[self.df['BoroName']=='Bronx', 'Shape_Leng'] = np.nan

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
        # Read layer back in
        df = GeoDataFrame.from_file(tempfilename)
        self.assertTrue('geometry' in df)
        self.assertTrue(len(df) == 5)
        self.assertTrue(np.alltrue(df['BoroName'].values == self.boros))

        # Write layer with null geometry out to file
        tempfilename = os.path.join(self.tempdir, 'null_geom.shp')
        self.df3.to_file(tempfilename)
        # Read layer back in
        df3 = GeoDataFrame.from_file(tempfilename)
        self.assertTrue('geometry' in df3)
        self.assertTrue(len(df3) == 2)
        self.assertTrue(np.alltrue(df3['Name'].values == self.line_paths))

    def test_to_file_types(self):
        """ Test various integer type columns (GH#93) """
        tempfilename = os.path.join(self.tempdir, 'int.shp')
        int_types = [np.int, np.int8, np.int16, np.int32, np.int64, np.intp,
                     np.uint8, np.uint16, np.uint32, np.uint64, np.long]
        geometry = self.df2.geometry
        data = dict((str(i), np.arange(len(geometry), dtype=dtype))
                     for i, dtype in enumerate(int_types))
        df = GeoDataFrame(data, geometry=geometry)
        df.to_file(tempfilename)

    def test_mixed_types_to_file(self):
        """ Test that mixed geometry types raise error when writing to file """
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        s = GeoDataFrame({'geometry': [Point(0, 0),
                                        Polygon([(0, 0), (1, 0), (1, 1)])]})
        with self.assertRaises(ValueError):
            s.to_file(tempfilename)

    def test_to_file_schema(self):
        """
        Ensure that the file is written according to the schema
        if it is specified

        """
        try:
            from collections import OrderedDict
        except ImportError:
            from ordereddict import OrderedDict

        tempfilename = os.path.join(self.tempdir, 'test.shp')
        properties = OrderedDict([
            ('Shape_Leng', 'float:19.11'),
            ('BoroName', 'str:40'),
            ('BoroCode', 'int:10'),
            ('Shape_Area', 'float:19.11'),
        ])
        schema = {'geometry': 'Polygon', 'properties': properties}

        # Take the first 2 features to speed things up a bit
        self.df.iloc[:2].to_file(tempfilename, schema=schema)

        with fiona.open(tempfilename) as f:
            result_schema = f.schema

        self.assertEqual(result_schema, schema)

    def test_bool_index(self):
        # Find boros with 'B' in their name
        df = self.df[self.df['BoroName'].str.contains('B')]
        self.assertTrue(len(df) == 2)
        boros = df['BoroName'].values
        self.assertTrue('Brooklyn' in boros)
        self.assertTrue('Bronx' in boros)
        self.assertTrue(type(df) is GeoDataFrame)

    def test_coord_slice_points(self):
        self.assertTrue(self.df2.cx[-2:-1, -2:-1].empty)
        assert_frame_equal(self.df2, self.df2.cx[:, :])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, :])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[:, 5:])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, 5:])

    def test_transform(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        self.assertTrue(all(df2['geometry'].geom_almost_equals(utm['geometry'], decimal=2)))

    def test_to_crs_geo_column_name(self):
        # Test to_crs() with different geometry column name (GH#339)
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        df2 = df2.rename(columns={'geometry': 'geom'})
        df2.set_geometry('geom', inplace=True)
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        self.assertEqual(lonlat.geometry.name, 'geom')
        self.assertEqual(utm.geometry.name, 'geom')
        self.assertTrue(all(df2.geometry.geom_almost_equals(utm.geometry, decimal=2)))

    def test_from_features(self):
        nybb_filename = geopandas.datasets.get_path('nybb')
        with fiona.open(nybb_filename) as f:
            features = list(f)
            crs = f.crs

        df = GeoDataFrame.from_features(features, crs=crs)
        df.rename(columns=lambda x: x.lower(), inplace=True)
        validate_boro_df(self, df)
        self.assert_(df.crs == crs)

    def test_from_features_unaligned_properties(self):
        p1 = Point(1, 1)
        f1 = {'type': 'Feature',
                'properties': {'a': 0},
                'geometry': p1.__geo_interface__}

        p2 = Point(2, 2)
        f2 = {'type': 'Feature',
                'properties': {'b': 1},
                'geometry': p2.__geo_interface__}

        p3 = Point(3, 3)
        f3 = {'type': 'Feature',
                'properties': {'a': 2},
                'geometry': p3.__geo_interface__}

        df = GeoDataFrame.from_features([f1, f2, f3])

        result = df[['a', 'b']]
        expected = pd.DataFrame.from_dict([{'a': 0, 'b': np.nan},
                                           {'a': np.nan, 'b': 1},
                                           {'a': 2, 'b': np.nan}])
        assert_frame_equal(expected, result)

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

    def test_geodataframe_geointerface(self):
        self.assertEqual(self.df.__geo_interface__['type'], 'FeatureCollection')
        self.assertEqual(len(self.df.__geo_interface__['features']),
                         self.df.shape[0])

    def test_geodataframe_geojson_no_bbox(self):
        geo = self.df._to_geo(na="null", show_bbox=False)
        self.assertFalse('bbox' in geo.keys())
        for feature in geo['features']:
            self.assertFalse('bbox' in feature.keys())

    def test_geodataframe_geojson_bbox(self):
        geo = self.df._to_geo(na="null", show_bbox=True)
        self.assertTrue('bbox' in geo.keys())
        self.assertEqual(len(geo['bbox']), 4)
        self.assertTrue(isinstance(geo['bbox'], tuple))
        for feature in geo['features']:
            self.assertTrue('bbox' in feature.keys())

    def test_pickle(self):
        filename = os.path.join(self.tempdir, 'df.pkl')
        self.df.to_pickle(filename)
        unpickled = pd.read_pickle(filename)
        assert_frame_equal(self.df, unpickled)
        self.assertEqual(self.df.crs, unpickled.crs)
