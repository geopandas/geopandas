from __future__ import absolute_import

import json
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import fiona

import geopandas
from geopandas import GeoDataFrame, read_file, GeoSeries

import pytest
from pandas.util.testing import (
    assert_frame_equal, assert_index_equal, assert_series_equal)
from geopandas.tests.util import (
    assert_geoseries_equal, connect, create_postgis, PACKAGE_DIR,
    validate_boro_df)


class TestDataFrame:

    def setup_method(self):
        N = 10

        nybb_filename = geopandas.datasets.get_path('nybb')

        self.df = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.boros = self.df['BoroName']
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry': Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)
        self.df3 = read_file(
            os.path.join(PACKAGE_DIR, 'examples', 'null_geom.geojson'))
        self.line_paths = self.df3['Name']

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        assert type(self.df2) is GeoDataFrame
        assert self.df2.crs == self.crs

    def test_different_geo_colname(self):
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        df = GeoDataFrame(data, crs=self.crs, geometry='location')
        locs = GeoSeries(data['location'], crs=self.crs)
        assert_geoseries_equal(df.geometry, locs)
        assert 'geometry' not in df
        assert df.geometry.name == 'location'
        # internal implementation detail
        assert df._geometry_column_name == 'location'

        geom2 = [Point(x, y) for x, y in zip(range(5, 10), range(5))]
        df2 = df.set_geometry(geom2, crs='dummy_crs')
        assert 'location' in df2
        assert df2.crs == 'dummy_crs'
        assert df2.geometry.crs == 'dummy_crs'
        # reset so it outputs okay
        df2.crs = df.crs
        assert_geoseries_equal(df2.geometry, GeoSeries(geom2, crs=df2.crs))

    def test_geo_getitem(self):
        data = {"A": range(5), "B": range(-5, 0),
                "location": [Point(x, y) for x, y in zip(range(5), range(5))]}
        df = GeoDataFrame(data, crs=self.crs, geometry='location')
        assert isinstance(df.geometry, GeoSeries)
        df['geometry'] = df["A"]
        assert isinstance(df.geometry, GeoSeries)
        assert df.geometry[0] == data['location'][0]
        # good if this changed in the future
        assert not isinstance(df['geometry'], GeoSeries)
        assert isinstance(df['location'], GeoSeries)

        data["geometry"] = [Point(x + 1, y - 1) for x, y in zip(range(5),
                                                                range(5))]
        df = GeoDataFrame(data, crs=self.crs)
        assert isinstance(df.geometry, GeoSeries)
        assert isinstance(df['geometry'], GeoSeries)
        # good if this changed in the future
        assert not isinstance(df['location'], GeoSeries)

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
        assert df.crs == "epsg:26018"

    def test_geometry_property_errors(self):
        with pytest.raises(AttributeError):
            df = self.df.copy()
            del df['geometry']
            df.geometry

        # list-like error
        with pytest.raises(ValueError):
            df = self.df2.copy()
            df.geometry = 'value1'

        # list-like error
        with pytest.raises(ValueError):
            df = self.df.copy()
            df.geometry = 'apple'

        # non-geometry error
        with pytest.raises(TypeError):
            df = self.df.copy()
            df.geometry = list(range(df.shape[0]))

        with pytest.raises(KeyError):
            df = self.df.copy()
            del df['geometry']
            df['geometry']

        # ndim error
        with pytest.raises(ValueError):
            df = self.df.copy()
            df.geometry = df

    def test_set_geometry(self):
        geom = GeoSeries([Point(x, y) for x, y in zip(range(5), range(5))])
        original_geom = self.df.geometry

        df2 = self.df.set_geometry(geom)
        assert self.df is not df2
        assert_geoseries_equal(df2.geometry, geom)
        assert_geoseries_equal(self.df.geometry, original_geom)
        assert_geoseries_equal(self.df['geometry'], self.df.geometry)
        # unknown column
        with pytest.raises(ValueError):
            self.df.set_geometry('nonexistent-column')

        # ndim error
        with pytest.raises(ValueError):
            self.df.set_geometry(self.df)

        # new crs - setting should default to GeoSeries' crs
        gs = GeoSeries(geom, crs="epsg:26018")
        new_df = self.df.set_geometry(gs)
        assert new_df.crs == "epsg:26018"

        # explicit crs overrides self and dataframe
        new_df = self.df.set_geometry(gs, crs="epsg:27159")
        assert new_df.crs == "epsg:27159"
        assert new_df.geometry.crs == "epsg:27159"

        # Series should use dataframe's
        new_df = self.df.set_geometry(geom.values)
        assert new_df.crs == self.df.crs
        assert new_df.geometry.crs == self.df.crs

    def test_set_geometry_col(self):
        g = self.df.geometry
        g_simplified = g.simplify(100)
        self.df['simplified_geometry'] = g_simplified
        df2 = self.df.set_geometry('simplified_geometry')

        # Drop is false by default
        assert 'simplified_geometry' in df2
        assert_geoseries_equal(df2.geometry, g_simplified)

        # If True, drops column and renames to geometry
        df3 = self.df.set_geometry('simplified_geometry', drop=True)
        assert 'simplified_geometry' not in df3
        assert_geoseries_equal(df3.geometry, g_simplified)

    def test_set_geometry_inplace(self):
        geom = [Point(x, y) for x, y in zip(range(5), range(5))]
        ret = self.df.set_geometry(geom, inplace=True)
        assert ret is None
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
            assert i == r['geometry'].x
            assert i == r['geometry'].y

    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        assert data['type'] == 'FeatureCollection'
        assert len(data['features']) == 5

    def test_to_json_geom_col(self):
        df = self.df.copy()
        df['geom'] = df['geometry']
        df['geometry'] = np.arange(len(df))
        df.set_geometry('geom', inplace=True)

        text = df.to_json()
        data = json.loads(text)
        assert data['type'] == 'FeatureCollection'
        assert len(data['features']) == 5

    def test_to_json_na(self):
        # Set a value as nan and make sure it's written
        self.df.loc[self.df['BoroName'] == 'Queens', 'Shape_Area'] = np.nan

        text = self.df.to_json()
        data = json.loads(text)
        assert len(data['features']) == 5
        for f in data['features']:
            props = f['properties']
            assert len(props) == 4
            if props['BoroName'] == 'Queens':
                assert props['Shape_Area'] is None

    def test_to_json_bad_na(self):
        # Check that a bad na argument raises error
        with pytest.raises(ValueError):
            self.df.to_json(na='garbage')

    def test_to_json_dropna(self):
        self.df.loc[self.df['BoroName'] == 'Queens', 'Shape_Area'] = np.nan
        self.df.loc[self.df['BoroName'] == 'Bronx', 'Shape_Leng'] = np.nan

        text = self.df.to_json(na='drop')
        data = json.loads(text)
        assert len(data['features']) == 5
        for f in data['features']:
            props = f['properties']
            if props['BoroName'] == 'Queens':
                assert len(props) == 3
                assert 'Shape_Area' not in props
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                assert 'Shape_Leng' in props
            elif props['BoroName'] == 'Bronx':
                assert len(props) == 3
                assert 'Shape_Leng' not in props
                assert 'Shape_Area' in props
            else:
                assert len(props) == 4

    def test_to_json_keepna(self):
        self.df.loc[self.df['BoroName'] == 'Queens', 'Shape_Area'] = np.nan
        self.df.loc[self.df['BoroName'] == 'Bronx', 'Shape_Leng'] = np.nan

        text = self.df.to_json(na='keep')
        data = json.loads(text)
        assert len(data['features']) == 5
        for f in data['features']:
            props = f['properties']
            assert len(props) == 4
            if props['BoroName'] == 'Queens':
                assert np.isnan(props['Shape_Area'])
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                assert 'Shape_Leng' in props
            elif props['BoroName'] == 'Bronx':
                assert np.isnan(props['Shape_Leng'])
                assert 'Shape_Area' in props

    def test_copy(self):
        df2 = self.df.copy()
        assert type(df2) is GeoDataFrame
        assert self.df.crs == df2.crs

    def test_to_file(self):
        """ Test to_file and from_file """
        tempfilename = os.path.join(self.tempdir, 'boros.shp')
        self.df.to_file(tempfilename)
        # Read layer back in
        df = GeoDataFrame.from_file(tempfilename)
        assert 'geometry' in df
        assert len(df) == 5
        assert np.alltrue(df['BoroName'].values == self.boros)

        # Write layer with null geometry out to file
        tempfilename = os.path.join(self.tempdir, 'null_geom.shp')
        self.df3.to_file(tempfilename)
        # Read layer back in
        df3 = GeoDataFrame.from_file(tempfilename)
        assert 'geometry' in df3
        assert len(df3) == 2
        assert np.alltrue(df3['Name'].values == self.line_paths)

    def test_to_file_bool(self):
        """Test error raise when writing with a boolean column (GH #437)."""

        # still want a temp dir in case this test passes
        tempfilename = os.path.join(self.tempdir, 'boros.shp')
        df_with_bool = self.df.copy()
        df_with_bool['bool_column'] = True
        with pytest.raises(ValueError):
            df_with_bool.to_file(tempfilename)

    def test_to_file_with_point_z(self):
        """Test that 3D geometries are retained in writes (GH #612)."""

        tempfilename = os.path.join(self.tempdir, 'test_3Dpoint.shp')
        point3d = Point(0, 0, 500)
        point2d = Point(1, 1)
        df = GeoDataFrame({'a': [1, 2]}, geometry=[point3d, point2d], crs={})
        df.to_file(tempfilename)
        df_read = GeoDataFrame.from_file(tempfilename)
        assert_geoseries_equal(df.geometry, df_read.geometry)

    def test_to_file_with_poly_z(self):
        """Test that 3D geometries are retained in writes (GH #612)."""

        tempfilename = os.path.join(self.tempdir, 'test_3Dpoly.shp')
        poly3d = Polygon([[0, 0, 5], [0, 1, 5], [1, 1, 5], [1, 0, 5]])
        poly2d = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        df = GeoDataFrame({'a': [1, 2]}, geometry=[poly3d, poly2d], crs={})
        df.to_file(tempfilename)
        df_read = GeoDataFrame.from_file(tempfilename)
        assert_geoseries_equal(df.geometry, df_read.geometry)

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
        with pytest.raises(ValueError):
            s.to_file(tempfilename)

    def test_empty_to_file(self):
        input_empty_df = GeoDataFrame()
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        with pytest.raises(
            ValueError, match="Cannot write empty DataFrame to file."):
            input_empty_df.to_file(tempfilename)

    def test_to_file_schema(self):
        """
        Ensure that the file is written according to the schema
        if it is specified

        """
        from collections import OrderedDict

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

        assert result_schema == schema

    def test_bool_index(self):
        # Find boros with 'B' in their name
        df = self.df[self.df['BoroName'].str.contains('B')]
        assert len(df) == 2
        boros = df['BoroName'].values
        assert 'Brooklyn' in boros
        assert 'Bronx' in boros
        assert type(df) is GeoDataFrame

    def test_coord_slice_points(self):
        assert self.df2.cx[-2:-1, -2:-1].empty
        assert_frame_equal(self.df2, self.df2.cx[:, :])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, :])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[:, 5:])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, 5:])

    def test_transform(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert all(df2['geometry'].geom_almost_equals(utm['geometry'],
                                                      decimal=2))

    def test_transform_inplace(self):
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        lonlat = df2.to_crs(epsg=4326)
        df2.to_crs(epsg=4326, inplace=True)
        assert all(df2['geometry'].geom_almost_equals(lonlat['geometry'],
                                                      decimal=2))

    def test_to_crs_geo_column_name(self):
        # Test to_crs() with different geometry column name (GH#339)
        df2 = self.df2.copy()
        df2.crs = {'init': 'epsg:26918', 'no_defs': True}
        df2 = df2.rename(columns={'geometry': 'geom'})
        df2.set_geometry('geom', inplace=True)
        lonlat = df2.to_crs(epsg=4326)
        utm = lonlat.to_crs(epsg=26918)
        assert lonlat.geometry.name == 'geom'
        assert utm.geometry.name == 'geom'
        assert all(df2.geometry.geom_almost_equals(utm.geometry, decimal=2))

    def test_from_features(self):
        nybb_filename = geopandas.datasets.get_path('nybb')
        with fiona.open(nybb_filename) as f:
            features = list(f)
            crs = f.crs

        df = GeoDataFrame.from_features(features, crs=crs)
        validate_boro_df(df, case_sensitive=True)
        assert df.crs == crs

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

    def test_from_feature_collection(self):
        data = {'name': ['a', 'b', 'c'],
                'lat': [45, 46, 47.5],
                'lon': [-120, -121.2, -122.9]}

        df = pd.DataFrame(data)
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        gdf = GeoDataFrame(df, geometry=geometry)
        # from_features returns sorted columns
        expected = gdf[['geometry', 'lat', 'lon', 'name']]

        # test FeatureCollection
        res = GeoDataFrame.from_features(gdf.__geo_interface__)
        assert_frame_equal(res, expected)

        # test list of Features
        res = GeoDataFrame.from_features(gdf.__geo_interface__['features'])
        assert_frame_equal(res, expected)

        # test __geo_interface__ attribute (a GeoDataFrame has one)
        res = GeoDataFrame.from_features(gdf)
        assert_frame_equal(res, expected)

    def test_from_postgis_default(self):
        con = connect('test_geopandas')
        if con is None or not create_postgis(self.df):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = GeoDataFrame.from_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(df, case_sensitive=False)

    def test_from_postgis_custom_geom_col(self):
        con = connect('test_geopandas')
        geom_col = "the_geom"
        if con is None or not create_postgis(self.df, geom_col=geom_col):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = GeoDataFrame.from_postgis(sql, con, geom_col=geom_col)
        finally:
            con.close()

        validate_boro_df(df, case_sensitive=False)

    def test_dataframe_to_geodataframe(self):
        df = pd.DataFrame({"A": range(len(self.df)), "location":
                           list(self.df.geometry)}, index=self.df.index)
        gf = df.set_geometry('location', crs=self.df.crs)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(gf, GeoDataFrame)
        assert_geoseries_equal(gf.geometry, self.df.geometry)
        assert gf.geometry.name == 'location'
        assert 'geometry' not in gf

        gf2 = df.set_geometry('location', crs=self.df.crs, drop=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(gf2, GeoDataFrame)
        assert gf2.geometry.name == 'geometry'
        assert 'geometry' in gf2
        assert 'location' not in gf2
        assert 'location' in df

        # should be a copy
        df.loc[0, "A"] = 100
        assert gf.loc[0, "A"] == 0
        assert gf2.loc[0, "A"] == 0

        with pytest.raises(ValueError):
            df.set_geometry('location', inplace=True)

    def test_geodataframe_geointerface(self):
        assert self.df.__geo_interface__['type'] == 'FeatureCollection'
        assert len(self.df.__geo_interface__['features']) == self.df.shape[0]

    def test_geodataframe_geojson_no_bbox(self):
        geo = self.df._to_geo(na="null", show_bbox=False)
        assert 'bbox' not in geo.keys()
        for feature in geo['features']:
            assert 'bbox' not in feature.keys()

    def test_geodataframe_geojson_bbox(self):
        geo = self.df._to_geo(na="null", show_bbox=True)
        assert 'bbox' in geo.keys()
        assert len(geo['bbox']) == 4
        assert isinstance(geo['bbox'], tuple)
        for feature in geo['features']:
            assert 'bbox' in feature.keys()

    def test_pickle(self):
        filename = os.path.join(self.tempdir, 'df.pkl')
        self.df.to_pickle(filename)
        unpickled = pd.read_pickle(filename)
        assert_frame_equal(self.df, unpickled)
        assert self.df.crs == unpickled.crs


def check_geodataframe(df, geometry_column='geometry'):
    assert isinstance(df, GeoDataFrame)
    assert isinstance(df.geometry, GeoSeries)
    assert isinstance(df[geometry_column], GeoSeries)
    assert df._geometry_column_name == geometry_column


class TestConstructor:

    def test_dict(self):
        data = {"A": range(3), "B": np.arange(3.0),
                "geometry": [Point(x, x) for x in range(3)]}
        df = GeoDataFrame(data)
        check_geodataframe(df)

        # with specifying other kwargs
        df = GeoDataFrame(data, index=list('abc'))
        check_geodataframe(df)
        assert_index_equal(df.index, pd.Index(list('abc')))

        df = GeoDataFrame(data, columns=['B', 'A', 'geometry'])
        check_geodataframe(df)
        assert_index_equal(df.columns, pd.Index(['B', 'A', 'geometry']))

        df = GeoDataFrame(data, columns=['A', 'geometry'])
        check_geodataframe(df)
        assert_index_equal(df.columns, pd.Index(['A', 'geometry']))
        assert_series_equal(df['A'], pd.Series(range(3), name='A'))

    def test_dict_of_series(self):

        data = {"A": pd.Series(range(3)), "B": pd.Series(np.arange(3.0)),
                "geometry": GeoSeries([Point(x, x) for x in range(3)])}

        df = GeoDataFrame(data)
        check_geodataframe(df)

        df = GeoDataFrame(data, index=pd.Index([1, 2]))
        check_geodataframe(df)
        assert_index_equal(df.index, pd.Index([1, 2]))
        assert df['A'].tolist() == [1, 2]

        # one non-series -> length is not correct
        data = {"A": pd.Series(range(3)), "B": np.arange(3.0),
                "geometry": GeoSeries([Point(x, x) for x in range(3)])}
        with pytest.raises(ValueError):
            GeoDataFrame(data, index=[1, 2])

    def test_dict_specified_geometry(self):

        data = {"A": range(3), "B": np.arange(3.0),
                "other_geom": [Point(x, x) for x in range(3)]}

        df = GeoDataFrame(data, geometry='other_geom')
        check_geodataframe(df, 'other_geom')

        with pytest.raises(ValueError):
            df = GeoDataFrame(data, geometry='geometry')

        # when no geometry specified -> works but raises error once
        # trying to access geometry
        df = GeoDataFrame(data)

        with pytest.raises(AttributeError):
            _ = df.geometry

        df = df.set_geometry('other_geom')
        check_geodataframe(df, 'other_geom')

        # combined with custom args
        df = GeoDataFrame(data, geometry='other_geom',
                          columns=['B', 'other_geom'])
        check_geodataframe(df, 'other_geom')
        assert_index_equal(df.columns, pd.Index(['B', 'other_geom']))
        assert_series_equal(df['B'], pd.Series(np.arange(3.), name='B'))

        df = GeoDataFrame(data, geometry='other_geom',
                          columns=['other_geom', 'A'])
        check_geodataframe(df, 'other_geom')
        assert_index_equal(df.columns, pd.Index(['other_geom', 'A']))
        assert_series_equal(df['A'], pd.Series(range(3), name='A'))

    def test_array(self):
        data = {"A": range(3), "B": np.arange(3.0),
                "geometry": [Point(x, x) for x in range(3)]}
        a = np.array([data['A'], data['B'], data['geometry']], dtype=object).T

        df = GeoDataFrame(a, columns=['A', 'B', 'geometry'])
        check_geodataframe(df)

        df = GeoDataFrame(a, columns=['A', 'B', 'other_geom'],
                          geometry='other_geom')
        check_geodataframe(df, 'other_geom')

    def test_from_frame(self):
        data = {"A": range(3), "B": np.arange(3.0),
                "geometry": [Point(x, x) for x in range(3)]}
        gpdf = GeoDataFrame(data)
        pddf = pd.DataFrame(data)
        check_geodataframe(gpdf)
        assert type(pddf) == pd.DataFrame

        for df in [gpdf, pddf]:

            res = GeoDataFrame(df)
            check_geodataframe(res)

            res = GeoDataFrame(df, index=pd.Index([0, 2]))
            check_geodataframe(res)
            assert_index_equal(res.index, pd.Index([0, 2]))
            assert res['A'].tolist() == [0, 2]

            res = GeoDataFrame(df, columns=['geometry', 'B'])
            check_geodataframe(res)
            assert_index_equal(res.columns, pd.Index(['geometry', 'B']))

            with pytest.raises(ValueError):
                GeoDataFrame(df, geometry='other_geom')

    def test_from_frame_specified_geometry(self):
        data = {"A": range(3), "B": np.arange(3.0),
                "other_geom": [Point(x, x) for x in range(3)]}

        gpdf = GeoDataFrame(data, geometry='other_geom')
        check_geodataframe(gpdf, 'other_geom')
        pddf = pd.DataFrame(data)

        for df in [gpdf, pddf]:
            res = GeoDataFrame(df, geometry='other_geom')
            check_geodataframe(res, 'other_geom')

        # when passing GeoDataFrame with custom geometry name to constructor
        # an invalid geodataframe is the result TODO is this desired ?
        df = GeoDataFrame(gpdf)
        with pytest.raises(AttributeError):
            df.geometry

    def test_only_geometry(self):
        df = GeoDataFrame(geometry=[Point(x, x) for x in range(3)])
        check_geodataframe(df)

        df = GeoDataFrame({'geometry': [Point(x, x) for x in range(3)]})
        check_geodataframe(df)

        df = GeoDataFrame({'other_geom': [Point(x, x) for x in range(3)]},
                          geometry='other_geom')
        check_geodataframe(df, 'other_geom')

    def test_no_geometries(self):
        # keeps GeoDataFrame class (no DataFrame)
        data = {"A": range(3), "B": np.arange(3.0)}
        df = GeoDataFrame(data)
        assert type(df) == GeoDataFrame

    def test_empty(self):
        df = GeoDataFrame()
        assert type(df) == GeoDataFrame

        df = GeoDataFrame({'A': [], 'B': []}, geometry=[])
        assert type(df) == GeoDataFrame
