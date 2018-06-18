from __future__ import absolute_import

import tempfile
import shutil

from shapely.geometry import Point

from geopandas import GeoDataFrame, read_file
from geopandas import overlay
from geopandas import datasets
from geopandas.testing import assert_geodataframe_equal

import unittest

# Load qgis overlays
qgispath = datasets._module_path+'/polys/'
union_qgis = read_file(qgispath+'qgis_union.shp')
diff_qgis = read_file(qgispath+'qgis_diff.shp')
symdiff_qgis = read_file(qgispath+'qgis_symdif.shp')
intersect_qgis = read_file(qgispath+'qgis_intersection.shp')
ident_qgis = union_qgis[union_qgis.df1.isnull()==False].copy()
df1 = read_file(qgispath+'df1.shp')
df2 = read_file(qgispath+'df2.shp')
# Eliminate observations without geometries (issue from QGIS)
union_qgis = union_qgis[union_qgis.is_valid]
union_qgis.reset_index(inplace=True, drop=True)
diff_qgis = diff_qgis[diff_qgis.is_valid]
diff_qgis.reset_index(inplace=True, drop=True)
symdiff_qgis = symdiff_qgis[symdiff_qgis.is_valid]
symdiff_qgis.reset_index(inplace=True, drop=True)
intersect_qgis = intersect_qgis[intersect_qgis.is_valid]
intersect_qgis.reset_index(inplace=True, drop=True)
ident_qgis = ident_qgis[ident_qgis.is_valid]
ident_qgis.reset_index(inplace=True, drop=True)
# Order GeoDataFrames
cols = ['df1', 'df2']
union_qgis.sort_values(cols, inplace=True)
union_qgis.reset_index(inplace=True, drop=True)
symdiff_qgis.sort_values(cols, inplace=True)
symdiff_qgis.reset_index(inplace=True, drop=True)
intersect_qgis.sort_values(cols, inplace=True)
intersect_qgis.reset_index(inplace=True, drop=True)
ident_qgis.sort_values(cols, inplace=True)
ident_qgis.reset_index(inplace=True, drop=True)
diff_qgis.sort_values(cols[:-1], inplace=True)
diff_qgis.reset_index(inplace=True, drop=True)

class TestDataFrame(unittest.TestCase):
    def setUp(self):
        # Create original data again
        self.polydf = df1.copy()
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        self.polydf2 = df2.copy()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df = df.drop(['idx1', 'idx2'], axis=1)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, union_qgis.shape)
        self.assertTrue('df2' in df.columns and 'df1' in df.columns)
        self.assertTrue((df.area/union_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/union_qgis.boundary.length).mean()==1)
        assert_geodataframe_equal(df, union_qgis)

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df = df.drop(['idx1', 'idx2'], axis=1)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, intersect_qgis.shape)
        self.assertTrue('df2' in df.columns and 'df1' in df.columns)
        self.assertTrue((df.area/intersect_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/intersect_qgis.boundary.length).mean()==1)
        assert_geodataframe_equal(df, intersect_qgis)

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df = df.drop(['idx1', 'idx2'], axis=1)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, ident_qgis.shape)
        self.assertTrue('df2' in df.columns and 'df1' in df.columns)
        self.assertTrue((df.area/ident_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/ident_qgis.boundary.length).mean()==1)
        assert_geodataframe_equal(df, ident_qgis)

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, symdiff_qgis.shape)
        self.assertTrue('df2' in df.columns and 'df1' in df.columns)
        self.assertTrue((df.area/symdiff_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/symdiff_qgis.boundary.length).mean()==1)
        assert_geodataframe_equal(df, symdiff_qgis)

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        df.sort_values(cols[:-1], inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, diff_qgis.shape)
        self.assertTrue('df2' not in df.columns and 'df1' in df.columns)
        self.assertTrue((df.area/diff_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/diff_qgis.boundary.length).mean()==1)
        assert_geodataframe_equal(df, diff_qgis)
