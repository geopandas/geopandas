from __future__ import absolute_import

import tempfile
import shutil

from shapely.geometry import Point

from geopandas import GeoDataFrame, read_file
from geopandas.tests.util import unittest, download_nybb
from geopandas import overlay
from geopandas import datasets

class TestDataFrame(unittest.TestCase):

    def setUp(self):
        # Load qgis overlays
        qgispath = datasets.module_path+'/qgis_overlay/'
        union_qgis = read_file(qgispath+'union_qgis.shp')
        diff_qgis = read_file(qgispath+'diff_qgis.shp')
        symdiff_qgis = read_file(qgispath+'symdiff_qgis.shp')
        intersect_qgis = read_file(qgispath+'intersect_qgis.shp')
        ident_qgis = union_qgis[union_qgis.BoroCode.isnull()==False].copy()
        df1 = read_file(qgispath+'polydf.shp')
        df2 = read_file(qgispath+'polydf2.shp')
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
        self.cols = ['BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area', 'value1', 'value2']
        union_qgis.sort_values(self.cols, inplace=True)
        union_qgis.reset_index(inplace=True, drop=True)
        symdiff_qgis.sort_values(self.cols, inplace=True)
        symdiff_qgis.reset_index(inplace=True, drop=True)
        intersect_qgis.sort_values(self.cols, inplace=True)
        intersect_qgis.reset_index(inplace=True, drop=True)
        ident_qgis.sort_values(self.cols, inplace=True)
        ident_qgis.reset_index(inplace=True, drop=True)
        diff_qgis.sort_values(self.cols[:-2], inplace=True)
        diff_qgis.reset_index(inplace=True, drop=True)

        # Create original data again
        N = 10
        nybb_filename, nybb_zip_path = download_nybb()
        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        b = [int(x) for x in self.polydf.total_bounds]
        self.polydf2 = GeoDataFrame([
            {'geometry' : Point(x, y).buffer(10000), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.polydf.crs)
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.polydf.crs)

        # TODO this appears to be necessary;
        # why is the sindex not generated automatically?
        #self.polydf2._generate_sindex()

        self.union_shape = union_qgis.shape

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, union_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/union_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/union_qgis.boundary.length).mean()==1)

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        df.sort_values(self.cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, intersect_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/intersect_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/intersect_qgis.boundary.length).mean()==1)

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        df.sort_values(self.cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, identity_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/identity_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/identity_qgis.boundary.length).mean()==1)

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        df.sort_values(self.cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, symdiff_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/symdiff_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/symdiff_qgis.boundary.length).mean()==1)

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        df.sort_values(self.cols[:-2], inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, diff_qgis.shape)
        self.assertTrue('value1' not in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/diff_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/diff_qgis.boundary.length).mean()==1)

    def test_bad_how(self):
        self.assertRaises(ValueError,
                          overlay, self.polydf, self.polydf, how="spandex")

    def test_nonpoly(self):
        self.assertRaises(TypeError,
                          overlay, self.pointdf, self.polydf, how="union")

    def test_duplicate_column_name(self):
        polydf2r = self.polydf2.rename(columns={'value2': 'Shape_Area'})
        df = overlay(self.polydf, polydf2r, how="union")
        self.assertTrue('Shape_Area_2' in df.columns and 'Shape_Area_1' in df.columns)

    def test_geometry_not_named_geometry(self):
        # Issue #306
        # Add points and flip names
        polydf3 = self.polydf.copy()
        polydf3 = polydf3.rename(columns={'geometry':'polygons'})
        polydf3 = polydf3.set_geometry('polygons')
        polydf3['geometry'] = self.pointdf.geometry.loc[0:4]
        self.assertTrue(polydf3.geometry.name == 'polygons')

        df = overlay(polydf3, self.polydf2, how="union")
        self.assertTrue(type(df) is GeoDataFrame)
        
        df2 = overlay(self.polydf, self.polydf2, how="union")
        self.assertTrue(df.geom_almost_equals(df2).all())

    def test_geoseries_warning(self):
        # Issue #305
        def f():
            overlay(self.polydf, self.polydf2.geometry, how="union")
        self.assertRaises(NotImplementedError, f)





