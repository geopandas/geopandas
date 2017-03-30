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
        self.assertEquals(df.area, union_qgis.area)
        self.assertEquals(df.boundary.length, union_qgis.boundary.length)

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, intersect_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEquals(df.area, intersect_qgis.area)
        self.assertEquals(df.boundary.length, intersect_qgis.boundary.length)

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, identity_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEquals(df.area, identity_qgis.area)
        self.assertEquals(df.boundary.length, identity_qgis.boundary.length)

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, symdiff_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEquals(df.area, symdiff_qgis.area)
        self.assertEquals(df.boundary.length, symdiff_qgis.boundary.length)

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, diff_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEquals(df.area, diff_qgis.area)
        self.assertEquals(df.boundary.length, diff_qgis.boundary.length)

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





