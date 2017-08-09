from __future__ import absolute_import

import tempfile
import shutil

from shapely.geometry import Point

import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas.tests.util import unittest
from geopandas import overlay


class TestDataFrame(unittest.TestCase):

    def setUp(self):
        N = 10

        nybb_filename = geopandas.datasets.get_path('nybb')

        self.polydf = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        b = [int(x) for x in self.polydf.total_bounds]
        self.polydf2 = GeoDataFrame([
            {'geometry' : Point(x, y).buffer(10000), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.crs)
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.crs)

        # TODO this appears to be necessary;
        # why is the sindex not generated automatically?
        self.polydf2._generate_sindex()

        self.union_shape = (180, 7)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, self.union_shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)

    def test_union_no_index(self):
        # explicitly ignore indicies
        dfB = overlay(self.polydf, self.polydf2, how="union", use_sindex=False)
        self.assertEquals(dfB.shape, self.union_shape)

        # remove indicies from df
        self.polydf._sindex = None
        self.polydf2._sindex = None
        dfC = overlay(self.polydf, self.polydf2, how="union")
        self.assertEquals(dfC.shape, self.union_shape)

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        self.assertIsNotNone(df['BoroName'][0])
        self.assertEquals(df.shape, (68, 7))

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        self.assertEquals(df.shape, (154, 7))

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        self.assertEquals(df.shape, (122, 7))

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        self.assertEquals(df.shape, (86, 7))

    def test_bad_how(self):
        self.assertRaises(ValueError,
                          overlay, self.polydf, self.polydf, how="spandex")

    def test_nonpoly(self):
        self.assertRaises(TypeError,
                          overlay, self.pointdf, self.polydf, how="union")

    def test_duplicate_column_name(self):
        polydf2r = self.polydf2.rename(columns={'value2': 'Shape_Area'})
        df = overlay(self.polydf, polydf2r, how="union")
        self.assertTrue('Shape_Area_2' in df.columns and 'Shape_Area' in df.columns)

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





