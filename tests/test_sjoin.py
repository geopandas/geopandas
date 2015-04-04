from __future__ import absolute_import
import tempfile
import shutil
from shapely.geometry import Point
from geopandas import GeoDataFrame, read_file
from geopandas.tools import sjoin
from .util import unittest, download_nybb


class TestSpatialJoin(unittest.TestCase):

    def setUp(self):
        nybb_filename = download_nybb()
        self.polydf = read_file('/nybb_14a_av/nybb.shp', vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        N = 20 
        b = [int(x) for x in self.polydf.total_bounds]
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'pointattr1': x + y, 'pointattr2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.crs)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_sjoin_left(self):
        df = sjoin(self.pointdf, self.polydf, crs_convert=False)
        self.assertEquals(df.shape, (11,9))
#        for i, row in df.iterrows():
#            self.assertEquals(row.geometry.type, 'Point')
        self.assertTrue('pointattr1' in df.columns)
        self.assertTrue('BoroCode' in df.columns)

    def test_sjoin_right(self):
        # the inverse of left
        df = sjoin(self.pointdf, self.polydf, how="right", crs_convert=False)
        df2 = sjoin(self.polydf, self.pointdf, how="left", crs_convert=False)
        self.assertEquals(df.shape, (12, 9))
#        self.assertEquals(df.shape, df2.shape)
#        for i, row in df.iterrows():
#            self.assertEquals(row.geometry.type, 'MultiPolygon')
#        for i, row in df2.iterrows():
#            self.assertEquals(row.geometry.type, 'MultiPolygon')

    def test_sjoin_inner(self):
        df = sjoin(self.pointdf, self.polydf, how="inner", crs_convert=False)
        self.assertEquals(df.shape, (11, 9))

    def test_sjoin_op(self):
        # points within polygons
        df = sjoin(self.pointdf, self.polydf, how="left", op="within", crs_convert=False)
        self.assertEquals(df.shape, (11,9))
        self.assertAlmostEquals(df.ix[1]['Shape_Leng'], 330454.175933)

        # points contain polygons? never happens so we should have nulls
#        df = sjoin(self.pointdf, self.polydf, how="left", op="contains", crs_convert=False)
#        self.assertEquals(df.shape, (11, 9))
#        self.assertEquals(df.ix[1]['Shape_Area'], None)

    def test_sjoin_bad_op(self, crs_convert=False):
        # AttributeError: 'Point' object has no attribute 'spandex'
        self.assertRaises(ValueError, sjoin,
            self.pointdf, self.polydf, how="left", op="spandex")

    @unittest.skip("Not implemented")
    def test_sjoin_duplicate_column_name(self, crs_convert=False):
        pointdf2 = self.pointdf.rename(columns={'pointattr1': 'Shape_Area'})
        df = sjoin(pointdf2, self.polydf, how="left", crs_convert=False)
        self.assertTrue('Shape_Area' in df.columns)
        self.assertTrue('Shape_Area_2' in df.columns)

    @unittest.skip("Not implemented")
    def test_sjoin_outer(self):
        df = sjoin(self.pointdf, self.polydf, how="outer")
        self.assertEquals(df.shape, (21,9))
