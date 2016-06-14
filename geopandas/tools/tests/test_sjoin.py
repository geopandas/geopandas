from __future__ import absolute_import

import tempfile
import shutil

import numpy as np
from shapely.geometry import Point

from geopandas import GeoDataFrame, read_file, base
from geopandas.tests.util import unittest, download_nybb
from geopandas import sjoin


@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestSpatialJoin(unittest.TestCase):

    def setUp(self):
        nybb_filename, nybb_zip_path = download_nybb()
        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)
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
        df = sjoin(self.pointdf, self.polydf, how='left')
        self.assertEquals(df.shape, (21,8))
        for i, row in df.iterrows():
            self.assertEquals(row.geometry.type, 'Point')
        self.assertTrue('pointattr1' in df.columns)
        self.assertTrue('BoroCode' in df.columns)

    def test_sjoin_right(self):
        # the inverse of left
        df = sjoin(self.pointdf, self.polydf, how="right")
        df2 = sjoin(self.polydf, self.pointdf, how="left")
        self.assertEquals(df.shape, (12, 8))
        self.assertEquals(df.shape, df2.shape)
        for i, row in df.iterrows():
            self.assertEquals(row.geometry.type, 'MultiPolygon')
        for i, row in df2.iterrows():
            self.assertEquals(row.geometry.type, 'MultiPolygon')

    def test_sjoin_inner(self):
        df = sjoin(self.pointdf, self.polydf, how="inner")
        self.assertEquals(df.shape, (11, 8))

    def test_sjoin_op(self):
        # points within polygons
        df = sjoin(self.pointdf, self.polydf, how="left", op="within")
        self.assertEquals(df.shape, (21,8))
        self.assertEquals(df.ix[1]['BoroName'], 'Staten Island')

        # points contain polygons? never happens so we should have nulls
        df = sjoin(self.pointdf, self.polydf, how="left", op="contains")
        self.assertEquals(df.shape, (21, 8))
        self.assertTrue(np.isnan(df.ix[1]['Shape_Area']))

    def test_sjoin_bad_op(self):
        # AttributeError: 'Point' object has no attribute 'spandex'
        self.assertRaises(ValueError, sjoin,
            self.pointdf, self.polydf, how="left", op="spandex")

    def test_sjoin_duplicate_column_name(self):
        pointdf2 = self.pointdf.rename(columns={'pointattr1': 'Shape_Area'})
        df = sjoin(pointdf2, self.polydf, how="left")
        self.assertTrue('Shape_Area_left' in df.columns)
        self.assertTrue('Shape_Area_right' in df.columns)

    def test_sjoin_values(self):
        # GH190
        self.polydf.index = [1, 3, 4, 5, 6]
        df = sjoin(self.pointdf, self.polydf, how='left')
        self.assertEquals(df.shape, (21,8))
        df = sjoin(self.polydf, self.pointdf, how='left')
        self.assertEquals(df.shape, (12,8))

    def test_no_overlapping_geometry(self):
        # Note: these tests are for correctly returning GeoDataFrame
        # when result of the join is empty

        subset_pointdf = self.pointdf.iloc[17:]

        df_inner = sjoin(subset_pointdf, self.polydf, how='inner')
        self.assertEquals(df_inner.columns.tolist()[:5], ['geometry', 'pointattr1', 'pointattr2','index_right','BoroCode'])
        self.assertEquals(df_inner.shape, (0,8))

        df_left = sjoin(subset_pointdf, self.polydf, how='left')
        self.assertEquals(df_left.shape, (4,8))
        self.assertTrue(df_left.BoroCode.notnull().sum()==0)
        self.assertTrue(df_left.pointattr2.isnull().sum()==0)

        df_right = sjoin(subset_pointdf, self.polydf, how='right')
        self.assertEquals(df_right.shape, (5,8))
        self.assertTrue(df_right.pointattr2.notnull().sum()==0)
        self.assertTrue(df_right.BoroCode.isnull().sum()==0)

    @unittest.skip("Not implemented")
    def test_sjoin_outer(self):
        df = sjoin(self.pointdf, self.polydf, how="outer")
        self.assertEquals(df.shape, (21,8))
