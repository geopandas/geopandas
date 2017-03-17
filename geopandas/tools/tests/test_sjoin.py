from __future__ import absolute_import

import tempfile
import shutil

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, GeoSeries, read_file, base
from geopandas.tests.util import unittest, download_nybb
from geopandas import sjoin
from distutils.version import LooseVersion

pandas_0_16_problem = 'fails under pandas < 0.17 due to issue 251,'\
                      'not problem with sjoin.'


@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestSpatialJoin(unittest.TestCase):

    def setUp(self):

        polys1 = GeoSeries(
            [Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
             Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
             Polygon([(6, 0), (9, 0), (9, 3), (6, 3)])])

        polys2 = GeoSeries(
            [Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),
             Polygon([(4, 4), (7, 4), (7, 7), (4, 7)]),
             Polygon([(7, 7), (10, 7), (10, 10), (7, 10)])])

        self.df1 = GeoDataFrame({'geometry': polys1, 'df1': [0, 1, 2]})
        self.df2 = GeoDataFrame({'geometry': polys2, 'df2': [3, 4, 5]})

        # construction expected frames
        self.expected = {}

        part1 = self.df1.copy().reset_index().rename(columns={'index': 'index_left'})
        part2 = self.df2.copy().reindex([0, 1, 1, 2]).reset_index().rename(
            columns={'index': 'index_right'})
        part1['_merge'] = [0, 1, 2]
        part2['_merge'] = [0, 0, 1, 3]
        expected = pd.merge(part1, part2, on='_merge', how='outer')
        self.expected['intersects'] = expected.drop('_merge', axis=1).copy()

        part1 = self.df1.copy().reset_index().rename(columns={'index': 'index_left'})
        part2 = self.df2.copy().reset_index().rename(columns={'index': 'index_right'})
        part1['_merge'] = [0, 1, 2]
        part2['_merge'] = [0, 3, 3]
        expected = pd.merge(part1, part2, on='_merge', how='outer')
        self.expected['contains'] = expected.drop('_merge', axis=1).copy()

        part1['_merge'] = [0, 1, 2]
        part2['_merge'] = [3, 1, 3]
        expected = pd.merge(part1, part2, on='_merge', how='outer')
        self.expected['within'] = expected.drop('_merge', axis=1).copy()

    def test_inner(self):
        for op in ['intersects', 'contains', 'within']:
            res = sjoin(self.df1, self.df2, how='inner', op=op)
            exp = self.expected[op].dropna().copy()
            exp = exp.drop('geometry_y', axis=1).rename(columns={'geometry_x': 'geometry'})
            exp[['index_left', 'index_right', 'df1', 'df2']] = \
                exp[['index_left', 'index_right', 'df1', 'df2']].astype('int64')
            exp = exp.set_index('index_left')
            exp.index.name = None
            assert_frame_equal(res, exp)

    def test_left(self):
        for op in ['intersects', 'contains', 'within']:
            res = sjoin(self.df1, self.df2, how='left', op=op)
            exp = self.expected[op].dropna(subset=['index_left']).copy()
            exp = exp.drop('geometry_y', axis=1).rename(
                columns={'geometry_x': 'geometry'})
            exp[['index_left', 'df1']] = \
                exp[['index_left', 'df1']].astype('int64')
            # TODO: in result the dtype is object
            res['index_right'] = res['index_right'].astype(float)
            exp = exp.set_index('index_left')
            exp.index.name = None
            assert_frame_equal(res, exp)

    def test_right(self):
        for op in ['intersects', 'contains', 'within']:
            res = sjoin(self.df1, self.df2, how='right', op=op)
            exp = self.expected[op].dropna(subset=['index_right']).copy()
            exp = exp.drop('geometry_x', axis=1).rename(
                columns={'geometry_y': 'geometry'})
            exp[['index_right', 'df2']] = exp[['index_right', 'df2']].astype('int64')
            res['index_left'] = res['index_left'].astype(float)
            exp = exp.set_index('index_right')
            exp = exp.reindex(columns=res.columns)
            assert_frame_equal(res, exp, check_index_type=False)


@unittest.skipIf(not base.HAS_SINDEX, 'Rtree absent, skipping')
class TestSpatialJoinNYBB(unittest.TestCase):

    def setUp(self):
        nybb_filename, nybb_zip_path = download_nybb()
        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = self.polydf.crs
        N = 20
        b = [int(x) for x in self.polydf.total_bounds]
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'pointattr1': x + y, 'pointattr2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.crs)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_geometry_name(self):
        # test sjoin is working with other geometry name
        polydf_original_geom_name = self.polydf.geometry.name
        self.polydf = (self.polydf.rename(columns={'geometry': 'new_geom'})
                                  .set_geometry('new_geom'))
        self.assertNotEqual(polydf_original_geom_name, self.polydf.geometry.name)
        res = sjoin(self.polydf, self.pointdf, how="left")
        self.assertEqual(self.polydf.geometry.name, res.geometry.name)

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

    @unittest.skipIf(str(pd.__version__) < LooseVersion('0.17'), pandas_0_16_problem)
    def test_no_overlapping_geometry(self):
        # Note: these tests are for correctly returning GeoDataFrame
        # when result of the join is empty

        df_inner = sjoin(self.pointdf.iloc[17:], self.polydf, how='inner')
        df_left = sjoin(self.pointdf.iloc[17:], self.polydf, how='left')
        df_right = sjoin(self.pointdf.iloc[17:], self.polydf, how='right')

        # Recent Pandas development has introduced a new way of handling merges
        # this change has altered the output when no overlapping geometries
        if str(pd.__version__) > LooseVersion('0.18.1'):
            right_idxs = pd.Series(range(0,5), name='index_right',dtype='int64')
        else:
            right_idxs = pd.Series(name='index_right',dtype='int64')

        expected_inner_df = pd.concat([self.pointdf.iloc[:0],
                                       pd.Series(name='index_right', dtype='int64'),
                                       self.polydf.drop('geometry', axis = 1).iloc[:0]], axis = 1)

        expected_inner = GeoDataFrame(expected_inner_df, crs = {'init': 'epsg:4326', 'no_defs': True})

        expected_right_df = pd.concat([self.pointdf.drop('geometry', axis = 1).iloc[:0],
                                       pd.concat([pd.Series(name='index_left',dtype='int64'), right_idxs], axis=1),
                                       self.polydf], axis = 1)

        expected_right = GeoDataFrame(expected_right_df, crs = {'init': 'epsg:4326', 'no_defs': True})\
                            .set_index('index_right')

        expected_left_df = pd.concat([self.pointdf.iloc[17:],
                                      pd.Series(name='index_right', dtype='int64'),
                                      self.polydf.iloc[:0].drop('geometry', axis=1)], axis = 1)

        expected_left = GeoDataFrame(expected_left_df, crs = {'init': 'epsg:4326', 'no_defs': True})

        self.assertTrue(expected_inner.equals(df_inner))
        self.assertTrue(expected_right.equals(df_right))
        self.assertTrue(expected_left.equals(df_left))

    @unittest.skip("Not implemented")
    def test_sjoin_outer(self):
        df = sjoin(self.pointdf, self.polydf, how="outer")
        self.assertEquals(df.shape, (21,8))
