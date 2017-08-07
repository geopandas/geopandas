from __future__ import absolute_import
import tempfile
import shutil
import numpy as np
from shapely.geometry import Point
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas.tools import overlay
from .util import unittest
from pandas.util.testing import assert_frame_equal
from pandas import Index
import pandas as pd


class TestDataFrame(unittest.TestCase):

    def setUp(self):

        nybb_filename = geopandas.datasets.get_path('nybb')
        self.polydf = read_file(nybb_filename)
        self.polydf = self.polydf[['geometry', 'BoroName', 'BoroCode']]

        self.polydf = self.polydf.rename(columns={'geometry':'myshapes'})
        self.polydf = self.polydf.set_geometry('myshapes')

        self.polydf['manhattan_bronx'] = 5
        self.polydf.loc[3:4,'manhattan_bronx']=6

        # Merged geometry
        manhattan_bronx = self.polydf.loc[3:4,]
        others = self.polydf.loc[0:2,]

        collapsed = [others.geometry.unary_union, manhattan_bronx.geometry.unary_union]
        merged_shapes = GeoDataFrame({'myshapes': collapsed}, geometry='myshapes',
                             index=Index([5,6], name='manhattan_bronx'))

        # Different expected results
        self.first = merged_shapes.copy()
        self.first['BoroName'] = ['Staten Island', 'Manhattan']
        self.first['BoroCode'] = [5, 1]

        self.mean = merged_shapes.copy()
        self.mean['BoroCode'] = [4,1.5]

    def test_geom_dissolve(self):
        test = self.polydf.dissolve('manhattan_bronx')
        self.assertTrue(test.geometry.name == 'myshapes')
        self.assertTrue(test.geom_almost_equals(self.first).all())

    def test_dissolve_retains_existing_crs(self):
        assert self.polydf.crs is not None
        test = self.polydf.dissolve('manhattan_bronx')
        assert test.crs is not None

    def test_dissolve_retains_nonexisting_crs(self):
        self.polydf.crs = None
        test = self.polydf.dissolve('manhattan_bronx')
        assert test.crs is None

    def test_first_dissolve(self):
        test = self.polydf.dissolve('manhattan_bronx')
        assert_frame_equal(self.first, test, check_column_type=False)

    def test_mean_dissolve(self):
        test = self.polydf.dissolve('manhattan_bronx', aggfunc='mean')
        assert_frame_equal(self.mean, test, check_column_type=False)

        test = self.polydf.dissolve('manhattan_bronx', aggfunc=np.mean)
        assert_frame_equal(self.mean, test, check_column_type=False)

    def test_multicolumn_dissolve(self):
        multi = self.polydf.copy()
        multi['dup_col'] = multi.manhattan_bronx
        multi_test = multi.dissolve(['manhattan_bronx', 'dup_col'], aggfunc='first')

        first = self.first.copy()
        first['dup_col'] = first.index
        first = first.set_index([first.index, 'dup_col'])

        assert_frame_equal(multi_test, first, check_column_type=False)

    def test_reset_index(self):
        test = self.polydf.dissolve('manhattan_bronx', as_index=False)
        comparison = self.first.reset_index()
        assert_frame_equal(comparison, test, check_column_type=False)
