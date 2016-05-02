from __future__ import absolute_import
import tempfile
import shutil
from shapely.geometry import Point
from geopandas import GeoDataFrame, read_file
from geopandas.tools import overlay
from .util import unittest, download_nybb
from pandas.util.testing import assert_frame_equal
from pandas import Index

class TestDataFrame(unittest.TestCase):

    def setUp(self):

        nybb_filename, nybb_zip_path = download_nybb()
        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)
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

    def test_first_dissolve(self):
        test = self.polydf.dissolve('manhattan_bronx')
        test = test.drop('myshapes', axis=1)
        first = self.first.drop('myshapes', axis=1)
        assert_frame_equal(first, test)

