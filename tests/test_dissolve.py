from __future__ import absolute_import
import tempfile
import shutil
from shapely.geometry import Point
from geopandas import GeoDataFrame, read_file
from geopandas.tools import overlay
from .util import unittest, download_nybb


class TestDataFrame(unittest.TestCase):

    def setUp(self):

        nybb_filename, nybb_zip_path = download_nybb()
        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)

        self.polydf = self.polydf.rename(columns={'geometry':'myshapes'})
        self.polydf = self.polydf.set_geometry('myshapes')

        self.polydf['manhattan_bronx'] = 0
        self.polydf.loc[3:4,'manhattan_bronx']=1

        # Merged geometry
        manhattan_bronx = self.polydf.loc[3:4,]
        others = self.polydf.loc[0:2,]
        

        self.merged_shapes = GeoDataFrame(columns=manhattan_bronx.columns)
        self.merged_shapes.loc[0, 'myshapes'] = others.geometry.unary_union
        self.merged_shapes.loc[1, 'myshapes'] = manhattan_bronx.geometry.unary_union
        self.merged_shapes = self.merged_shapes.set_geometry('myshapes')

    def test_geom_dissolve(self):
        test = self.polydf.dissolve('manhattan_bronx')
        self.assertTrue(test.geometry.name == 'myshapes')

        known = self.merged_shapes.geometry
        self.assertTrue(test.geometry.geom_almost_equals(known).all())


