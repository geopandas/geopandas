import unittest
import json
import os
import tempfile
import shutil
import urllib2

import numpy as np
from shapely.geometry import Point

from geopandas import GeoDataFrame


class TestDataFrame(unittest.TestCase):

    def setUp(self):
        N = 10
        # Data from http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip
        # saved as geopandas/examples/nybb_13a.zip.
        if not os.path.exists(os.path.join('examples', 'nybb_13a.zip')):
            with open(os.path.join('examples', 'nybb_13a.zip'), 'w') as f:
                response = urllib2.urlopen('http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip')
                f.write(response.read())
        self.df = GeoDataFrame.from_file(
            '/nybb_13a/nybb.shp', vfs='zip://examples/nybb_13a.zip')
        self.tempdir = tempfile.mkdtemp()
        self.boros = np.array(['Staten Island', 'Queens', 'Brooklyn',
                               'Manhattan', 'Bronx'])
        self.crs = {'init': 'epsg:4326'}
        self.df2 = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x * y}
            for x, y in zip(range(N), range(N))], crs=self.crs)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        self.assertTrue(type(self.df2) is GeoDataFrame)
        self.assertTrue(self.df2.crs == self.crs)

    def test_from_file_(self):
        self.assertTrue('geometry' in self.df)
        self.assertTrue(len(self.df) == 5)
        self.assertTrue(np.alltrue(self.df['BoroName'].values == self.boros))

    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)

    def test_to_file(self):
        tempfilename = os.path.join(self.tempdir, 'boros.shp')
        self.df.to_file(tempfilename)
        # Read layer back in?
        df = GeoDataFrame.from_file(tempfilename)
        self.assertTrue('geometry' in df)
        self.assertTrue(len(df) == 5)
        self.assertTrue(np.alltrue(df['BoroName'].values == self.boros))
