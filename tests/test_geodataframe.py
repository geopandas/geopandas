import unittest
import json
import numpy as np

from geopandas import GeoDataFrame

class TestDataFrame(unittest.TestCase):

    def setUp(self):
        # Data from http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip
        # saved as geopandas/examples/nybb_13a.zip.
        self.df = GeoDataFrame.from_file(
            '/nybb_13a/nybb.shp', vfs='zip://examples/nybb_13a.zip')

    def test_from_file_(self):
        self.assertTrue('geometry' in self.df)
        self.assertTrue(len(self.df) == 5)
        self.assertTrue(np.alltrue(self.df['BoroName'].values == np.array(['Staten Island',
                                   'Queens', 'Brooklyn', 'Manhattan', 'Bronx'])))
    
    def test_to_json(self):
        text = self.df.to_json()
        data = json.loads(text)
        self.assertTrue(data['type'] == 'FeatureCollection')
        self.assertTrue(len(data['features']) == 5)
