import unittest
import json
import numpy as np

import tempfile

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
        
    def test_to_file(self):
        with tempfile.NamedTemporaryFile(suffix='.shp') as t:
            self.df.to_file(t.name)
            # Read layer back in?
            df = GeoDataFrame.from_file(t.name)
            self.assertTrue('geometry' in df)
            self.assertTrue(len(df) == 5)
            self.assertTrue(np.alltrue(df['BoroName'].values == np.array(['Staten Island',
                                   'Queens', 'Brooklyn', 'Manhattan', 'Bronx'])))
