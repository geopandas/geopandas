import unittest

import fiona
from shapely.geometry import Point
import geopandas as gpd

from geopandas.geocode import geocode, _prepare_geocode_result

class TestGeocode(unittest.TestCase):
    def test_prepare_result(self):
        # Calls _prepare_result with sample results from the geocoder call
        # loop
        p0 = Point(12.3, -45.6) # Treat these as lat/lon
        p1 = Point(-23.4, 56.7)
        d = {'a': ('address0', p0.coords[0]),
             'b': ('address1', p1.coords[0])}

        df = _prepare_geocode_result(d)
        assert type(df) is gpd.GeoDataFrame
        self.assertEqual(fiona.crs.from_epsg(4326), df.crs)
        self.assertEqual(len(df), 2)
        self.assertIn('address', df)

        coords = df.loc['a']['geometry'].coords[0]
        test = p0.coords[0]
        # Output from the df should be lon/lat
        self.assertAlmostEqual(coords[0], test[1])
        self.assertAlmostEqual(coords[1], test[0])

        coords = df.loc['b']['geometry'].coords[0]
        test = p1.coords[0]
        self.assertAlmostEqual(coords[0], test[1])
        self.assertAlmostEqual(coords[1], test[0])

    def test_bad_provider(self):
        self.assertRaises(ValueError, geocode, ['cambridge, ma'], 'badprovider')
