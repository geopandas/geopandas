from __future__ import absolute_import

import sys

from fiona.crs import from_epsg
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import nose

from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result

from .util import unittest


def _skip_if_no_geopy():
    try:
        import geopy
    except ImportError:
        raise nose.SkipTest("Geopy not installed. Skipping tests.")
    except SyntaxError:
        raise nose.SkipTest("Geopy is known to be broken on Python 3.2. "
                            "Skipping tests.")

class TestGeocode(unittest.TestCase):
    def setUp(self):
        _skip_if_no_geopy()
        self.locations = ['260 Broadway, New York, NY',
                          '77 Massachusetts Ave, Cambridge, MA']
        self.points = [Point(-71.0597732, 42.3584308),
                       Point(-77.0365305, 38.8977332)]

    def test_prepare_result(self):
        # Calls _prepare_result with sample results from the geocoder call
        # loop
        from geopandas.tools.geocoding import _prepare_geocode_result
        p0 = Point(12.3, -45.6) # Treat these as lat/lon
        p1 = Point(-23.4, 56.7)
        d = {'a': ('address0', p0.coords[0]),
             'b': ('address1', p1.coords[0])}

        df = _prepare_geocode_result(d)
        assert type(df) is gpd.GeoDataFrame
        self.assertEqual(from_epsg(4326), df.crs)
        self.assertEqual(len(df), 2)
        self.assert_('address' in df)

        coords = df.loc['a']['geometry'].coords[0]
        test = p0.coords[0]
        # Output from the df should be lon/lat
        self.assertAlmostEqual(coords[0], test[1])
        self.assertAlmostEqual(coords[1], test[0])

        coords = df.loc['b']['geometry'].coords[0]
        test = p1.coords[0]
        self.assertAlmostEqual(coords[0], test[1])
        self.assertAlmostEqual(coords[1], test[0])

    def test_prepare_result_none(self):
        from geopandas.tools.geocoding import _prepare_geocode_result
        p0 = Point(12.3, -45.6) # Treat these as lat/lon
        d = {'a': ('address0', p0.coords[0]),
             'b': (None, None)}

        df = _prepare_geocode_result(d)
        assert type(df) is gpd.GeoDataFrame
        self.assertEqual(from_epsg(4326), df.crs)
        self.assertEqual(len(df), 2)
        self.assert_('address' in df)

        row = df.loc['b']
        self.assertEqual(len(row['geometry'].coords), 0)
        self.assert_(pd.np.isnan(row['address']))
    
    def test_bad_provider_forward(self):
        with self.assertRaises(ValueError):
            geocode(['cambridge, ma'], 'badprovider')

    def test_bad_provider_reverse(self):
        with self.assertRaises(ValueError):
            reverse_geocode(['cambridge, ma'], 'badprovider')

    def test_googlev3_forward(self):
        g = geocode(self.locations, provider='googlev3', timeout=2)
        self.assertIsInstance(g, gpd.GeoDataFrame)

    def test_googlev3_reverse(self):
        g = reverse_geocode(self.points, provider='googlev3', timeout=2)
        self.assertIsInstance(g, gpd.GeoDataFrame)

    def test_openmapquest_forward(self):
        g = geocode(self.locations, provider='openmapquest', timeout=2)
        self.assertIsInstance(g, gpd.GeoDataFrame)

    # openmapquest does not have reverse implemented in geopy

    @unittest.skip('Nominatim server is unreliable for tests.')
    def test_nominatim(self):
        g = geocode(self.locations, provider='nominatim', timeout=2)
        self.assertIsInstance(g, gpd.GeoDataFrame)
