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

    @classmethod
    def setUpClass(cls):
        _skip_if_no_geopy()
        cls.locations = ['260 Broadway, New York, NY',
                          '77 Massachusetts Ave, Cambridge, MA']
        cls.points = [Point(-71.0597732, 42.3584308),
                       Point(-77.0365305, 38.8977332)]

    def test_prepare_result(self):
        """
        _prepare_geocode_result with data for every item
        """
        from geopy.location import Location

        # lon, lat, alt
        p0_coords = (-45.6, 12.3, 0.1)
        p1_coords = (56.7, -23.4, 0.2)

        d = {
            'a': Location(
                'address0',
                (p0_coords[1], p0_coords[0], p0_coords[2])
            ),
            'b': Location(
                'address1',
                (p1_coords[1], p1_coords[0], p1_coords[2])
            ),
        }

        df = _prepare_geocode_result(d)
        self.assertTrue(type(df) is gpd.GeoDataFrame)
        self.assertEqual(from_epsg(4326), df.crs)
        self.assertEqual(len(df), 2)
        self.assert_('address' in df)

        coords = df.loc['a']['geometry'].coords[0]

        test = p0_coords
        # Output from the df should be lon/lat
        self.assertAlmostEqual(coords[0], test[0])
        self.assertAlmostEqual(coords[1], test[1])

        coords = df.loc['b']['geometry'].coords[0]
        test = p1_coords
        self.assertAlmostEqual(coords[0], test[0])
        self.assertAlmostEqual(coords[1], test[1])

    def test_prepare_result_none(self):
        """
        _prepare_geocode_result when some items are null
        """
        from geopy.location import Location

        # lon, lat, alt
        p0_coords = (-45.6, 12.3, 0.1)

        d = {
            'a': Location(
                'address0',
                (p0_coords[1], p0_coords[0], p0_coords[2])
            ),
            'b': None
        }

        df = _prepare_geocode_result(d)
        self.assertTrue(type(df) is gpd.GeoDataFrame)
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
