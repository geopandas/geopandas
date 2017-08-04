from __future__ import absolute_import

from geopandas import read_file, GeoDataFrame
from geopandas.datasets import get_path
from geopandas.tests.util import unittest


class TestDatasets(unittest.TestCase):

    def test_read_paths(self):

        gdf = read_file(get_path('naturalearth_lowres'))
        assert isinstance(gdf, GeoDataFrame)

        gdf = read_file(get_path('naturalearth_cities'))
        assert isinstance(gdf, GeoDataFrame)

        gdf = read_file(get_path('nybb'))
        assert isinstance(gdf, GeoDataFrame)
