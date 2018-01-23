from __future__ import absolute_import

import pytest

from geopandas import read_file, GeoDataFrame
from geopandas.datasets import get_path

@pytest.mark.parametrize("test_gdf",
                         [read_file(get_path('naturalearth_lowres')),
                          read_file(get_path('naturalearth_cities')),
                          read_file(get_path('nybb'))])
def test_read_paths(test_gdf):
    assert isinstance(test_gdf, GeoDataFrame)
