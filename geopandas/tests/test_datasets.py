from __future__ import absolute_import

import pytest

from geopandas import read_file, GeoDataFrame
from geopandas.datasets import get_path

@pytest.mark.parametrize("test_dataset",
                         ['naturalearth_lowres',
                          'naturalearth_cities',
                          'nybb'])
def test_read_paths(test_dataset):
    assert isinstance(read_file(get_path(test_dataset)), GeoDataFrame)
