from __future__ import absolute_import

from geopandas import GeoDataFrame, read_file
from geopandas.datasets import get_path

import pytest


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_read_paths(test_dataset):
    assert isinstance(read_file(get_path(test_dataset)), GeoDataFrame)
