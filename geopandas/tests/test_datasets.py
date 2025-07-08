from geopandas import GeoDataFrame, read_file
from geopandas.datasets import get_path

import pytest


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb", "foo"]
)
def test_read_paths(test_dataset):
    with pytest.raises(
        AttributeError,
        match=r"The geopandas\.dataset has been deprecated and was removed",
    ):
        assert isinstance(read_file(get_path(test_dataset)), GeoDataFrame)
