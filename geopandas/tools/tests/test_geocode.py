import numpy as np
from shapely.geometry import Point

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from geopandas.tools.geocoding import _prepare_geocode_result


def test_prepare_geocode_result_when_result_is_tuple_of_None():

    result = {0: (None, None)}
    expected_output = gpd.GeoDataFrame(
        {"geometry": [Point()], "address": [np.nan]}, crs="EPSG:4326",
    )

    output = _prepare_geocode_result(result)

    assert_geodataframe_equal(output, expected_output)


def test_prepare_geocode_result_when_result_is_None():

    result = {0: None}
    expected_output = gpd.GeoDataFrame(
        {"geometry": [Point()], "address": [np.nan]}, crs="EPSG:4326",
    )

    output = _prepare_geocode_result(result)

    assert_geodataframe_equal(output, expected_output)
