import pytest

import geopandas as gpd


@pytest.fixture
def gdf():
    return gpd.GeoDataFrame({"a": [1, 2]})


def test_getitem_access(gdf):
    with pytest.warns(FutureWarning, match="You are adding a column named"):
        gdf["geometry"] = gpd.GeoSeries.from_xy([1, 3], [3, 3])
    assert gdf._geometry_column_name == "geometry"


def test_getitem_access2(gdf):
    with pytest.warns(FutureWarning, match="You are adding a column named"):
        gdf.geometry = gpd.GeoSeries.from_xy([1, 3], [3, 3])
    assert gdf._geometry_column_name == "geometry"


def test_getitem_access3(gdf):
    with pytest.warns(UserWarning, match="Geometry column does not contain geometry"):
        gdf["geometry"] = "foo"
    assert gdf._geometry_column_name is gpd.geodataframe.DEFAULT_GEO_COLUMN_NAME


# def test_getattr_crs(gdf):
#
#     assert gdf.crs is None
