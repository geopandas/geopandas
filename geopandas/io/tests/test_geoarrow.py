import os
import pathlib

from geopandas import GeoDataFrame, GeoSeries

import pytest
from geopandas.testing import assert_geodataframe_equal

pytest.importorskip("pyarrow")
from pyarrow import feather

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / "data"


@pytest.mark.parametrize("dim", ["xy", "xyz"])
@pytest.mark.parametrize(
    "geometry_type",
    [
        "point",
        "linestring",
        "polygon",
        "multipoint",
        "multilinestring",
        "multipolygon",
    ],
)
def test_geoarrow_import(geometry_type, dim):
    base_path = DATA_PATH / "geoarrow"
    suffix = geometry_type + ("_z" if dim == "xyz" else "")

    # Read the example data
    df = feather.read_feather(base_path / f"example-{suffix}-wkb.arrow")
    df["geometry"] = GeoSeries.from_wkb(df["geometry"])
    df = GeoDataFrame(df)
    df.geometry.crs = None

    table1 = feather.read_table(base_path / f"example-{suffix}-wkb.arrow")
    result1 = GeoDataFrame.from_arrow(table1)
    assert_geodataframe_equal(result1, df)

    table2 = feather.read_table(base_path / f"example-{suffix}-interleaved.arrow")
    result2 = GeoDataFrame.from_arrow(table2)
    assert_geodataframe_equal(result2, df)

    table3 = feather.read_table(base_path / f"example-{suffix}.arrow")
    result3 = GeoDataFrame.from_arrow(table3)
    assert_geodataframe_equal(result3, df)


# def test_geoarrow_unsupported_encoding():
#     gdf = GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="epsg:4326")

#     with pytest.raises(ValueError, match="Expected geometry encoding"):
#         gdf.to_arrow(geometry_encoding="invalid")
