import json
import os
import pathlib

import numpy as np
from shapely import box

from geopandas import GeoDataFrame, GeoSeries

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa
from pyarrow import feather


DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / "data"


def assert_table_equal(left, right, check_metadata=False):
    if left.equals(right, check_metadata=check_metadata):
        return

    if not left.schema.equals(right.schema):
        raise AssertionError(
            "Schema not equal\nLeft:\n{0}\nRight:\n{1}".format(
                left.schema, right.schema
            )
        )

    if check_metadata:
        if not left.schema.equals(right.schema, check_metadata=True):
            if not left.schema.metadata == right.schema.metadata:
                raise AssertionError(
                    "Metadata not equal\nLeft:\n{0}\nRight:\n{1}".format(
                        left.schema.metadata, right.schema.metadata
                    )
                )
        # TODO also check field metadata
        for col in left.schema.names:
            assert left.schema.field(col).equals(
                right.schema.field(col), check_metadata=True
            )

    for col in left.column_names:
        a_left = pa.concat_arrays(left.column(col).chunks)
        a_right = pa.concat_arrays(right.column(col).chunks)
        if not a_left.equals(a_right):
            raise AssertionError(
                "Column '{0}' not equal:\n{1}".format(col, a_left.diff(a_right))
            )

    raise AssertionError("Tables not equal for unknown reason")


@pytest.mark.parametrize("dim", ["xy", "xyz"])
@pytest.mark.parametrize(
    "geometry_type",
    [
        pytest.param("point", marks=pytest.mark.xfail),
        "linestring",
        "polygon",
        "multipoint",
        "multilinestring",
        "multipolygon",
    ],
)
def test_geoarrow_export(geometry_type, dim):
    base_path = DATA_PATH / "geoarrow"
    suffix = geometry_type + ("_z" if dim == "xyz" else "")

    # Read the example data
    df = feather.read_feather(base_path / f"example-{suffix}-wkb.arrow")
    df["geometry"] = GeoSeries.from_wkb(df["geometry"])
    df["row_number"] = df["row_number"].astype("int32")
    df = GeoDataFrame(df)
    df.geometry.crs = None

    result1 = df.to_arrow(geometry_encoding="WKB")
    # TODO this still contains "geo" key in addition to pandas
    # remove the "pandas" metadata
    result1 = result1.replace_schema_metadata(None)
    expected1 = feather.read_table(base_path / f"example-{suffix}-wkb.arrow")

    if dim == "xyz" and geometry_type.startswith("multi"):
        # for collections with z dimension, drop the empties because those don't
        # roundtrip correctly to WKB
        # (https://github.com/libgeos/geos/issues/888)
        result1 = result1.filter(np.asarray(~df.geometry.is_empty))
        expected1 = expected1.filter(np.asarray(~df.geometry.is_empty))

    assert_table_equal(result1, expected1, check_metadata=True)

    result2 = df.to_arrow(geometry_encoding="geoarrow")
    # TODO this still contains "geo" key in addition to pandas
    # remove the "pandas" metadata
    result2 = result2.replace_schema_metadata(None)
    expected2 = feather.read_table(base_path / f"example-{suffix}-interleaved.arrow")

    assert_table_equal(result2, expected2, check_metadata=True)

    result3 = df.to_arrow(geometry_encoding="geoarrow", interleaved=False)
    # TODO this still contains "geo" key in addition to pandas
    # remove the "pandas" metadata
    result3 = result3.replace_schema_metadata(None)
    expected3 = feather.read_table(base_path / f"example-{suffix}.arrow")

    assert_table_equal(result3, expected3, check_metadata=True)


@pytest.mark.parametrize("encoding", ["WKB", "geoarrow"])
def test_geoarrow_multiple_geometry_crs(encoding):
    # ensure each geometry column has its own crs
    gdf = GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="epsg:4326")
    gdf["geom2"] = gdf.geometry.to_crs("epsg:3857")

    result = gdf.to_arrow(geometry_encoding=encoding)
    meta1 = json.loads(
        result.schema.field("geometry").metadata[b"ARROW:extension:metadata"]
    )
    assert json.loads(meta1["crs"])["id"]["code"] == 4326
    meta2 = json.loads(
        result.schema.field("geom2").metadata[b"ARROW:extension:metadata"]
    )
    assert json.loads(meta2["crs"])["id"]["code"] == 3857
