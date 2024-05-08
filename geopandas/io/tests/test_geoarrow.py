import io
import urllib

import pytest

import pyarrow as pa
from pyarrow import feather
import geopandas


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
def test_geoarrow_export(geometry_type):
    base_url = (
        "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.1.0/example/"
    )

    # Read the example data
    df = geopandas.read_file(base_url + f"example-{geometry_type}.gpkg")
    df["row_number"] = df["row_number"].astype("int32")
    df.geometry.crs = None

    result1 = df.to_arrow(geometry_encoding="WKB")
    # TODO this still contains "geo" key in addition to pandas
    # remove the "pandas" metadata
    result1 = result1.replace_schema_metadata(None)
    with urllib.request.urlopen(base_url + f"example-{geometry_type}-wkb.arrow") as req:
        expected1 = feather.read_table(io.BytesIO(req.read()))

    assert_table_equal(result1, expected1, check_metadata=True)

    result2 = df.to_arrow(geometry_encoding="geoarrow")
    # TODO this still contains "geo" key in addition to pandas
    # remove the "pandas" metadata
    result2 = result2.replace_schema_metadata(None)
    with urllib.request.urlopen(
        base_url + f"example-{geometry_type}-interleaved.arrow"
    ) as req:
        expected2 = feather.read_table(io.BytesIO(req.read()))

    assert_table_equal(result2, expected2, check_metadata=True)

    result3 = df.to_arrow(geometry_encoding="geoarrow", interleaved=False)
    # TODO this still contains "geo" key in addition to pandas
    # remove the "pandas" metadata
    result3 = result3.replace_schema_metadata(None)
    with urllib.request.urlopen(base_url + f"example-{geometry_type}.arrow") as req:
        expected3 = feather.read_table(io.BytesIO(req.read()))

    assert_table_equal(result3, expected3, check_metadata=True)
