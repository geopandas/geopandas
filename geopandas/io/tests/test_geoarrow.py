import contextlib
import json
import os
import pathlib
from packaging.version import Version

import numpy as np

import shapely
from shapely import MultiPoint, Point, box

from geopandas import GeoDataFrame, GeoSeries

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa
from pyarrow import feather

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / "data"


def pa_table(table):
    if Version(pa.__version__) < Version("14.0.0"):
        return table._pa_table
    else:
        return pa.table(table)


def pa_array(array):
    if Version(pa.__version__) < Version("14.0.0"):
        return array._pa_array
    else:
        return pa.array(array)


def assert_table_equal(left, right, check_metadata=True):
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


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="Checking for empty is buggy with GEOS<3.9",
)  # an old GEOS is installed in the CI builds with the defaults channel
@pytest.mark.parametrize(
    "dim",
    [
        "xy",
        pytest.param(
            "xyz",
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0),
                reason="Cannot write 3D geometries with GEOS<3.10",
            ),
        ),
    ],
)
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
@pytest.mark.parametrize(
    "geometry_encoding, interleaved",
    [("WKB", None), ("geoarrow", True), ("geoarrow", False)],
    ids=["WKB", "geoarrow-interleaved", "geoarrow-separated"],
)
def test_geoarrow_export(geometry_type, dim, geometry_encoding, interleaved):
    base_path = DATA_PATH / "geoarrow"
    suffix = geometry_type + ("_z" if dim == "xyz" else "")

    # Read the example data
    df = feather.read_feather(base_path / f"example-{suffix}-wkb.arrow")
    df["geometry"] = GeoSeries.from_wkb(df["geometry"])
    df["row_number"] = df["row_number"].astype("int32")
    df = GeoDataFrame(df)
    df.geometry.crs = None

    # Read the expected data
    if geometry_encoding == "WKB":
        filename = f"example-{suffix}-wkb.arrow"
    else:
        filename = f"example-{suffix}{'-interleaved' if interleaved else ''}.arrow"
    expected = feather.read_table(base_path / filename)

    # GeoDataFrame -> Arrow Table
    result = pa_table(
        df.to_arrow(geometry_encoding=geometry_encoding, interleaved=interleaved)
    )
    # remove the "pandas" metadata
    result = result.replace_schema_metadata(None)

    mask_nonempty = None
    if (
        geometry_encoding == "WKB"
        and dim == "xyz"
        and geometry_type.startswith("multi")
    ):
        # for collections with z dimension, drop the empties because those don't
        # roundtrip correctly to WKB
        # (https://github.com/libgeos/geos/issues/888)
        mask_nonempty = pa.array(np.asarray(~df.geometry.is_empty))
        result = result.filter(mask_nonempty)
        expected = expected.filter(mask_nonempty)

    assert_table_equal(result, expected)

    # GeoSeries -> Arrow array
    result_arr = pa_array(
        df.geometry.to_arrow(
            geometry_encoding=geometry_encoding, interleaved=interleaved
        )
    )
    if mask_nonempty is not None:
        result_arr = result_arr.filter(mask_nonempty)
    assert result_arr.equals(expected["geometry"].chunk(0))


@pytest.mark.parametrize("encoding", ["WKB", "geoarrow"])
def test_geoarrow_multiple_geometry_crs(encoding):
    pytest.importorskip("pyproj")
    # ensure each geometry column has its own crs
    gdf = GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="epsg:4326")
    gdf["geom2"] = gdf.geometry.to_crs("epsg:3857")

    result = pa_table(gdf.to_arrow(geometry_encoding=encoding))
    meta1 = json.loads(
        result.schema.field("geometry").metadata[b"ARROW:extension:metadata"]
    )
    assert json.loads(meta1["crs"])["id"]["code"] == 4326
    meta2 = json.loads(
        result.schema.field("geom2").metadata[b"ARROW:extension:metadata"]
    )
    assert json.loads(meta2["crs"])["id"]["code"] == 3857


@pytest.mark.parametrize("encoding", ["WKB", "geoarrow"])
def test_geoarrow_series_name_crs(encoding):
    pytest.importorskip("pyproj")
    pytest.importorskip("pyarrow", minversion="14.0.0")

    gser = GeoSeries([box(0, 0, 10, 10)], crs="epsg:4326", name="geom")
    schema_capsule, _ = gser.to_arrow(geometry_encoding=encoding).__arrow_c_array__()
    field = pa.Field._import_from_c_capsule(schema_capsule)
    assert field.name == "geom"
    assert (
        field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
        if encoding == "WKB"
        else b"geoarrow.polygon"
    )
    meta = json.loads(field.metadata[b"ARROW:extension:metadata"])
    assert json.loads(meta["crs"])["id"]["code"] == 4326

    # ensure it also works without a name
    gser = GeoSeries([box(0, 0, 10, 10)])
    schema_capsule, _ = gser.to_arrow(geometry_encoding=encoding).__arrow_c_array__()
    field = pa.Field._import_from_c_capsule(schema_capsule)
    assert field.name == ""


def test_geoarrow_unsupported_encoding():
    gdf = GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="epsg:4326")

    with pytest.raises(ValueError, match="Expected geometry encoding"):
        gdf.to_arrow(geometry_encoding="invalid")

    with pytest.raises(ValueError, match="Expected geometry encoding"):
        gdf.geometry.to_arrow(geometry_encoding="invalid")


def test_geoarrow_mixed_geometry_types():
    gdf = GeoDataFrame(
        {"geometry": [Point(0, 0), box(0, 0, 10, 10)]},
        crs="epsg:4326",
    )

    with pytest.raises(ValueError, match="Geometry type combination is not supported"):
        gdf.to_arrow(geometry_encoding="geoarrow")

    gdf = GeoDataFrame(
        {"geometry": [Point(0, 0), MultiPoint([(0, 0), (1, 1)])]},
        crs="epsg:4326",
    )
    result = pa_table(gdf.to_arrow(geometry_encoding="geoarrow"))
    assert (
        result.schema.field("geometry").metadata[b"ARROW:extension:name"]
        == b"geoarrow.multipoint"
    )


@pytest.mark.parametrize("geom_type", ["point", "polygon"])
@pytest.mark.parametrize(
    "encoding, interleaved", [("WKB", True), ("geoarrow", True), ("geoarrow", False)]
)
def test_geoarrow_missing(encoding, interleaved, geom_type):
    # dummy test for single geometry type until missing values are included
    # in the test data for test_geoarrow_export
    gdf = GeoDataFrame(
        geometry=[Point(0, 0) if geom_type == "point" else box(0, 0, 10, 10), None],
        crs="epsg:4326",
    )
    if (
        encoding == "geoarrow"
        and geom_type == "point"
        and interleaved
        and Version(pa.__version__) < Version("15.0.0")
    ):
        with pytest.raises(
            ValueError,
            match="Converting point geometries with missing values is not supported",
        ):
            gdf.to_arrow(geometry_encoding=encoding, interleaved=interleaved)
        return
    result = pa_table(gdf.to_arrow(geometry_encoding=encoding, interleaved=interleaved))
    assert result["geometry"].null_count == 1
    assert result["geometry"].is_null().to_pylist() == [False, True]


@contextlib.contextmanager
def with_geoarrow_extension_types():
    gp = pytest.importorskip("geoarrow.pyarrow")
    gp.register_extension_types()
    try:
        yield
    finally:
        gp.unregister_extension_types()


@pytest.mark.parametrize("dim", ["xy", "xyz"])
@pytest.mark.parametrize(
    "geometry_type",
    ["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
)
def test_geoarrow_export_with_extension_types(geometry_type, dim):
    # ensure the exported data can be imported by geoarrow-pyarrow and are
    # recognized as extension types
    base_path = DATA_PATH / "geoarrow"
    suffix = geometry_type + ("_z" if dim == "xyz" else "")

    # Read the example data
    df = feather.read_feather(base_path / f"example-{suffix}-wkb.arrow")
    df["geometry"] = GeoSeries.from_wkb(df["geometry"])
    df["row_number"] = df["row_number"].astype("int32")
    df = GeoDataFrame(df)
    df.geometry.crs = None

    pytest.importorskip("geoarrow.pyarrow")

    with with_geoarrow_extension_types():
        result1 = pa_table(df.to_arrow(geometry_encoding="WKB"))
        assert isinstance(result1["geometry"].type, pa.ExtensionType)

        result2 = pa_table(df.to_arrow(geometry_encoding="geoarrow"))
        assert isinstance(result2["geometry"].type, pa.ExtensionType)

        result3 = pa_table(df.to_arrow(geometry_encoding="geoarrow", interleaved=False))
        assert isinstance(result3["geometry"].type, pa.ExtensionType)
