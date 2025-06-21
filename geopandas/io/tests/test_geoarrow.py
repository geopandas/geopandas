import contextlib
import json
import os
import pathlib
from packaging.version import Version

import numpy as np
from pandas import ArrowDtype

import shapely
from shapely import MultiPoint, Point, box

from geopandas import GeoDataFrame, GeoSeries

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.compute as pc
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
    geom_type = left["geometry"].type
    # in case of Points (directly the inner fixed_size_list or struct type)
    # -> there are NaNs for empties -> we need to compare them separately
    # and then fill, because pyarrow.Table.equals considers NaNs as not equal
    if pa.types.is_fixed_size_list(geom_type):
        left_values = left["geometry"].chunk(0).values
        right_values = right["geometry"].chunk(0).values
        assert pc.is_nan(left_values).equals(pc.is_nan(right_values))
        left_geoms = pa.FixedSizeListArray.from_arrays(
            pc.replace_with_mask(left_values, pc.is_nan(left_values), 0.0),
            type=left["geometry"].type,
        )
        right_geoms = pa.FixedSizeListArray.from_arrays(
            pc.replace_with_mask(right_values, pc.is_nan(right_values), 0.0),
            type=right["geometry"].type,
        )
        left = left.set_column(1, left.schema.field("geometry"), left_geoms)
        right = right.set_column(1, right.schema.field("geometry"), right_geoms)

    elif pa.types.is_struct(geom_type):
        left_arr = left["geometry"].chunk(0)
        right_arr = right["geometry"].chunk(0)

        for i in range(left_arr.type.num_fields):
            assert pc.is_nan(left_arr.field(i)).equals(pc.is_nan(right_arr.field(i)))

        left_geoms = pa.StructArray.from_arrays(
            [
                pc.replace_with_mask(
                    left_arr.field(i), pc.is_nan(left_arr.field(i)), 0.0
                )
                for i in range(left_arr.type.num_fields)
            ],
            fields=list(left["geometry"].type),
        )
        right_geoms = pa.StructArray.from_arrays(
            [
                pc.replace_with_mask(
                    right_arr.field(i), pc.is_nan(right_arr.field(i)), 0.0
                )
                for i in range(right_arr.type.num_fields)
            ],
            fields=list(right["geometry"].type),
        )

        left = left.set_column(1, left.schema.field("geometry"), left_geoms)
        right = right.set_column(1, right.schema.field("geometry"), right_geoms)

    if left.equals(right, check_metadata=check_metadata):
        return

    if not left.schema.equals(right.schema):
        raise AssertionError(
            f"Schema not equal\nLeft:\n{left.schema}\nRight:\n{right.schema}"
        )

    if check_metadata:
        if not left.schema.equals(right.schema, check_metadata=True):
            if not left.schema.metadata == right.schema.metadata:
                raise AssertionError(
                    f"Metadata not equal\nLeft:\n{left.schema.metadata}\n"
                    f"Right:\n{right.schema.metadata}"
                )
        for col in left.schema.names:
            assert left.schema.field(col).equals(
                right.schema.field(col), check_metadata=True
            )

    for col in left.column_names:
        a_left = pa.concat_arrays(left.column(col).chunks)
        a_right = pa.concat_arrays(right.column(col).chunks)
        if not a_left.equals(a_right):
            raise AssertionError(f"Column '{col}' not equal:\n{a_left.diff(a_right)}")

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
    ["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
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
    df.geometry.array.crs = None

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
    if geometry_encoding != "WKB" and geometry_type == "point":
        # for points, we again have to handle NaNs separately, we already did that
        # for table so let's just skip this part
        return
    result_arr = pa_array(
        df.geometry.to_arrow(
            geometry_encoding=geometry_encoding, interleaved=interleaved
        )
    )
    if mask_nonempty is not None:
        result_arr = result_arr.filter(mask_nonempty)
    assert result_arr.equals(expected["geometry"].chunk(0))


@pytest.mark.skipif(
    Version(shapely.__version__) < Version("2.0.2"),
    reason="from_ragged_array failing with read-only array input",
)
@pytest.mark.parametrize("encoding", ["WKB", "geoarrow"])
def test_geoarrow_to_pandas_kwargs(encoding):
    g = box(0, 0, 10, 10)
    gdf = GeoDataFrame({"geometry": [g], "i": [1], "s": ["a"]})
    table = pa_table(gdf.to_arrow(geometry_encoding=encoding))
    # simulate the `dtype_backend="pyarrow"` option in `pandas.read_parquet`
    gdf_roundtrip = GeoDataFrame.from_arrow(
        table, to_pandas_kwargs={"types_mapper": ArrowDtype}
    )
    assert isinstance(gdf_roundtrip, GeoDataFrame)
    assert isinstance(gdf_roundtrip.dtypes["i"], ArrowDtype)
    assert isinstance(gdf_roundtrip.dtypes["s"], ArrowDtype)


@pytest.mark.skipif(
    Version(shapely.__version__) < Version("2.0.2"),
    reason="from_ragged_array failing with read-only array input",
)
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
    assert meta1["crs"]["id"]["code"] == 4326
    meta2 = json.loads(
        result.schema.field("geom2").metadata[b"ARROW:extension:metadata"]
    )
    assert meta2["crs"]["id"]["code"] == 3857

    roundtripped = GeoDataFrame.from_arrow(result)
    assert_geodataframe_equal(gdf, roundtripped)
    assert gdf.geometry.crs == "epsg:4326"
    assert gdf.geom2.crs == "epsg:3857"


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
    assert meta["crs"]["id"]["code"] == 4326

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


def test_geoarrow_include_z():
    gdf = GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1), Point()]})

    table = pa_table(gdf.to_arrow(geometry_encoding="geoarrow"))
    assert table["geometry"].type.value_field.name == "xy"
    assert table["geometry"].type.list_size == 2

    table = pa_table(gdf.to_arrow(geometry_encoding="geoarrow", include_z=True))
    assert table["geometry"].type.value_field.name == "xyz"
    assert table["geometry"].type.list_size == 3
    assert np.isnan(table["geometry"].chunk(0).values.to_numpy()[2::3]).all()

    gdf = GeoDataFrame({"geometry": [Point(0, 0, 0), Point(1, 1, 1), Point()]})

    table = pa_table(gdf.to_arrow(geometry_encoding="geoarrow"))
    assert table["geometry"].type.value_field.name == "xyz"
    assert table["geometry"].type.list_size == 3

    table = pa_table(gdf.to_arrow(geometry_encoding="geoarrow", include_z=False))
    assert table["geometry"].type.value_field.name == "xy"
    assert table["geometry"].type.list_size == 2


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
    df.geometry.array.crs = None

    pytest.importorskip("geoarrow.pyarrow")

    with with_geoarrow_extension_types():
        result1 = pa_table(df.to_arrow(geometry_encoding="WKB"))
        assert isinstance(result1["geometry"].type, pa.ExtensionType)

        result2 = pa_table(df.to_arrow(geometry_encoding="geoarrow"))
        assert isinstance(result2["geometry"].type, pa.ExtensionType)

        result3 = pa_table(df.to_arrow(geometry_encoding="geoarrow", interleaved=False))
        assert isinstance(result3["geometry"].type, pa.ExtensionType)


def test_geoarrow_export_empty():
    gdf_empty = GeoDataFrame(columns=["col", "geometry"], geometry="geometry")
    gdf_all_missing = GeoDataFrame(
        {"col": [1], "geometry": [None]}, geometry="geometry"
    )

    # no geometries to infer the geometry type -> raise error for now
    with pytest.raises(NotImplementedError):
        gdf_empty.to_arrow(geometry_encoding="geoarrow")

    with pytest.raises(NotImplementedError):
        gdf_all_missing.to_arrow(geometry_encoding="geoarrow")

    # with WKB encoding it roundtrips fine
    result = pa_table(gdf_empty.to_arrow(geometry_encoding="WKB"))
    roundtripped = GeoDataFrame.from_arrow(result)
    assert_geodataframe_equal(gdf_empty, roundtripped)

    result = pa_table(gdf_all_missing.to_arrow(geometry_encoding="WKB"))
    roundtripped = GeoDataFrame.from_arrow(result)
    assert_geodataframe_equal(gdf_all_missing, roundtripped)


@pytest.mark.skipif(
    Version(shapely.__version__) < Version("2.0.2"),
    reason="from_ragged_array failing with read-only array input",
)
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


@pytest.mark.skipif(
    Version(shapely.__version__) < Version("2.0.2"),
    reason="from_ragged_array failing with read-only array input",
)
@pytest.mark.parametrize("encoding", ["WKB", "geoarrow"])
def test_geoarrow_import_geometry_column(encoding):
    pytest.importorskip("pyproj")
    # ensure each geometry column has its own crs
    gdf = GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    gdf["centroid"] = gdf.geometry.centroid

    result = GeoDataFrame.from_arrow(pa_table(gdf.to_arrow(geometry_encoding=encoding)))
    assert_geodataframe_equal(result, gdf)
    assert result.active_geometry_name == "geometry"

    result = GeoDataFrame.from_arrow(
        pa_table(gdf[["centroid"]].to_arrow(geometry_encoding=encoding))
    )
    assert result.active_geometry_name == "centroid"

    result = GeoDataFrame.from_arrow(
        pa_table(gdf.to_arrow(geometry_encoding=encoding)), geometry="centroid"
    )
    assert result.active_geometry_name == "centroid"
    assert_geodataframe_equal(result, gdf.set_geometry("centroid"))


def test_geoarrow_import_missing_geometry():
    pytest.importorskip("pyarrow", minversion="14.0.0")

    table = pa.table({"a": [0, 1, 2], "b": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="No geometry column found"):
        GeoDataFrame.from_arrow(table)

    with pytest.raises(ValueError, match="No GeoArrow geometry field found"):
        GeoSeries.from_arrow(table["a"].chunk(0))


def test_geoarrow_import_capsule_interface():
    # ensure we can import non-pyarrow object
    pytest.importorskip("pyarrow", minversion="14.0.0")
    gdf = GeoDataFrame({"col": [1]}, geometry=[box(0, 0, 10, 10)])

    result = GeoDataFrame.from_arrow(gdf.to_arrow())
    assert_geodataframe_equal(result, gdf)


@pytest.mark.parametrize("dim", ["xy", "xyz"])
@pytest.mark.parametrize(
    "geometry_type",
    ["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
)
def test_geoarrow_import_from_extension_types(geometry_type, dim):
    # ensure the exported data can be imported by geoarrow-pyarrow and are
    # recognized as extension types
    pytest.importorskip("pyproj")
    base_path = DATA_PATH / "geoarrow"
    suffix = geometry_type + ("_z" if dim == "xyz" else "")

    # Read the example data
    df = feather.read_feather(base_path / f"example-{suffix}-wkb.arrow")
    df["geometry"] = GeoSeries.from_wkb(df["geometry"])
    df = GeoDataFrame(df, crs="EPSG:3857")

    pytest.importorskip("geoarrow.pyarrow")

    with with_geoarrow_extension_types():
        result1 = GeoDataFrame.from_arrow(
            pa_table(df.to_arrow(geometry_encoding="WKB"))
        )
        assert_geodataframe_equal(result1, df)

        result2 = GeoDataFrame.from_arrow(
            pa_table(df.to_arrow(geometry_encoding="geoarrow"))
        )
        assert_geodataframe_equal(result2, df)

        result3 = GeoDataFrame.from_arrow(
            pa_table(df.to_arrow(geometry_encoding="geoarrow", interleaved=False))
        )
        assert_geodataframe_equal(result3, df)


def test_geoarrow_import_geoseries():
    pytest.importorskip("pyproj")
    gp = pytest.importorskip("geoarrow.pyarrow")
    ser = GeoSeries.from_wkt(["POINT (1 1)", "POINT (2 2)"], crs="EPSG:3857")

    with with_geoarrow_extension_types():
        arr = gp.array(ser.to_arrow(geometry_encoding="WKB"))
        result = GeoSeries.from_arrow(arr)
        assert_geoseries_equal(result, ser)

        arr = gp.array(ser.to_arrow(geometry_encoding="geoarrow"))
        result = GeoSeries.from_arrow(arr)
        assert_geoseries_equal(result, ser)

        # the name is lost when going through a pyarrow.Array
        ser.name = "name"
        arr = gp.array(ser.to_arrow())
        result = GeoSeries.from_arrow(arr)
        assert result.name is None
        # we can specify the name as one of the kwargs
        result = GeoSeries.from_arrow(arr, name="test")
        assert_geoseries_equal(result, ser)


def test_geoarrow_import_unknown_geoarrow_type():
    gdf = GeoDataFrame({"col": [1]}, geometry=[box(0, 0, 10, 10)])
    table = pa_table(gdf.to_arrow())
    schema = table.schema
    new_field = schema.field("geometry").with_metadata(
        {
            b"ARROW:extension:name": b"geoarrow.unknown",
            b"ARROW:extension:metadata": b"{}",
        }
    )

    new_schema = pa.schema([schema.field(0), new_field])
    new_table = table.cast(new_schema)

    with pytest.raises(TypeError, match="Unknown GeoArrow extension type"):
        GeoDataFrame.from_arrow(new_table)
