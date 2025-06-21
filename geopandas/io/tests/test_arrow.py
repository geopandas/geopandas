import json
import os
import pathlib
from itertools import product
from packaging.version import Version

import numpy as np
from pandas import ArrowDtype, DataFrame
from pandas import read_parquet as pd_read_parquet

import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

import geopandas
from geopandas import GeoDataFrame, read_feather, read_file, read_parquet
from geopandas._compat import HAS_PYPROJ
from geopandas.array import to_wkb
from geopandas.io.arrow import (
    METADATA_VERSION,
    SUPPORTED_VERSIONS,
    _convert_bbox_to_parquet_filter,
    _create_metadata,
    _decode_metadata,
    _encode_metadata,
    _geopandas_to_arrow,
    _get_filesystem_path,
    _remove_id_from_member_of_ensembles,
    _validate_dataframe,
    _validate_geo_metadata,
)

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock
from pandas.testing import assert_frame_equal

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / "data"


# Skip all tests in this module if pyarrow is not available
pyarrow = pytest.importorskip("pyarrow")

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import feather


@pytest.fixture(params=["parquet", pytest.param("feather")])
def file_format(request):
    if request.param == "parquet":
        return read_parquet, GeoDataFrame.to_parquet
    elif request.param == "feather":
        return read_feather, GeoDataFrame.to_feather


def test_create_metadata(naturalearth_lowres):
    df = read_file(naturalearth_lowres)
    metadata = _create_metadata(df, geometry_encoding={"geometry": "WKB"})

    assert isinstance(metadata, dict)
    assert metadata["version"] == METADATA_VERSION
    assert metadata["primary_column"] == "geometry"
    assert "geometry" in metadata["columns"]
    if HAS_PYPROJ:
        crs_expected = df.crs.to_json_dict()
        _remove_id_from_member_of_ensembles(crs_expected)
        assert metadata["columns"]["geometry"]["crs"] == crs_expected
    assert metadata["columns"]["geometry"]["encoding"] == "WKB"
    assert metadata["columns"]["geometry"]["geometry_types"] == [
        "MultiPolygon",
        "Polygon",
    ]

    assert np.array_equal(
        metadata["columns"]["geometry"]["bbox"], df.geometry.total_bounds
    )

    assert metadata["creator"]["library"] == "geopandas"
    assert metadata["creator"]["version"] == geopandas.__version__

    # specifying non-WKB encoding sets default schema to 1.1.0
    metadata = _create_metadata(df, geometry_encoding={"geometry": "point"})
    assert metadata["version"] == "1.1.0"
    assert metadata["columns"]["geometry"]["encoding"] == "point"

    # check that providing no geometry encoding defaults to WKB
    metadata = _create_metadata(df)
    assert metadata["columns"]["geometry"]["encoding"] == "WKB"


def test_create_metadata_with_z_geometries():
    geometry_types = [
        "Point",
        "Point Z",
        "LineString",
        "LineString Z",
        "Polygon",
        "Polygon Z",
        "MultiPolygon",
        "MultiPolygon Z",
    ]
    df = geopandas.GeoDataFrame(
        {
            "geo_type": geometry_types,
            "geometry": [
                Point(1, 2),
                Point(1, 2, 3),
                LineString([(0, 0), (1, 1), (2, 2)]),
                LineString([(0, 0, 1), (1, 1, 2), (2, 2, 3)]),
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                Polygon([(0, 0, 0), (0, 1, 0.5), (1, 1, 1), (1, 0, 0.5)]),
                MultiPolygon(
                    [
                        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                        Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]),
                    ]
                ),
                MultiPolygon(
                    [
                        Polygon([(0, 0, 0), (0, 1, 0.5), (1, 1, 1), (1, 0, 0.5)]),
                        Polygon(
                            [
                                (0.5, 0.5, 1),
                                (0.5, 1.5, 1.5),
                                (1.5, 1.5, 2),
                                (1.5, 0.5, 1.5),
                            ]
                        ),
                    ]
                ),
            ],
        },
    )
    metadata = _create_metadata(df, geometry_encoding={"geometry": "WKB"})
    assert sorted(metadata["columns"]["geometry"]["geometry_types"]) == sorted(
        geometry_types
    )
    # only 3D geometries
    metadata = _create_metadata(df.iloc[1::2], geometry_encoding={"geometry": "WKB"})
    assert all(
        geom_type.endswith(" Z")
        for geom_type in metadata["columns"]["geometry"]["geometry_types"]
    )

    metadata = _create_metadata(df.iloc[5:7], geometry_encoding={"geometry": "WKB"})
    assert metadata["columns"]["geometry"]["geometry_types"] == [
        "MultiPolygon",
        "Polygon Z",
    ]


def test_crs_metadata_datum_ensemble():
    pyproj = pytest.importorskip("pyproj")
    # compatibility for older PROJ versions using PROJJSON with datum ensembles
    # https://github.com/geopandas/geopandas/pull/2453
    crs = pyproj.CRS("EPSG:4326")
    crs_json = crs.to_json_dict()
    check_ensemble = False
    if "datum_ensemble" in crs_json:
        # older version of PROJ don't yet have datum ensembles
        check_ensemble = True
        assert "id" in crs_json["datum_ensemble"]["members"][0]
    _remove_id_from_member_of_ensembles(crs_json)
    if check_ensemble:
        assert "id" not in crs_json["datum_ensemble"]["members"][0]
    # ensure roundtrip still results in an equivalent CRS
    assert pyproj.CRS(crs_json) == crs


def test_write_metadata_invalid_spec_version(tmp_path):
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="schema_version must be one of"):
        _create_metadata(gdf, schema_version="invalid")

    with pytest.raises(
        ValueError,
        match="'geoarrow' encoding is only supported with schema version >= 1.1.0",
    ):
        gdf.to_parquet(tmp_path, schema_version="1.0.0", geometry_encoding="geoarrow")


def test_encode_metadata():
    metadata = {"a": "b"}

    expected = b'{"a": "b"}'
    assert _encode_metadata(metadata) == expected


def test_decode_metadata():
    metadata_str = b'{"a": "b"}'

    expected = {"a": "b"}
    assert _decode_metadata(metadata_str) == expected

    assert _decode_metadata(None) is None


def test_validate_dataframe(naturalearth_lowres):
    df = read_file(naturalearth_lowres)

    # valid: should not raise ValueError
    _validate_dataframe(df)
    _validate_dataframe(df.set_index("iso_a3"))

    # add column with non-string type
    df[0] = 1

    # invalid: should raise ValueError
    with pytest.raises(ValueError):
        _validate_dataframe(df)

    with pytest.raises(ValueError):
        _validate_dataframe(df.set_index(0))

    # not a DataFrame: should raise ValueError
    with pytest.raises(ValueError):
        _validate_dataframe("not a dataframe")


def test_validate_geo_metadata_valid():
    _validate_geo_metadata(
        {
            "primary_column": "geometry",
            "columns": {"geometry": {"crs": None, "encoding": "WKB"}},
            "schema_version": "0.1.0",
        }
    )

    _validate_geo_metadata(
        {
            "primary_column": "geometry",
            "columns": {"geometry": {"crs": None, "encoding": "WKB"}},
            "version": "<version>",
        }
    )

    _validate_geo_metadata(
        {
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "crs": {
                        # truncated PROJJSON for testing, as PROJJSON contents
                        # not validated here
                        "id": {"authority": "EPSG", "code": 4326},
                    },
                    "encoding": "point",
                }
            },
            "version": "0.4.0",
        }
    )


@pytest.mark.parametrize(
    "metadata,error",
    [
        (None, "Missing or malformed geo metadata in Parquet/Feather file"),
        ({}, "Missing or malformed geo metadata in Parquet/Feather file"),
        # missing "version" key:
        (
            {"primary_column": "foo", "columns": None},
            "'geo' metadata in Parquet/Feather file is missing required key",
        ),
        # missing "columns" key:
        (
            {"primary_column": "foo", "version": "<version>"},
            "'geo' metadata in Parquet/Feather file is missing required key:",
        ),
        # missing "primary_column"
        (
            {"columns": [], "version": "<version>"},
            "'geo' metadata in Parquet/Feather file is missing required key:",
        ),
        (
            {"primary_column": "foo", "columns": [], "version": "<version>"},
            "'columns' in 'geo' metadata must be a dict",
        ),
        # missing "encoding" for column
        (
            {"primary_column": "foo", "columns": {"foo": {}}, "version": "<version>"},
            (
                "'geo' metadata in Parquet/Feather file is missing required key "
                "'encoding' for column 'foo'"
            ),
        ),
        # invalid column encoding
        (
            {
                "primary_column": "foo",
                "columns": {"foo": {"crs": None, "encoding": None}},
                "version": "<version>",
            },
            "Only WKB geometry encoding",
        ),
        (
            {
                "primary_column": "foo",
                "columns": {"foo": {"crs": None, "encoding": "BKW"}},
                "version": "<version>",
            },
            "Only WKB geometry encoding",
        ),
    ],
)
def test_validate_geo_metadata_invalid(metadata, error):
    with pytest.raises(ValueError, match=error):
        _validate_geo_metadata(metadata)


def test_validate_geo_metadata_edges():
    metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"crs": None, "encoding": "WKB", "edges": "spherical"}},
        "version": "1.0.0-beta.1",
    }
    with pytest.warns(
        UserWarning,
        match="The geo metadata indicate that column 'geometry' has spherical edges",
    ):
        _validate_geo_metadata(metadata)


def test_to_parquet_fails_on_invalid_engine(tmpdir):
    df = GeoDataFrame(data=[[1, 2, 3]], columns=["a", "b", "a"], geometry=[Point(1, 1)])

    with pytest.raises(
        ValueError,
        match=(
            "GeoPandas only supports using pyarrow as the engine for "
            "to_parquet: 'fastparquet' passed instead."
        ),
    ):
        df.to_parquet(tmpdir / "test.parquet", engine="fastparquet")


@mock.patch("geopandas.io.arrow._to_parquet")
def test_to_parquet_does_not_pass_engine_along(mock_to_parquet):
    df = GeoDataFrame(data=[[1, 2, 3]], columns=["a", "b", "a"], geometry=[Point(1, 1)])
    df.to_parquet("", engine="pyarrow")
    # assert that engine keyword is not passed through to _to_parquet (and thus
    # parquet.write_table)
    mock_to_parquet.assert_called_with(
        df,
        "",
        compression="snappy",
        geometry_encoding="WKB",
        index=None,
        schema_version=None,
        write_covering_bbox=False,
    )


# TEMPORARY: used to determine if pyarrow fails for roundtripping pandas data
# without geometries
def test_pandas_parquet_roundtrip1(tmpdir):
    df = DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    pq_df = pd_read_parquet(filename)

    assert_frame_equal(df, pq_df)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb_filename"]
)
def test_pandas_parquet_roundtrip2(test_dataset, tmpdir, request):
    path = request.getfixturevalue(test_dataset)
    df = DataFrame(read_file(path).drop(columns=["geometry"]))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    pq_df = pd_read_parquet(filename)

    assert_frame_equal(df, pq_df)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb_filename"]
)
def test_roundtrip(tmpdir, file_format, test_dataset, request):
    """Writing to parquet should not raise errors, and should not alter original
    GeoDataFrame
    """
    path = request.getfixturevalue(test_dataset)
    reader, writer = file_format

    df = read_file(path)
    orig = df.copy()

    filename = os.path.join(str(tmpdir), "test.pq")

    writer(df, filename)

    assert os.path.exists(filename)

    # make sure that the original data frame is unaltered
    assert_geodataframe_equal(df, orig)

    # make sure that we can roundtrip the data frame
    pq_df = reader(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


def test_index(tmpdir, file_format, naturalearth_lowres):
    """Setting index=`True` should preserve index in output, and
    setting index=`False` should drop index from output.
    """
    reader, writer = file_format

    df = read_file(naturalearth_lowres).set_index("iso_a3")

    filename = os.path.join(str(tmpdir), "test_with_index.pq")
    writer(df, filename, index=True)
    pq_df = reader(filename)
    assert_geodataframe_equal(df, pq_df)

    filename = os.path.join(str(tmpdir), "drop_index.pq")
    writer(df, filename, index=False)
    pq_df = reader(filename)
    assert_geodataframe_equal(df.reset_index(drop=True), pq_df)


def test_column_order(tmpdir, file_format, naturalearth_lowres):
    """The order of columns should be preserved in the output."""
    reader, writer = file_format

    df = read_file(naturalearth_lowres)
    df = df.set_index("iso_a3")
    df["geom2"] = df.geometry.representative_point()
    table = _geopandas_to_arrow(df)
    custom_column_order = [
        "iso_a3",
        "geom2",
        "pop_est",
        "continent",
        "name",
        "geometry",
        "gdp_md_est",
    ]
    table = table.select(custom_column_order)

    if reader is read_parquet:
        filename = os.path.join(str(tmpdir), "test_column_order.pq")
        pq.write_table(table, filename)
    else:
        filename = os.path.join(str(tmpdir), "test_column_order.feather")
        feather.write_feather(table, filename)

    result = reader(filename)
    assert list(result.columns) == custom_column_order[1:]
    assert_geodataframe_equal(result, df[custom_column_order[1:]])


@pytest.mark.parametrize(
    "compression", ["snappy", "gzip", "brotli", "lz4", "zstd", None]
)
def test_parquet_compression(compression, tmpdir, naturalearth_lowres):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """

    df = read_file(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, compression=compression)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


@pytest.mark.parametrize("compression", ["uncompressed", "lz4", "zstd"])
def test_feather_compression(compression, tmpdir, naturalearth_lowres):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """

    df = read_file(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.feather")
    df.to_feather(filename, compression=compression)
    pq_df = read_feather(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


def test_parquet_multiple_geom_cols(tmpdir, file_format, naturalearth_lowres):
    """If multiple geometry columns are present when written to parquet,
    they should all be returned as such when read from parquet.
    """
    reader, writer = file_format

    df = read_file(naturalearth_lowres)
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)

    assert os.path.exists(filename)

    pq_df = reader(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)

    assert_geoseries_equal(df.geom2, pq_df.geom2, check_geom_type=True)


def test_parquet_missing_metadata(tmpdir, naturalearth_lowres):
    """Missing geo metadata, such as from a parquet file created
    from a pandas DataFrame, will raise a ValueError.
    """

    df = read_file(naturalearth_lowres)

    # convert to DataFrame
    df = DataFrame(df)

    # convert the geometry column so we can extract later
    df["geometry"] = to_wkb(df["geometry"].values)

    filename = os.path.join(str(tmpdir), "test.pq")

    # use pandas to_parquet (no geo metadata)
    df.to_parquet(filename)

    # missing metadata will raise ValueError
    with pytest.raises(
        ValueError, match="Missing geo metadata in Parquet/Feather file."
    ):
        read_parquet(filename)


def test_parquet_missing_metadata2(tmpdir):
    """Missing geo metadata, such as from a parquet file created
    from a pyarrow Table (which will also not contain pandas metadata),
    will raise a ValueError.
    """
    import pyarrow.parquet as pq

    table = pyarrow.table({"a": [1, 2, 3]})
    filename = os.path.join(str(tmpdir), "test.pq")

    # use pyarrow.parquet write_table (no geo metadata, but also no pandas metadata)
    pq.write_table(table, filename)

    # missing metadata will raise ValueError
    with pytest.raises(
        ValueError, match="Missing geo metadata in Parquet/Feather file."
    ):
        read_parquet(filename)


@pytest.mark.parametrize(
    "geo_meta,error",
    [
        ({"geo": b""}, "Missing or malformed geo metadata in Parquet/Feather file"),
        (
            {"geo": _encode_metadata({})},
            "Missing or malformed geo metadata in Parquet/Feather file",
        ),
        (
            {"geo": _encode_metadata({"foo": "bar"})},
            "'geo' metadata in Parquet/Feather file is missing required key",
        ),
    ],
)
def test_parquet_invalid_metadata(tmpdir, geo_meta, error, naturalearth_lowres):
    """Has geo metadata with missing required fields will raise a ValueError.

    This requires writing the parquet file directly below, so that we can
    control the metadata that is written for this test.
    """

    from pyarrow import Table, parquet

    df = read_file(naturalearth_lowres)

    # convert to DataFrame and encode geometry to WKB
    df = DataFrame(df)
    df["geometry"] = to_wkb(df["geometry"].values)

    table = Table.from_pandas(df)
    metadata = table.schema.metadata
    metadata.update(geo_meta)
    table = table.replace_schema_metadata(metadata)

    filename = os.path.join(str(tmpdir), "test.pq")
    parquet.write_table(table, filename)

    with pytest.raises(ValueError, match=error):
        read_parquet(filename)


def test_subset_columns(tmpdir, file_format, naturalearth_lowres):
    """Reading a subset of columns should correctly decode selected geometry
    columns.
    """
    reader, writer = file_format

    df = read_file(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename, columns=["name", "geometry"])

    assert_geodataframe_equal(df[["name", "geometry"]], pq_df)

    with pytest.raises(
        ValueError, match="No geometry columns are included in the columns read"
    ):
        reader(filename, columns=["name"])


def test_promote_secondary_geometry(tmpdir, file_format, naturalearth_lowres):
    """Reading a subset of columns that does not include the primary geometry
    column should promote the first geometry column present.
    """
    reader, writer = file_format

    df = read_file(naturalearth_lowres)
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename, columns=["name", "geom2"])

    assert_geodataframe_equal(df.set_geometry("geom2")[["name", "geom2"]], pq_df)

    df["geom3"] = df.geometry.copy()

    writer(df, filename)
    with pytest.warns(
        UserWarning,
        match="Multiple non-primary geometry columns read from Parquet/Feather file.",
    ):
        pq_df = reader(filename, columns=["name", "geom2", "geom3"])

    assert_geodataframe_equal(
        df.set_geometry("geom2")[["name", "geom2", "geom3"]], pq_df
    )


def test_columns_no_geometry(tmpdir, file_format, naturalearth_lowres):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""
    reader, writer = file_format

    df = read_file(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)

    with pytest.raises(ValueError):
        reader(filename, columns=["name"])


def test_missing_crs(tmpdir, file_format, naturalearth_lowres):
    """If CRS is `None`, it should be properly handled
    and remain `None` when read from parquet`.
    """
    reader, writer = file_format

    df = read_file(naturalearth_lowres)
    df.geometry.array.crs = None

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename)

    assert pq_df.crs is None

    assert_geodataframe_equal(df, pq_df, check_crs=True)


def test_default_geo_col_writes(tmp_path):
    # edge case geo col name None writes successfully
    df = GeoDataFrame({"a": [1, 2]})
    df.to_parquet(tmp_path / "test.pq")
    # cannot be round tripped as gdf due to invalid geom col
    pq_df = pd_read_parquet(tmp_path / "test.pq")
    assert_frame_equal(df, pq_df)


def test_fsspec_url(naturalearth_lowres):
    _ = pytest.importorskip("fsspec")
    import fsspec.implementations.memory

    class MyMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
        # Simple fsspec filesystem that adds a required keyword.
        # Attempting to use this filesystem without the keyword will raise an exception.
        def __init__(self, is_set, *args, **kwargs):
            self.is_set = is_set
            super().__init__(*args, **kwargs)

    fsspec.register_implementation("memory", MyMemoryFileSystem, clobber=True)
    memfs = MyMemoryFileSystem(is_set=True)

    df = read_file(naturalearth_lowres)

    with memfs.open("data.parquet", "wb") as f:
        df.to_parquet(f)

    result = read_parquet("memory://data.parquet", storage_options={"is_set": True})
    assert_geodataframe_equal(result, df)

    result = read_parquet("memory://data.parquet", filesystem=memfs)
    assert_geodataframe_equal(result, df)

    # reset fsspec registry
    fsspec.register_implementation(
        "memory", fsspec.implementations.memory.MemoryFileSystem, clobber=True
    )


def test_non_fsspec_url_with_storage_options_raises(naturalearth_lowres):
    with pytest.raises(ValueError, match="storage_options"):
        read_parquet(naturalearth_lowres, storage_options={"foo": "bar"})


def test_prefers_pyarrow_fs():
    filesystem, _ = _get_filesystem_path("file:///data.parquet")
    assert isinstance(filesystem, pyarrow.fs.LocalFileSystem)


def test_write_read_parquet_expand_user():
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="epsg:4326")
    test_file = "~/test_file.parquet"
    gdf.to_parquet(test_file)
    pq_df = geopandas.read_parquet(test_file)
    assert_geodataframe_equal(gdf, pq_df, check_crs=True)
    os.remove(os.path.expanduser(test_file))


def test_write_read_feather_expand_user():
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="epsg:4326")
    test_file = "~/test_file.feather"
    gdf.to_feather(test_file)
    f_df = geopandas.read_feather(test_file)
    assert_geodataframe_equal(gdf, f_df, check_crs=True)
    os.remove(os.path.expanduser(test_file))


@pytest.mark.parametrize("geometry", [[], [None]])
def test_write_empty_bbox(tmpdir, geometry):
    # empty dataframe or all missing geometries -> avoid bbox with NaNs
    gdf = geopandas.GeoDataFrame({"col": [1] * len(geometry)}, geometry=geometry)
    gdf.to_parquet(tmpdir / "test.parquet")

    from pyarrow.parquet import read_table

    table = read_table(tmpdir / "test.parquet")
    metadata = json.loads(table.schema.metadata[b"geo"])
    assert "encoding" in metadata["columns"]["geometry"]
    assert "bbox" not in metadata["columns"]["geometry"]


@pytest.mark.parametrize("format", ["feather", "parquet"])
def test_write_read_to_pandas_kwargs(tmpdir, format):
    filename = os.path.join(str(tmpdir), f"test.{format}")
    g = box(0, 0, 10, 10)
    gdf = geopandas.GeoDataFrame({"geometry": [g], "i": [1], "s": ["a"]})

    if format == "feather":
        gdf.to_feather(filename)
        read_func = read_feather
    else:
        gdf.to_parquet(filename)
        read_func = read_parquet

    # simulate the `dtype_backend="pyarrow"` option in `pandas.read_parquet`
    gdf_roundtrip = read_func(filename, to_pandas_kwargs={"types_mapper": ArrowDtype})
    assert isinstance(gdf_roundtrip, geopandas.GeoDataFrame)
    assert isinstance(gdf_roundtrip.dtypes["i"], ArrowDtype)
    assert isinstance(gdf_roundtrip.dtypes["s"], ArrowDtype)


@pytest.mark.parametrize("format", ["feather", "parquet"])
def test_write_read_default_crs(tmpdir, format):
    pyproj = pytest.importorskip("pyproj")
    if format == "feather":
        from pyarrow.feather import write_feather as write
    else:
        from pyarrow.parquet import write_table as write

    filename = os.path.join(str(tmpdir), f"test.{format}")
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    table = _geopandas_to_arrow(gdf)

    # update the geo metadata to strip 'crs' entry
    metadata = table.schema.metadata
    geo_metadata = _decode_metadata(metadata[b"geo"])
    del geo_metadata["columns"]["geometry"]["crs"]
    metadata.update({b"geo": _encode_metadata(geo_metadata)})
    table = table.replace_schema_metadata(metadata)

    write(table, filename)

    read = getattr(geopandas, f"read_{format}")
    df = read(filename)
    assert df.crs.equals(pyproj.CRS("OGC:CRS84"))


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="requires GEOS>=3.10")
def test_write_iso_wkb(tmpdir):
    gdf = geopandas.GeoDataFrame(
        geometry=geopandas.GeoSeries.from_wkt(["POINT Z (1 2 3)"])
    )
    gdf.to_parquet(tmpdir / "test.parquet")

    from pyarrow.parquet import read_table

    table = read_table(tmpdir / "test.parquet")
    wkb = table["geometry"][0].as_py().hex()

    # correct ISO flavor
    assert wkb == "01e9030000000000000000f03f00000000000000400000000000000840"


@pytest.mark.skipif(shapely.geos_version >= (3, 10, 0), reason="tests GEOS<3.10")
def test_write_iso_wkb_old_geos(tmpdir):
    gdf = geopandas.GeoDataFrame(
        geometry=geopandas.GeoSeries.from_wkt(["POINT Z (1 2 3)"])
    )
    with pytest.raises(ValueError, match="Cannot write 3D"):
        gdf.to_parquet(tmpdir / "test.parquet")


@pytest.mark.parametrize(
    "format,schema_version",
    product(["feather", "parquet"], [None] + SUPPORTED_VERSIONS),
)
def test_write_spec_version(tmpdir, format, schema_version):
    if format == "feather":
        from pyarrow.feather import read_table
    else:
        from pyarrow.parquet import read_table

    filename = os.path.join(str(tmpdir), f"test.{format}")
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    write = getattr(gdf, f"to_{format}")
    write(filename, schema_version=schema_version)

    # ensure that we can roundtrip data regardless of version
    read = getattr(geopandas, f"read_{format}")
    df = read(filename)
    assert_geodataframe_equal(df, gdf)

    # verify the correct version is written in the metadata
    schema_version = schema_version or METADATA_VERSION
    table = read_table(filename)
    metadata = json.loads(table.schema.metadata[b"geo"])
    assert metadata["version"] == schema_version

    # verify that CRS is correctly handled between versions
    if HAS_PYPROJ:
        if schema_version == "0.1.0":
            assert metadata["columns"]["geometry"]["crs"] == gdf.crs.to_wkt()

        else:
            crs_expected = gdf.crs.to_json_dict()
            _remove_id_from_member_of_ensembles(crs_expected)
            assert metadata["columns"]["geometry"]["crs"] == crs_expected

    # verify that geometry_type(s) is correctly handled between versions
    if Version(schema_version) <= Version("0.4.0"):
        assert "geometry_type" in metadata["columns"]["geometry"]
        assert metadata["columns"]["geometry"]["geometry_type"] == "Polygon"
    else:
        assert "geometry_types" in metadata["columns"]["geometry"]
        assert metadata["columns"]["geometry"]["geometry_types"] == ["Polygon"]


@pytest.mark.parametrize("version", ["0.1.0", "0.4.0", "1.0.0-beta.1"])
def test_read_versioned_file(version):
    """
    Verify that files for different metadata spec versions can be read
    created for each supported version:

    # small dummy test dataset (not naturalearth_lowres, as this can change over time)
    from shapely.geometry import box, MultiPolygon
    df = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5,5)],
        crs="EPSG:4326",
    )
    df.to_feather(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.feather')
    df.to_parquet(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.parquet')
    """
    expected = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)],
        crs="EPSG:4326",
    )

    df = geopandas.read_feather(DATA_PATH / "arrow" / f"test_data_v{version}.feather")
    assert_geodataframe_equal(df, expected, check_crs=True)

    df = geopandas.read_parquet(DATA_PATH / "arrow" / f"test_data_v{version}.parquet")
    assert_geodataframe_equal(df, expected, check_crs=True)


def test_read_gdal_files():
    """
    Verify that files written by GDAL can be read by geopandas.
    Since it is currently not yet straightforward to install GDAL with
    Parquet/Arrow enabled in our conda setup, we are testing with some
    generated files included in the repo (using GDAL 3.5.0):

    # small dummy test dataset (not naturalearth_lowres, as this can change over time)
    from shapely.geometry import box, MultiPolygon
    df = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5,5)],
        crs="EPSG:4326",
    )
    df.to_file("test_data.gpkg", GEOMETRY_NAME="geometry")
    and then the gpkg file is converted to Parquet/Arrow with:
    $ ogr2ogr -f Parquet -lco FID= test_data_gdal350.parquet test_data.gpkg
    $ ogr2ogr -f Arrow -lco FID= -lco GEOMETRY_ENCODING=WKB test_data_gdal350.arrow test_data.gpkg

    Repeated for GDAL 3.9 which adds a bbox covering column:
    $ ogr2ogr -f Parquet -lco FID= test_data_gdal390.parquet test_data.gpkg
    """  # noqa: E501
    pytest.importorskip("pyproj")
    expected = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)],
        crs="EPSG:4326",
    )

    df = geopandas.read_parquet(DATA_PATH / "arrow" / "test_data_gdal350.parquet")
    assert_geodataframe_equal(df, expected, check_crs=True)

    df = geopandas.read_feather(DATA_PATH / "arrow" / "test_data_gdal350.arrow")
    assert_geodataframe_equal(df, expected, check_crs=True)

    df = geopandas.read_parquet(DATA_PATH / "arrow" / "test_data_gdal390.parquet")
    # recent GDAL no longer writes CRS in metadata in case of EPSG:4326, so comes back
    # as default OGC:CRS84
    expected = expected.to_crs("OGC:CRS84")
    assert_geodataframe_equal(df, expected, check_crs=True)

    df = geopandas.read_parquet(
        DATA_PATH / "arrow" / "test_data_gdal390.parquet", bbox=(0, 0, 2, 2)
    )
    assert len(df) == 1


def test_parquet_read_partitioned_dataset(tmpdir, naturalearth_lowres):
    # we don't yet explicitly support this (in writing), but for Parquet it
    # works for reading (by relying on pyarrow.read_table)
    df = read_file(naturalearth_lowres)

    # manually create partitioned dataset
    basedir = tmpdir / "partitioned_dataset"
    basedir.mkdir()
    df[:100].to_parquet(basedir / "data1.parquet")
    df[100:].to_parquet(basedir / "data2.parquet")

    result = read_parquet(basedir)
    assert_geodataframe_equal(result, df)


def test_parquet_read_partitioned_dataset_fsspec(tmpdir, naturalearth_lowres):
    fsspec = pytest.importorskip("fsspec")

    df = read_file(naturalearth_lowres)

    # manually create partitioned dataset
    memfs = fsspec.filesystem("memory")
    memfs.mkdir("partitioned_dataset")
    with memfs.open("partitioned_dataset/data1.parquet", "wb") as f:
        df[:100].to_parquet(f)
    with memfs.open("partitioned_dataset/data2.parquet", "wb") as f:
        df[100:].to_parquet(f)

    result = read_parquet("memory://partitioned_dataset")
    assert_geodataframe_equal(result, df)


@pytest.mark.parametrize(
    "geometry_type",
    ["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
)
def test_read_parquet_geoarrow(geometry_type):
    result = geopandas.read_parquet(
        DATA_PATH
        / "arrow"
        / "geoparquet"
        / f"data-{geometry_type}-encoding_native.parquet"
    )
    expected = geopandas.read_parquet(
        DATA_PATH
        / "arrow"
        / "geoparquet"
        / f"data-{geometry_type}-encoding_wkb.parquet"
    )
    assert_geodataframe_equal(result, expected, check_crs=True)


@pytest.mark.parametrize(
    "geometry_type",
    ["point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"],
)
def test_geoarrow_roundtrip(tmp_path, geometry_type):
    df = geopandas.read_parquet(
        DATA_PATH
        / "arrow"
        / "geoparquet"
        / f"data-{geometry_type}-encoding_wkb.parquet"
    )

    df.to_parquet(tmp_path / "test.parquet", geometry_encoding="geoarrow")
    result = geopandas.read_parquet(tmp_path / "test.parquet")
    assert_geodataframe_equal(result, df, check_crs=True)


def test_to_parquet_bbox_structure_and_metadata(tmpdir, naturalearth_lowres):
    # check metadata being written for covering.
    from pyarrow import parquet

    df = read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)

    table = parquet.read_table(filename)
    metadata = json.loads(table.schema.metadata[b"geo"].decode("utf-8"))
    assert metadata["columns"]["geometry"]["covering"] == {
        "bbox": {
            "xmin": ["bbox", "xmin"],
            "ymin": ["bbox", "ymin"],
            "xmax": ["bbox", "xmax"],
            "ymax": ["bbox", "ymax"],
        }
    }
    assert "bbox" in table.schema.names
    assert [field.name for field in table.schema.field("bbox").type] == [
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]


@pytest.mark.parametrize(
    "geometry, expected_bbox",
    [
        (Point(1, 3), {"xmin": 1.0, "ymin": 3.0, "xmax": 1.0, "ymax": 3.0}),
        (
            LineString([(1, 1), (3, 3)]),
            {"xmin": 1.0, "ymin": 1.0, "xmax": 3.0, "ymax": 3.0},
        ),
        (
            Polygon([(2, 1), (1, 2), (2, 3), (3, 2)]),
            {"xmin": 1.0, "ymin": 1.0, "xmax": 3.0, "ymax": 3.0},
        ),
        (
            MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]),
            {"xmin": 0.0, "ymin": 0.0, "xmax": 5.0, "ymax": 5.0},
        ),
    ],
    ids=["Point", "LineString", "Polygon", "Multipolygon"],
)
def test_to_parquet_bbox_values(tmpdir, geometry, expected_bbox):
    # check bbox bounds being written for different geometry types.
    import pyarrow.parquet as pq

    df = GeoDataFrame(data=[[1, 2]], columns=["a", "b"], geometry=[geometry])
    filename = os.path.join(str(tmpdir), "test.pq")

    df.to_parquet(filename, write_covering_bbox=True)

    result = pq.read_table(filename).to_pandas()
    assert result["bbox"][0] == expected_bbox


def test_read_parquet_bbox_single_point(tmpdir):
    # confirm that on a single point, bbox will pick it up.
    df = GeoDataFrame(data=[[1, 2]], columns=["a", "b"], geometry=[Point(1, 1)])
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)
    pq_df = read_parquet(filename, bbox=(1, 1, 1, 1))
    assert len(pq_df) == 1
    assert pq_df.geometry[0] == Point(1, 1)


@pytest.mark.parametrize("geometry_name", ["geometry", "custum_geom_col"])
def test_read_parquet_bbox(tmpdir, naturalearth_lowres, geometry_name):
    # check bbox is being used to filter results.
    df = read_file(naturalearth_lowres)
    if geometry_name != "geometry":
        df = df.rename_geometry(geometry_name)

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)

    pq_df = read_parquet(filename, bbox=(0, 0, 10, 10))

    assert pq_df["name"].values.tolist() == [
        "France",
        "Benin",
        "Nigeria",
        "Cameroon",
        "Togo",
        "Ghana",
        "Burkina Faso",
        "Gabon",
        "Eq. Guinea",
    ]


@pytest.mark.parametrize("geometry_name", ["geometry", "custum_geom_col"])
def test_read_parquet_bbox_partitioned(tmpdir, naturalearth_lowres, geometry_name):
    # check bbox is being used to filter results on partioned data.
    df = read_file(naturalearth_lowres)
    if geometry_name != "geometry":
        df = df.rename_geometry(geometry_name)

    # manually create partitioned dataset
    basedir = tmpdir / "partitioned_dataset"
    basedir.mkdir()
    df[:100].to_parquet(basedir / "data1.parquet", write_covering_bbox=True)
    df[100:].to_parquet(basedir / "data2.parquet", write_covering_bbox=True)

    pq_df = read_parquet(basedir, bbox=(0, 0, 10, 10))

    assert pq_df["name"].values.tolist() == [
        "France",
        "Benin",
        "Nigeria",
        "Cameroon",
        "Togo",
        "Ghana",
        "Burkina Faso",
        "Gabon",
        "Eq. Guinea",
    ]


@pytest.mark.parametrize(
    "geometry, bbox",
    [
        (LineString([(1, 1), (3, 3)]), (1.5, 1.5, 3.5, 3.5)),
        (LineString([(1, 1), (3, 3)]), (3, 3, 3, 3)),
        (LineString([(1, 1), (3, 3)]), (1.5, 1.5, 2.5, 2.5)),
        (Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), (1, 1, 3, 3)),
        (Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), (1, 1, 5, 5)),
        (Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), (2, 2, 4, 4)),
        (Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), (4, 4, 4, 4)),
        (Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), (1, 1, 5, 3)),
    ],
)
def test_read_parquet_bbox_partial_overlap_of_geometry(tmpdir, geometry, bbox):
    df = GeoDataFrame(data=[[1, 2]], columns=["a", "b"], geometry=[geometry])
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)

    pq_df = read_parquet(filename, bbox=bbox)
    assert len(pq_df) == 1


def test_read_parquet_no_bbox(tmpdir, naturalearth_lowres):
    # check error message when parquet lacks a bbox column but
    # want to use bbox kwarg in read_parquet.
    df = read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)
    with pytest.raises(ValueError, match="Specifying 'bbox' not supported"):
        read_parquet(filename, bbox=(0, 0, 20, 20))


def test_read_parquet_no_bbox_partitioned(tmpdir, naturalearth_lowres):
    # check error message when partitioned parquet data does not have
    # a bbox column but want to use kwarg to read_parquet.
    df = read_file(naturalearth_lowres)

    # manually create partitioned dataset
    basedir = tmpdir / "partitioned_dataset"
    basedir.mkdir()
    df[:100].to_parquet(basedir / "data1.parquet")
    df[100:].to_parquet(basedir / "data2.parquet")

    with pytest.raises(ValueError, match="Specifying 'bbox' not supported"):
        read_parquet(basedir, bbox=(0, 0, 20, 20))


def test_convert_bbox_to_parquet_filter():
    # check conversion of bbox to parquet filter expression
    import pyarrow.compute as pc

    bbox = (0, 0, 25, 35)
    expected = ~(
        (pc.field(("bbox", "xmin")) > 25)
        | (pc.field(("bbox", "ymin")) > 35)
        | (pc.field(("bbox", "xmax")) < 0)
        | (pc.field(("bbox", "ymax")) < 0)
    )
    assert expected.equals(_convert_bbox_to_parquet_filter(bbox, "bbox"))


def test_read_parquet_bbox_column_default_behaviour(tmpdir, naturalearth_lowres):
    # check that bbox column is not read in by default

    df = read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)
    result1 = read_parquet(filename)
    assert "bbox" not in result1

    result2 = read_parquet(filename, columns=["name", "geometry"])
    assert "bbox" not in result2
    assert list(result2.columns) == ["name", "geometry"]


@pytest.mark.parametrize(
    "filters",
    [
        [("gdp_md_est", ">", 20000)],
        pc.field("gdp_md_est") > 20000,
    ],
)
def test_read_parquet_filters_and_bbox(tmpdir, naturalearth_lowres, filters):
    df = read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)

    result = read_parquet(filename, filters=filters, bbox=(0, 0, 20, 20))
    assert result["name"].values.tolist() == [
        "Dem. Rep. Congo",
        "France",
        "Nigeria",
        "Cameroon",
        "Ghana",
        "Algeria",
        "Libya",
    ]


@pytest.mark.parametrize(
    "filters",
    [
        ([("gdp_md_est", ">", 15000), ("gdp_md_est", "<", 16000)]),
        ((pc.field("gdp_md_est") > 15000) & (pc.field("gdp_md_est") < 16000)),
    ],
)
def test_read_parquet_filters_without_bbox(tmpdir, naturalearth_lowres, filters):
    df = read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, write_covering_bbox=True)

    result = read_parquet(filename, filters=filters)
    assert result["name"].values.tolist() == ["Burkina Faso", "Mozambique", "Albania"]


def test_read_parquet_file_with_custom_bbox_encoding_fieldname(tmpdir):
    import pyarrow.parquet as pq

    data = {
        "name": ["point1", "point2", "point3"],
        "geometry": [Point(1, 1), Point(2, 2), Point(3, 3)],
    }
    df = GeoDataFrame(data)
    filename = os.path.join(str(tmpdir), "test.pq")

    table = _geopandas_to_arrow(
        df,
        schema_version="1.1.0",
        write_covering_bbox=True,
    )
    metadata = table.schema.metadata  # rename_columns results in wiping of metadata

    table = table.rename_columns(["name", "geometry", "custom_bbox_name"])

    geo_metadata = json.loads(metadata[b"geo"])
    geo_metadata["columns"]["geometry"]["covering"]["bbox"] = {
        "xmin": ["custom_bbox_name", "xmin"],
        "ymin": ["custom_bbox_name", "ymin"],
        "xmax": ["custom_bbox_name", "xmax"],
        "ymax": ["custom_bbox_name", "ymax"],
    }
    metadata.update({b"geo": _encode_metadata(geo_metadata)})

    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, filename)

    pq_table = pq.read_table(filename)
    assert "custom_bbox_name" in pq_table.schema.names

    pq_df = read_parquet(filename, bbox=(1.5, 1.5, 2.5, 2.5))
    assert pq_df["name"].values.tolist() == ["point2"]


def test_to_parquet_with_existing_bbox_column(tmpdir, naturalearth_lowres):
    df = read_file(naturalearth_lowres)
    df = df.assign(bbox=[0] * len(df))
    filename = os.path.join(str(tmpdir), "test.pq")

    with pytest.raises(
        ValueError, match="An existing column 'bbox' already exists in the dataframe"
    ):
        df.to_parquet(filename, write_covering_bbox=True)


def test_read_parquet_bbox_points(tmp_path):
    # check bbox filtering on point geometries
    df = geopandas.GeoDataFrame(
        {"col": range(10)}, geometry=[Point(i, i) for i in range(10)]
    )
    df.to_parquet(tmp_path / "test.parquet", geometry_encoding="geoarrow")

    result = geopandas.read_parquet(tmp_path / "test.parquet", bbox=(0, 0, 10, 10))
    assert len(result) == 10
    result = geopandas.read_parquet(tmp_path / "test.parquet", bbox=(3, 3, 5, 5))
    assert len(result) == 3


def test_non_geo_parquet_read_with_proper_error(tmp_path):
    # https://github.com/geopandas/geopandas/issues/3556

    gdf = geopandas.GeoDataFrame(
        {"col": [1, 2, 3]},
        geometry=geopandas.points_from_xy([1, 2, 3], [1, 2, 3]),
        crs="EPSG:4326",
    )
    del gdf["geometry"]

    gdf.to_parquet(tmp_path / "test_no_geometry.parquet")
    with pytest.raises(
        ValueError, match="No geometry columns are included in the columns read"
    ):
        geopandas.read_parquet(tmp_path / "test_no_geometry.parquet")
