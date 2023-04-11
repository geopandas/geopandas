from __future__ import absolute_import

from itertools import product
import json
from packaging.version import Version
import os
import pathlib

import pytest
from pandas import DataFrame, read_parquet as pd_read_parquet
from pandas.testing import assert_frame_equal
import numpy as np
import pyproj
from shapely.geometry import box, Point, MultiPolygon


import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_parquet, read_feather
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.arrow import (
    SUPPORTED_VERSIONS,
    _create_metadata,
    _decode_metadata,
    _encode_metadata,
    _geopandas_to_arrow,
    _get_filesystem_path,
    _remove_id_from_member_of_ensembles,
    _validate_dataframe,
    _validate_metadata,
    METADATA_VERSION,
)
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock


DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / "data"


# Skip all tests in this module if pyarrow is not available
pyarrow = pytest.importorskip("pyarrow")


@pytest.fixture(
    params=[
        "parquet",
        pytest.param(
            "feather",
            marks=pytest.mark.skipif(
                Version(pyarrow.__version__) < Version("0.17.0"),
                reason="needs pyarrow >= 0.17",
            ),
        ),
    ]
)
def file_format(request):
    if request.param == "parquet":
        return read_parquet, GeoDataFrame.to_parquet
    elif request.param == "feather":
        return read_feather, GeoDataFrame.to_feather


def test_create_metadata():
    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    metadata = _create_metadata(df)

    assert isinstance(metadata, dict)
    assert metadata["version"] == METADATA_VERSION
    assert metadata["primary_column"] == "geometry"
    assert "geometry" in metadata["columns"]
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


def test_crs_metadata_datum_ensemble():
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


def test_write_metadata_invalid_spec_version():
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="schema_version must be one of"):
        _create_metadata(gdf, schema_version="invalid")


def test_encode_metadata():
    metadata = {"a": "b"}

    expected = b'{"a": "b"}'
    assert _encode_metadata(metadata) == expected


def test_decode_metadata():
    metadata_str = b'{"a": "b"}'

    expected = {"a": "b"}
    assert _decode_metadata(metadata_str) == expected

    assert _decode_metadata(None) is None


def test_validate_dataframe():
    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

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


def test_validate_metadata_valid():
    _validate_metadata(
        {
            "primary_column": "geometry",
            "columns": {"geometry": {"crs": None, "encoding": "WKB"}},
            "schema_version": "0.1.0",
        }
    )

    _validate_metadata(
        {
            "primary_column": "geometry",
            "columns": {"geometry": {"crs": None, "encoding": "WKB"}},
            "version": "<version>",
        }
    )

    _validate_metadata(
        {
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "crs": {
                        # truncated PROJJSON for testing, as PROJJSON contents
                        # not validated here
                        "id": {"authority": "EPSG", "code": 4326},
                    },
                    "encoding": "WKB",
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
            "Only WKB geometry encoding is supported",
        ),
        (
            {
                "primary_column": "foo",
                "columns": {"foo": {"crs": None, "encoding": "BKW"}},
                "version": "<version>",
            },
            "Only WKB geometry encoding is supported",
        ),
    ],
)
def test_validate_metadata_invalid(metadata, error):
    with pytest.raises(ValueError, match=error):
        _validate_metadata(metadata)


def test_validate_metadata_edges():
    metadata = {
        "primary_column": "geometry",
        "columns": {"geometry": {"crs": None, "encoding": "WKB", "edges": "spherical"}},
        "version": "1.0.0-beta.1",
    }
    with pytest.warns(
        UserWarning,
        match="The geo metadata indicate that column 'geometry' has spherical edges",
    ):
        _validate_metadata(metadata)


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
        df, "", compression="snappy", index=None, schema_version=None
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
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_pandas_parquet_roundtrip2(test_dataset, tmpdir):
    test_dataset = "naturalearth_lowres"
    df = DataFrame(read_file(get_path(test_dataset)).drop(columns=["geometry"]))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename)

    pq_df = pd_read_parquet(filename)

    assert_frame_equal(df, pq_df)


@pytest.mark.parametrize(
    "test_dataset", ["naturalearth_lowres", "naturalearth_cities", "nybb"]
)
def test_roundtrip(tmpdir, file_format, test_dataset):
    """Writing to parquet should not raise errors, and should not alter original
    GeoDataFrame
    """
    reader, writer = file_format

    df = read_file(get_path(test_dataset))
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


def test_index(tmpdir, file_format):
    """Setting index=`True` should preserve index in output, and
    setting index=`False` should drop index from output.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset)).set_index("iso_a3")

    filename = os.path.join(str(tmpdir), "test_with_index.pq")
    writer(df, filename, index=True)
    pq_df = reader(filename)
    assert_geodataframe_equal(df, pq_df)

    filename = os.path.join(str(tmpdir), "drop_index.pq")
    writer(df, filename, index=False)
    pq_df = reader(filename)
    assert_geodataframe_equal(df.reset_index(drop=True), pq_df)


@pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", None])
def test_parquet_compression(compression, tmpdir):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    df.to_parquet(filename, compression=compression)
    pq_df = read_parquet(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


@pytest.mark.skipif(
    Version(pyarrow.__version__) < Version("0.17.0"),
    reason="Feather only supported for pyarrow >= 0.17",
)
@pytest.mark.parametrize("compression", ["uncompressed", "lz4", "zstd"])
def test_feather_compression(compression, tmpdir):
    """Using compression options should not raise errors, and should
    return identical GeoDataFrame.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.feather")
    df.to_feather(filename, compression=compression)
    pq_df = read_feather(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)


def test_parquet_multiple_geom_cols(tmpdir, file_format):
    """If multiple geometry columns are present when written to parquet,
    they should all be returned as such when read from parquet.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
    df["geom2"] = df.geometry.copy()

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)

    assert os.path.exists(filename)

    pq_df = reader(filename)

    assert isinstance(pq_df, GeoDataFrame)
    assert_geodataframe_equal(df, pq_df)

    assert_geoseries_equal(df.geom2, pq_df.geom2, check_geom_type=True)


def test_parquet_missing_metadata(tmpdir):
    """Missing geo metadata, such as from a parquet file created
    from a pandas DataFrame, will raise a ValueError.
    """

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

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
def test_parquet_invalid_metadata(tmpdir, geo_meta, error):
    """Has geo metadata with missing required fields will raise a ValueError.

    This requires writing the parquet file directly below, so that we can
    control the metadata that is written for this test.
    """

    from pyarrow import parquet, Table

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

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


def test_subset_columns(tmpdir, file_format):
    """Reading a subset of columns should correctly decode selected geometry
    columns.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)
    pq_df = reader(filename, columns=["name", "geometry"])

    assert_geodataframe_equal(df[["name", "geometry"]], pq_df)

    with pytest.raises(
        ValueError, match="No geometry columns are included in the columns read"
    ):
        reader(filename, columns=["name"])


def test_promote_secondary_geometry(tmpdir, file_format):
    """Reading a subset of columns that does not include the primary geometry
    column should promote the first geometry column present.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))
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


def test_columns_no_geometry(tmpdir, file_format):
    """Reading a parquet file that is missing all of the geometry columns
    should raise a ValueError"""
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    filename = os.path.join(str(tmpdir), "test.pq")
    writer(df, filename)

    with pytest.raises(ValueError):
        reader(filename, columns=["name"])


def test_missing_crs(tmpdir, file_format):
    """If CRS is `None`, it should be properly handled
    and remain `None` when read from parquet`.
    """
    reader, writer = file_format

    test_dataset = "naturalearth_lowres"

    df = read_file(get_path(test_dataset))
    df.crs = None

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


@pytest.mark.skipif(
    Version(pyarrow.__version__) >= Version("0.17.0"),
    reason="Feather only supported for pyarrow >= 0.17",
)
def test_feather_arrow_version(tmpdir):
    df = read_file(get_path("naturalearth_lowres"))
    filename = os.path.join(str(tmpdir), "test.feather")

    with pytest.raises(
        ImportError, match="pyarrow >= 0.17 required for Feather support"
    ):
        df.to_feather(filename)


def test_fsspec_url():
    fsspec = pytest.importorskip("fsspec")
    import fsspec.implementations.memory

    class MyMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
        # Simple fsspec filesystem that adds a required keyword.
        # Attempting to use this filesystem without the keyword will raise an exception.
        def __init__(self, is_set, *args, **kwargs):
            self.is_set = is_set
            super().__init__(*args, **kwargs)

    fsspec.register_implementation("memory", MyMemoryFileSystem, clobber=True)
    memfs = MyMemoryFileSystem(is_set=True)

    test_dataset = "naturalearth_lowres"
    df = read_file(get_path(test_dataset))

    with memfs.open("data.parquet", "wb") as f:
        df.to_parquet(f)

    result = read_parquet("memory://data.parquet", storage_options=dict(is_set=True))
    assert_geodataframe_equal(result, df)

    result = read_parquet("memory://data.parquet", filesystem=memfs)
    assert_geodataframe_equal(result, df)

    # reset fsspec registry
    fsspec.register_implementation(
        "memory", fsspec.implementations.memory.MemoryFileSystem, clobber=True
    )


def test_non_fsspec_url_with_storage_options_raises():
    with pytest.raises(ValueError, match="storage_options"):
        test_dataset = "naturalearth_lowres"
        read_parquet(get_path(test_dataset), storage_options={"foo": "bar"})


@pytest.mark.skipif(
    Version(pyarrow.__version__) < Version("5.0.0"),
    reason="pyarrow.fs requires pyarrow>=5.0.0",
)
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
def test_write_read_default_crs(tmpdir, format):
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


def test_write_iso_wkb(tmpdir):
    gdf = geopandas.GeoDataFrame(
        geometry=geopandas.GeoSeries.from_wkt(["POINT Z (1 2 3)"])
    )
    if compat.USE_SHAPELY_20:
        gdf.to_parquet(tmpdir / "test.parquet")
    else:
        with pytest.warns(UserWarning, match="The GeoDataFrame contains 3D geometries"):
            gdf.to_parquet(tmpdir / "test.parquet")

    from pyarrow.parquet import read_table

    table = read_table(tmpdir / "test.parquet")
    wkb = table["geometry"][0].as_py().hex()

    if compat.USE_SHAPELY_20:
        # correct ISO flavor
        assert wkb == "01e9030000000000000000f03f00000000000000400000000000000840"
    else:
        assert wkb == "0101000080000000000000f03f00000000000000400000000000000840"


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


@pytest.mark.parametrize(
    "format,version", product(["feather", "parquet"], [None] + SUPPORTED_VERSIONS)
)
def test_write_deprecated_version_parameter(tmpdir, format, version):
    if format == "feather":
        from pyarrow.feather import read_table

        version = version or 2

    else:
        from pyarrow.parquet import read_table

        version = version or "2.6"

    filename = os.path.join(str(tmpdir), f"test.{format}")
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    write = getattr(gdf, f"to_{format}")

    if version in SUPPORTED_VERSIONS:
        with pytest.warns(
            FutureWarning,
            match="the `version` parameter has been replaced with `schema_version`",
        ):
            write(filename, version=version)

    else:
        # no warning raised if not one of the captured versions
        write(filename, version=version)

    table = read_table(filename)
    metadata = json.loads(table.schema.metadata[b"geo"])

    if version in SUPPORTED_VERSIONS:
        # version is captured as a parameter
        assert metadata["version"] == version
    else:
        # version is passed to underlying writer
        assert metadata["version"] == METADATA_VERSION


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
    df.to_feather(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.feather')  # noqa: E501
    df.to_parquet(DATA_PATH / 'arrow' / f'test_data_v{METADATA_VERSION}.parquet')  # noqa: E501
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
    $ ogr2ogr -f Arrow -lco FID= -lco GEOMETRY_ENCODING=WKB test_data_gdal350.arrow test_data.gpkg  # noqa: E501
    """
    expected = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)],
        crs="EPSG:4326",
    )

    df = geopandas.read_parquet(DATA_PATH / "arrow" / "test_data_gdal350.parquet")
    assert_geodataframe_equal(df, expected, check_crs=True)

    df = geopandas.read_feather(DATA_PATH / "arrow" / "test_data_gdal350.arrow")
    assert_geodataframe_equal(df, expected, check_crs=True)


def test_parquet_read_partitioned_dataset(tmpdir):
    # we don't yet explicitly support this (in writing), but for Parquet it
    # works for reading (by relying on pyarrow.read_table)
    df = read_file(get_path("naturalearth_lowres"))

    # manually create partitioned dataset
    basedir = tmpdir / "partitioned_dataset"
    basedir.mkdir()
    df[:100].to_parquet(basedir / "data1.parquet")
    df[100:].to_parquet(basedir / "data2.parquet")

    result = read_parquet(basedir)
    assert_geodataframe_equal(result, df)


def test_parquet_read_partitioned_dataset_fsspec(tmpdir):
    fsspec = pytest.importorskip("fsspec")

    df = read_file(get_path("naturalearth_lowres"))

    # manually create partitioned dataset
    memfs = fsspec.filesystem("memory")
    memfs.mkdir("partitioned_dataset")
    with memfs.open("partitioned_dataset/data1.parquet", "wb") as f:
        df[:100].to_parquet(f)
    with memfs.open("partitioned_dataset/data2.parquet", "wb") as f:
        df[100:].to_parquet(f)

    result = read_parquet("memory://partitioned_dataset")
    assert_geodataframe_equal(result, df)
