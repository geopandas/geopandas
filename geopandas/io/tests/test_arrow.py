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
    assert metadata["columns"]["geometry"]["crs"] == df.geometry.crs.to_json_dict()
    assert metadata["columns"]["geometry"]["encoding"] == "WKB"
    assert metadata["columns"]["geometry"]["geometry_type"] == [
        "MultiPolygon",
        "Polygon",
    ]

    assert np.array_equal(
        metadata["columns"]["geometry"]["bbox"], df.geometry.total_bounds
    )

    assert metadata["creator"]["library"] == "geopandas"
    assert metadata["creator"]["version"] == geopandas.__version__


def test_write_metadata_invalid_spec_version():
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="version must be one of"):
        _create_metadata(gdf, version="invalid")


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
        df, "", compression="snappy", index=None, version=None
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


@pytest.mark.parametrize(
    "format,version", product(["feather", "parquet"], [None] + SUPPORTED_VERSIONS)
)
def test_write_spec_version(tmpdir, format, version):
    if format == "feather":
        from pyarrow.feather import read_table

    else:
        from pyarrow.parquet import read_table

    filename = os.path.join(str(tmpdir), f"test.{format}")
    gdf = geopandas.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    write = getattr(gdf, f"to_{format}")
    write(filename, version=version)

    # ensure that we can roundtrip data regardless of version
    read = getattr(geopandas, f"read_{format}")
    df = read(filename)
    assert_geodataframe_equal(df, gdf)

    table = read_table(filename)
    metadata = json.loads(table.schema.metadata[b"geo"])
    assert metadata["version"] == version or METADATA_VERSION

    # verify that CRS is correctly handled between versions
    if version == "0.1.0":
        assert metadata["columns"]["geometry"]["crs"] == gdf.crs.to_wkt()

    else:
        assert metadata["columns"]["geometry"]["crs"] == gdf.crs.to_json_dict()


@pytest.mark.parametrize("version", ["0.1.0", "0.4.0"])
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
    check_crs = Version(pyproj.__version__) >= Version("3.0.0")

    expected = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)],
        crs="EPSG:4326",
    )

    df = geopandas.read_feather(DATA_PATH / "arrow" / f"test_data_v{version}.feather")
    assert_geodataframe_equal(df, expected, check_crs=check_crs)

    df = geopandas.read_parquet(DATA_PATH / "arrow" / f"test_data_v{version}.parquet")
    assert_geodataframe_equal(df, expected, check_crs=check_crs)


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
    check_crs = Version(pyproj.__version__) >= Version("3.0.0")

    expected = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)],
        crs="EPSG:4326",
    )

    df = geopandas.read_parquet(DATA_PATH / "arrow" / "test_data_gdal350.parquet")
    assert_geodataframe_equal(df, expected, check_crs=check_crs)

    df = geopandas.read_feather(DATA_PATH / "arrow" / "test_data_gdal350.arrow")
    assert_geodataframe_equal(df, expected, check_crs=check_crs)
